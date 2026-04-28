/*
 * Fused TX+RX SIMUS kernel -- v19: fp16 half2 cmul chains.
 *
 * Based on v15. The cmul dependency chain (dominant bottleneck per ncu:
 * wait stall = 0.61) is converted to fp16 half2 operations via inline PTX.
 *
 * Precision strategy:
 *   - Geometry (Phase 1): fp32
 *   - TX sweep (Phase 2): cv/sv as packed half2, accumulation in fp32
 *   - RX sweep (Phase 3): cv/sv as packed half2, accumulation in fp32
 *   - Output: fp32 atomicAdd
 *
 * half2 is represented as unsigned int (32 bits = two fp16 values).
 * All half2 intrinsics use inline PTX (no cuda_fp16.h needed for NVRTC).
 *
 * ELEM_TILE must be even (for half2 pairing).
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, TILE_SE, TG_SIZE, MAX_FPT,
 *               B_SCAT, ELEM_TILE
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/* ===== half2 as unsigned int, all ops via inline PTX ===== */
typedef unsigned int h2_t;

__device__ __forceinline__ h2_t h2_pack(float a, float b) {
    h2_t r;
    asm("{ .reg .f16 ha, hb;\n\t"
        "  cvt.rn.f16.f32 ha, %1;\n\t"
        "  cvt.rn.f16.f32 hb, %2;\n\t"
        "  mov.b32 %0, {ha, hb}; }"
        : "=r"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float h2_low(h2_t v) {
    float r;
    asm("{ .reg .f16 lo, hi;\n\t"
        "  mov.b32 {lo, hi}, %1;\n\t"
        "  cvt.f32.f16 %0, lo; }"
        : "=f"(r) : "r"(v));
    return r;
}

__device__ __forceinline__ float h2_high(h2_t v) {
    float r;
    asm("{ .reg .f16 lo, hi;\n\t"
        "  mov.b32 {lo, hi}, %1;\n\t"
        "  cvt.f32.f16 %0, hi; }"
        : "=f"(r) : "r"(v));
    return r;
}

__device__ __forceinline__ h2_t h2_mul(h2_t a, h2_t b) {
    h2_t r;
    asm("mul.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ h2_t h2_add(h2_t a, h2_t b) {
    h2_t r;
    asm("add.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ h2_t h2_sub(h2_t a, h2_t b) {
    h2_t r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ h2_t h2_fma(h2_t a, h2_t b, h2_t c) {
    h2_t r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

/* h2_cmul: complex multiply on two pairs simultaneously.
 * a_re, a_im, b_re, b_im each hold two fp16 values.
 * out_re = a_re*b_re - a_im*b_im
 * out_im = a_re*b_im + a_im*b_re */
__device__ __forceinline__ void h2_cmul(
    h2_t& out_re, h2_t& out_im,
    h2_t a_re, h2_t a_im, h2_t b_re, h2_t b_im)
{
    out_re = h2_sub(h2_mul(a_re, b_re), h2_mul(a_im, b_im));
    out_im = h2_add(h2_mul(a_re, b_im), h2_mul(a_im, b_re));
}

struct f2 { float x, y; };

__device__ __forceinline__ f2 cmul(f2 a, f2 b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

#define GEO_AMP(s)       (shmem + ((0*B_SCAT + (s)) * N_ES))
#define GEO_KW_R(s)      (shmem + ((1*B_SCAT + (s)) * N_ES))
#define GEO_KR_STEP(s)   (shmem + ((2*B_SCAT + (s)) * N_ES))
#define GEO_ALPHA_R(s)   (shmem + ((3*B_SCAT + (s)) * N_ES))
#define GEO_AR_STEP(s)   (shmem + ((4*B_SCAT + (s)) * N_ES))
#define GEO_STP_RX_RE(s) (shmem + ((5*B_SCAT + (s)) * N_ES))
#define GEO_STP_RX_IM(s) (shmem + ((6*B_SCAT + (s)) * N_ES))

__device__ __forceinline__ unsigned short f2h(float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}
__device__ __forceinline__ float h2f(unsigned short h) {
    float v;
    asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h));
    return v;
}

#define SH_TX_HALF_OFFSET (7 * B_SCAT * N_ES)
#define SH_TX_RE_HALF(ptr) ((unsigned short*)((ptr) + SH_TX_HALF_OFFSET))
#define SH_TX_IM_HALF(ptr) ((unsigned short*)((ptr) + SH_TX_HALF_OFFSET) + B_SCAT * N_FREQ)

#define TX_HALF_FLOATS ((2 * B_SCAT * N_FREQ + 1) / 2)
#define SH_DELAY_OFFSET (SH_TX_HALF_OFFSET + TX_HALF_FLOATS)

#define ET_PAIRS (ELEM_TILE / 2)
#define TS_PAIRS (TILE_SE / 2)

extern "C" __global__
void simus_fused_kernel(
    const float* __restrict__ scat_x,
    const float* __restrict__ scat_z,
    const float* __restrict__ rc_arr,
    const float* __restrict__ elem_x,
    const float* __restrict__ elem_z,
    const float* __restrict__ cos_te,
    const float* __restrict__ sin_neg_te,
    const float* __restrict__ sub_dx,
    const float* __restrict__ sub_dz,
    const float* __restrict__ da_init_re,
    const float* __restrict__ da_init_im,
    const float* __restrict__ dps,
    const float* __restrict__ pp_re,
    const float* __restrict__ pp_im,
    const float* __restrict__ probe,
    float* __restrict__ spect_re,
    float* __restrict__ spect_im,
    int   n_scat,
    float kw_init, float alpha_init,
    float kw_step, float alpha_step,
    float min_dist, float seg_len,
    float center_kw, float inv_nsub,
    float radius, float apex_offset
) {
    int lid = threadIdx.x;
    float lid_f = (float)lid;
    float stride_f = (float)TG_SIZE;

    extern __shared__ float shmem[];

    unsigned short* sh_tx_re = SH_TX_RE_HALF(shmem);
    unsigned short* sh_tx_im = SH_TX_IM_HALF(shmem);
    float* sh_da_init_re_l = shmem + SH_DELAY_OFFSET;
    float* sh_da_init_im_l = sh_da_init_re_l + N_ELEM;
    float* sh_dps_l        = sh_da_init_im_l + N_ELEM;

    for (int e = lid; e < N_ELEM; e += TG_SIZE) {
        sh_da_init_re_l[e] = da_init_re[e];
        sh_da_init_im_l[e] = da_init_im[e];
        sh_dps_l[e]        = dps[e];
    }
    __syncthreads();

    int my_n_freq = 0;
    for (int f = lid; f < N_FREQ; f += TG_SIZE) my_n_freq++;

    const int N_TILES = (N_ES + TILE_SE - 1) / TILE_SE;
    const int N_ELEM_GROUPS = (N_ES + ELEM_TILE - 1) / ELEM_TILE;

    bool out_flag[B_SCAT];

    for (int scat_base = blockIdx.x * B_SCAT;
         scat_base < n_scat;
         scat_base += gridDim.x * B_SCAT)
    {
        int actual_b = B_SCAT;
        if (scat_base + B_SCAT > n_scat)
            actual_b = n_scat - scat_base;

        for (int si = 0; si < actual_b; si++) {
            int scat_idx = scat_base + si;
            float sx = scat_x[scat_idx];
            float sz = scat_z[scat_idx];
            float rc = rc_arr[scat_idx];

            bool is_out = (sz < 0.0f);
            if (radius < 1e30f) {
                float da = sx, db = sz + apex_offset;
                is_out = is_out || ((da*da + db*db) <= radius*radius);
            }
            out_flag[si] = is_out;

            for (int se = lid; se < N_ES; se += TG_SIZE) {
                int elem = se / N_SUB;
                float ex_ = elem_x[elem], ez_ = elem_z[elem];
                float ct = cos_te[elem], snt = sin_neg_te[elem];
                float dx = sx - ex_ - sub_dx[se];
                float dz = sz - ez_ - sub_dz[se];
                float r2 = dx*dx + dz*dz;
                float inv_r = rsqrtf(r2 + 1e-30f);
                float r = r2 * inv_r;
                float rc_ = fmaxf(r, min_dist);

                float sin_th = (dx*ct + dz*snt) * inv_r;
                float cos_th = (dz*ct - dx*snt) * inv_r;
                float obliq = (cos_th <= 0.0f) ? 1e-16f : cos_th;
                float sa = center_kw * seg_len * 0.5f * sin_th;
                float sv = (fabsf(sa) < 1e-8f) ? 1.0f : __fdividef(__sinf(sa), sa);

                GEO_AMP(si)[se]       = obliq * sv * rsqrtf(rc_);
                GEO_KW_R(si)[se]      = kw_init * rc_;
                GEO_KR_STEP(si)[se]   = kw_step * rc_;
                GEO_ALPHA_R(si)[se]   = alpha_init * rc_;
                GEO_AR_STEP(si)[se]   = alpha_step * rc_;

                float stp_phase = stride_f * kw_step * rc_;
                float stp_alpha = stride_f * alpha_step * rc_;
                float sm = expf(-stp_alpha);
                float sp_re, sp_im;
                __sincosf(stp_phase, &sp_im, &sp_re);
                sp_re *= sm; sp_im *= sm;
                GEO_STP_RX_RE(si)[se] = sp_re;
                GEO_STP_RX_IM(si)[se] = sp_im;
            }
            __syncthreads();

            if (is_out) {
                for (int f = lid; f < N_FREQ; f += TG_SIZE) {
                    sh_tx_re[si * N_FREQ + f] = f2h(0.0f);
                    sh_tx_im[si * N_FREQ + f] = f2h(0.0f);
                }
                __syncthreads();
                continue;
            }

            /* Phase 2: TX sweep with half2 cmul chains. */
            float sum_re[MAX_FPT], sum_im[MAX_FPT];
            for (int i = 0; i < MAX_FPT; i++) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }

            for (int tile = 0; tile < N_TILES; tile++) {
                int ts = tile * TILE_SE;
                int te = ts + TILE_SE;
                if (te > N_ES) te = N_ES;
                int tl = te - ts;

                h2_t cv_re_h[TS_PAIRS], cv_im_h[TS_PAIRS];
                h2_t sv_re_h[TS_PAIRS], sv_im_h[TS_PAIRS];

                #pragma unroll
                for (int jp = 0; jp < TS_PAIRS; jp++) {
                    int j0 = jp * 2;
                    int j1 = j0 + 1;

                    float vr0=0, vi0=0, spr0=1, spi0=0;
                    float vr1=0, vi1=0, spr1=1, spi1=0;

                    if (j0 < tl) {
                        int se0 = ts+j0, em0 = se0/N_SUB;
                        float ph0 = GEO_KW_R(si)[se0] + lid_f*GEO_KR_STEP(si)[se0];
                        float av0 = GEO_ALPHA_R(si)[se0] + lid_f*GEO_AR_STEP(si)[se0];
                        float ai0 = GEO_AMP(si)[se0] * expf(-av0);
                        __sincosf(ph0, &vi0, &vr0);
                        vr0 *= ai0; vi0 *= ai0;
                        float dp0 = lid_f * sh_dps_l[em0];
                        float dr0, di0;
                        __sincosf(dp0, &di0, &dr0);
                        float dvr0 = sh_da_init_re_l[em0]*dr0 - sh_da_init_im_l[em0]*di0;
                        float dvi0 = sh_da_init_re_l[em0]*di0 + sh_da_init_im_l[em0]*dr0;
                        float cr0 = vr0*dvr0 - vi0*dvi0;
                        float ci0 = vr0*dvi0 + vi0*dvr0;
                        vr0 = cr0; vi0 = ci0;
                        spr0 = GEO_STP_RX_RE(si)[se0];
                        spi0 = GEO_STP_RX_IM(si)[se0];
                        float ds0 = stride_f * sh_dps_l[em0];
                        float dsr0, dsi0;
                        __sincosf(ds0, &dsi0, &dsr0);
                        float s0r = spr0*dsr0-spi0*dsi0, s0i = spr0*dsi0+spi0*dsr0;
                        spr0 = s0r; spi0 = s0i;
                    }
                    if (j1 < tl) {
                        int se1 = ts+j1, em1 = se1/N_SUB;
                        float ph1 = GEO_KW_R(si)[se1] + lid_f*GEO_KR_STEP(si)[se1];
                        float av1 = GEO_ALPHA_R(si)[se1] + lid_f*GEO_AR_STEP(si)[se1];
                        float ai1 = GEO_AMP(si)[se1] * expf(-av1);
                        __sincosf(ph1, &vi1, &vr1);
                        vr1 *= ai1; vi1 *= ai1;
                        float dp1 = lid_f * sh_dps_l[em1];
                        float dr1, di1;
                        __sincosf(dp1, &di1, &dr1);
                        float dvr1 = sh_da_init_re_l[em1]*dr1 - sh_da_init_im_l[em1]*di1;
                        float dvi1 = sh_da_init_re_l[em1]*di1 + sh_da_init_im_l[em1]*dr1;
                        float cr1 = vr1*dvr1 - vi1*dvi1;
                        float ci1 = vr1*dvi1 + vi1*dvr1;
                        vr1 = cr1; vi1 = ci1;
                        spr1 = GEO_STP_RX_RE(si)[se1];
                        spi1 = GEO_STP_RX_IM(si)[se1];
                        float ds1 = stride_f * sh_dps_l[em1];
                        float dsr1, dsi1;
                        __sincosf(ds1, &dsi1, &dsr1);
                        float s1r = spr1*dsr1-spi1*dsi1, s1i = spr1*dsi1+spi1*dsr1;
                        spr1 = s1r; spi1 = s1i;
                    }

                    cv_re_h[jp] = h2_pack(vr0, vr1);
                    cv_im_h[jp] = h2_pack(vi0, vi1);
                    sv_re_h[jp] = h2_pack(spr0, spr1);
                    sv_im_h[jp] = h2_pack(spi0, spi1);
                }

                for (int fi = 0; fi < my_n_freq; fi++) {
                    #pragma unroll
                    for (int jp = 0; jp < TS_PAIRS; jp++) {
                        sum_re[fi] += h2_low(cv_re_h[jp]) + h2_high(cv_re_h[jp]);
                        sum_im[fi] += h2_low(cv_im_h[jp]) + h2_high(cv_im_h[jp]);
                        h2_cmul(cv_re_h[jp], cv_im_h[jp],
                                cv_re_h[jp], cv_im_h[jp],
                                sv_re_h[jp], sv_im_h[jp]);
                    }
                }
            }

            {
                int fi = 0;
                for (int f = lid; f < N_FREQ; f += TG_SIZE, fi++) {
                    float tr = sum_re[fi] * inv_nsub;
                    float ti = sum_im[fi] * inv_nsub;
                    float ppr = pp_re[f], ppi = pp_im[f];
                    sh_tx_re[si * N_FREQ + f] = f2h((ppr*tr - ppi*ti) * rc);
                    sh_tx_im[si * N_FREQ + f] = f2h((ppr*ti + ppi*tr) * rc);
                }
            }
            __syncthreads();
        }

        /* Phase 3: Element-tiled RX with half2 cmul chains. */
        for (int eg = 0; eg < N_ELEM_GROUPS; eg++) {
            int se_base = eg * ELEM_TILE;
            int etl = ELEM_TILE;
            if (se_base + etl > N_ES) etl = N_ES - se_base;

            h2_t cv_re3[B_SCAT * ET_PAIRS];
            h2_t cv_im3[B_SCAT * ET_PAIRS];
            h2_t sv_re3[B_SCAT * ET_PAIRS];
            h2_t sv_im3[B_SCAT * ET_PAIRS];

            #pragma unroll
            for (int si = 0; si < B_SCAT; si++) {
                #pragma unroll
                for (int ep = 0; ep < ET_PAIRS; ep++) {
                    int et0 = ep*2, et1 = et0+1;
                    int idx = si * ET_PAIRS + ep;
                    float vr0=0,vi0=0,spr0=1,spi0=0;
                    float vr1=0,vi1=0,spr1=1,spi1=0;

                    if (si < actual_b && !out_flag[si] && et0 < etl) {
                        int se0 = se_base+et0;
                        float ph0 = GEO_KW_R(si)[se0] + lid_f*GEO_KR_STEP(si)[se0];
                        float av0 = GEO_ALPHA_R(si)[se0] + lid_f*GEO_AR_STEP(si)[se0];
                        float ai0 = GEO_AMP(si)[se0] * expf(-av0);
                        __sincosf(ph0, &vi0, &vr0);
                        vr0 *= ai0; vi0 *= ai0;
                        spr0 = GEO_STP_RX_RE(si)[se0];
                        spi0 = GEO_STP_RX_IM(si)[se0];
                    }
                    if (si < actual_b && !out_flag[si] && et1 < etl) {
                        int se1 = se_base+et1;
                        float ph1 = GEO_KW_R(si)[se1] + lid_f*GEO_KR_STEP(si)[se1];
                        float av1 = GEO_ALPHA_R(si)[se1] + lid_f*GEO_AR_STEP(si)[se1];
                        float ai1 = GEO_AMP(si)[se1] * expf(-av1);
                        __sincosf(ph1, &vi1, &vr1);
                        vr1 *= ai1; vi1 *= ai1;
                        spr1 = GEO_STP_RX_RE(si)[se1];
                        spi1 = GEO_STP_RX_IM(si)[se1];
                    }

                    cv_re3[idx] = h2_pack(vr0, vr1);
                    cv_im3[idx] = h2_pack(vi0, vi1);
                    sv_re3[idx] = h2_pack(spr0, spr1);
                    sv_im3[idx] = h2_pack(spi0, spi1);
                }
            }

            for (int fi = 0; fi < my_n_freq; fi++) {
                int f = lid + fi * TG_SIZE;
                if (f >= N_FREQ) break;
                float pf = probe[f];

                float acc_re[ELEM_TILE];
                float acc_im[ELEM_TILE];
                #pragma unroll
                for (int et = 0; et < ELEM_TILE; et++) {
                    acc_re[et] = 0.0f; acc_im[et] = 0.0f;
                }

                #pragma unroll
                for (int si = 0; si < B_SCAT; si++) {
                    float tk_re = h2f(sh_tx_re[si * N_FREQ + f]);
                    float tk_im = h2f(sh_tx_im[si * N_FREQ + f]);

                    #pragma unroll
                    for (int ep = 0; ep < ET_PAIRS; ep++) {
                        int idx = si * ET_PAIRS + ep;
                        int et0 = ep*2, et1 = et0+1;

                        float rr0 = h2_low(cv_re3[idx]) * inv_nsub;
                        float ri0 = h2_low(cv_im3[idx]) * inv_nsub;
                        float rr1 = h2_high(cv_re3[idx]) * inv_nsub;
                        float ri1 = h2_high(cv_im3[idx]) * inv_nsub;

                        acc_re[et0] += (tk_re*rr0 - tk_im*ri0) * pf;
                        acc_im[et0] += (tk_re*ri0 + tk_im*rr0) * pf;
                        acc_re[et1] += (tk_re*rr1 - tk_im*ri1) * pf;
                        acc_im[et1] += (tk_re*ri1 + tk_im*rr1) * pf;

                        h2_cmul(cv_re3[idx], cv_im3[idx],
                                cv_re3[idx], cv_im3[idx],
                                sv_re3[idx], sv_im3[idx]);
                    }
                }

                #pragma unroll
                for (int et = 0; et < ELEM_TILE; et++) {
                    if (et >= etl) break;
                    int elem = (se_base + et) / N_SUB;
                    atomicAdd(&spect_re[elem * N_FREQ + f], acc_re[et]);
                    atomicAdd(&spect_im[elem * N_FREQ + f], acc_im[et]);
                }
            }
        }

        __syncthreads();
    }
}
