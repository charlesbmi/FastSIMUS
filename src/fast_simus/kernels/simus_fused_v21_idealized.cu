/*
 * IDEALIZED v21 -- exp19 combined idealized ceiling variant.
 *
 * Combines CONSTFREQ + SINGLEELEM + no-atom sink.
 *
 * - Per-scatterer scalar registers for kw_r/kr_step/alpha_r/ar_step
 *   (derived once from elem[0]/sub[0]) -- CONSTFREQ simplification.
 * - Per-scatterer scalar shmem for AMP, STP_RX_RE, STP_RX_IM
 *   (no per-se variation; all threads race-store the same value) --
 *   SINGLEELEM simplification.
 * - Delay arrays collapse to 3 scalars.
 * - Phase 3 atomics replaced with dead-code-eliminated sink.
 *
 * Shmem layout:
 *   geo:   3 * B_SCAT              floats  (AMP + STP_RX_RE + STP_RX_IM)
 *   tx_re: B_SCAT * N_FREQ          ushort  (fp16 bits)
 *   tx_im: B_SCAT * N_FREQ          ushort  (fp16 bits)
 *   delay: 3                        floats
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct f2 { float x, y; };

__device__ __forceinline__ f2 cmul(f2 a, f2 b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

#define GEO_AMP(s)       (shmem + (0*B_SCAT + (s)))
#define GEO_STP_RX_RE(s) (shmem + (1*B_SCAT + (s)))
#define GEO_STP_RX_IM(s) (shmem + (2*B_SCAT + (s)))

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

#define SH_TX_HALF_OFFSET (3 * B_SCAT)
#define SH_TX_RE_HALF(ptr) ((unsigned short*)((ptr) + SH_TX_HALF_OFFSET))
#define SH_TX_IM_HALF(ptr) ((unsigned short*)((ptr) + SH_TX_HALF_OFFSET) + B_SCAT * N_FREQ)

#define TX_HALF_FLOATS ((2 * B_SCAT * N_FREQ + 1) / 2)
#define SH_DELAY_OFFSET (SH_TX_HALF_OFFSET + TX_HALF_FLOATS)

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
    float* sh_da_init_im_l = sh_da_init_re_l + 1;
    float* sh_dps_l        = sh_da_init_im_l + 1;

    if (lid == 0) {
        sh_da_init_re_l[0] = da_init_re[0];
        sh_da_init_im_l[0] = da_init_im[0];
        sh_dps_l[0]        = dps[0];
    }
    __syncthreads();

    int my_n_freq = 0;
    for (int f = lid; f < N_FREQ; f += TG_SIZE) my_n_freq++;

    const int N_TILES = (N_ES + TILE_SE - 1) / TILE_SE;
    const int N_ELEM_GROUPS = (N_ES + ELEM_TILE - 1) / ELEM_TILE;

    bool out_flag[B_SCAT];
    /* CONSTFREQ: per-scatterer scalar registers. */
    float kw_r_s[B_SCAT], kr_step_s[B_SCAT];
    float alpha_r_s[B_SCAT], ar_step_s[B_SCAT];

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

            /* Capture per-scatterer rc_0 from elem[0]/sub[0]. */
            {
                float ex0 = elem_x[0], ez0 = elem_z[0];
                float dx0 = sx - ex0 - sub_dx[0];
                float dz0 = sz - ez0 - sub_dz[0];
                float r20 = dx0*dx0 + dz0*dz0;
                float inv_r0 = rsqrtf(r20 + 1e-30f);
                float r0 = r20 * inv_r0;
                float rc_0 = fmaxf(r0, min_dist);
                kw_r_s[si]    = kw_init    * rc_0;
                kr_step_s[si] = kw_step    * rc_0;
                alpha_r_s[si] = alpha_init * rc_0;
                ar_step_s[si] = alpha_step * rc_0;
            }

            /* SINGLEELEM + CONSTFREQ: loop preserved, stores collapse. */
            for (int se = lid; se < N_ES; se += TG_SIZE) {
                float ex_ = elem_x[0], ez_ = elem_z[0];
                float ct = cos_te[0], snt = sin_neg_te[0];
                float dx = sx - ex_ - sub_dx[0];
                float dz = sz - ez_ - sub_dz[0];
                float r2 = dx*dx + dz*dz;
                float inv_r = rsqrtf(r2 + 1e-30f);
                float r = r2 * inv_r;
                float rc_ = fmaxf(r, min_dist);

                float sin_th = (dx*ct + dz*snt) * inv_r;
                float cos_th = (dz*ct - dx*snt) * inv_r;
                float obliq = (cos_th <= 0.0f) ? 1e-16f : cos_th;
                float sa = center_kw * seg_len * 0.5f * sin_th;
                float sv = (fabsf(sa) < 1e-8f) ? 1.0f : __fdividef(__sinf(sa), sa);

                GEO_AMP(si)[0] = obliq * sv * rsqrtf(rc_);

                float stp_phase = stride_f * kw_step * rc_;
                float stp_alpha = stride_f * alpha_step * rc_;
                float sm = expf(-stp_alpha);
                float sp_re, sp_im;
                __sincosf(stp_phase, &sp_im, &sp_re);
                sp_re *= sm; sp_im *= sm;
                GEO_STP_RX_RE(si)[0] = sp_re;
                GEO_STP_RX_IM(si)[0] = sp_im;
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

            float sum_re[MAX_FPT], sum_im[MAX_FPT];
            for (int i = 0; i < MAX_FPT; i++) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }

            for (int tile = 0; tile < N_TILES; tile++) {
                int ts = tile * TILE_SE;
                int te = ts + TILE_SE;
                if (te > N_ES) te = N_ES;
                int tl = te - ts;

                f2 cv[TILE_SE], sv[TILE_SE];
                #pragma unroll
                for (int j = 0; j < TILE_SE; j++) {
                    if (j >= tl) { cv[j] = {0.0f, 0.0f}; sv[j] = {1.0f, 0.0f}; continue; }
                    float ph = kw_r_s[si]    + lid_f * kr_step_s[si];
                    float av = alpha_r_s[si] + lid_f * ar_step_s[si];
                    float ai = GEO_AMP(si)[0] * expf(-av);
                    float vr, vi;
                    __sincosf(ph, &vi, &vr);
                    vr *= ai; vi *= ai;
                    float dp = lid_f * sh_dps_l[0];
                    float dr, di;
                    __sincosf(dp, &di, &dr);
                    float dvr = sh_da_init_re_l[0]*dr - sh_da_init_im_l[0]*di;
                    float dvi = sh_da_init_re_l[0]*di + sh_da_init_im_l[0]*dr;
                    cv[j] = {vr*dvr - vi*dvi, vr*dvi + vi*dvr};

                    float sp_re = GEO_STP_RX_RE(si)[0];
                    float sp_im = GEO_STP_RX_IM(si)[0];
                    float das_phase = stride_f * sh_dps_l[0];
                    float das_re, das_im;
                    __sincosf(das_phase, &das_im, &das_re);
                    sv[j] = {sp_re*das_re - sp_im*das_im, sp_re*das_im + sp_im*das_re};
                }

                for (int fi = 0; fi < my_n_freq; fi++) {
                    #pragma unroll
                    for (int j = 0; j < TILE_SE; j++) {
                        sum_re[fi] += cv[j].x; sum_im[fi] += cv[j].y;
                        cv[j] = cmul(cv[j], sv[j]);
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

        /* Phase 3 */
        for (int eg = 0; eg < N_ELEM_GROUPS; eg++) {
            int se_base = eg * ELEM_TILE;
            int etl = ELEM_TILE;
            if (se_base + etl > N_ES) etl = N_ES - se_base;

            f2 cv[B_SCAT * ELEM_TILE];
            f2 sv_arr[B_SCAT * ELEM_TILE];

            #pragma unroll
            for (int si = 0; si < B_SCAT; si++) {
                #pragma unroll
                for (int et = 0; et < ELEM_TILE; et++) {
                    int idx = si * ELEM_TILE + et;
                    if (si >= actual_b || out_flag[si] || et >= etl) {
                        cv[idx] = {0.0f, 0.0f};
                        sv_arr[idx] = {1.0f, 0.0f};
                        continue;
                    }
                    float ph = kw_r_s[si]    + lid_f * kr_step_s[si];
                    float av = alpha_r_s[si] + lid_f * ar_step_s[si];
                    float ai = GEO_AMP(si)[0] * expf(-av);
                    float vr, vi;
                    __sincosf(ph, &vi, &vr);
                    cv[idx] = {vr * ai, vi * ai};
                    sv_arr[idx] = {GEO_STP_RX_RE(si)[0], GEO_STP_RX_IM(si)[0]};
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
                    acc_re[et] = 0.0f;
                    acc_im[et] = 0.0f;
                }

                #pragma unroll
                for (int si = 0; si < B_SCAT; si++) {
                    float tk_re = h2f(sh_tx_re[si * N_FREQ + f]);
                    float tk_im = h2f(sh_tx_im[si * N_FREQ + f]);

                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        int idx = si * ELEM_TILE + et;
                        float rr = cv[idx].x * inv_nsub;
                        float ri = cv[idx].y * inv_nsub;
                        acc_re[et] += (tk_re*rr - tk_im*ri) * pf;
                        acc_im[et] += (tk_re*ri + tk_im*rr) * pf;
                        cv[idx] = cmul(cv[idx], sv_arr[idx]);
                    }
                }

                #pragma unroll
                for (int et = 0; et < ELEM_TILE; et++) {
                    if (et >= etl) break;
                    int elem = (se_base + et) / N_SUB;
                    // Sink to prevent dead-code elimination
                    if (acc_re[et] == -1e30f && acc_im[et] == -1e30f)
                        spect_re[elem * N_FREQ + f] = acc_re[et] + acc_im[et];
                }
            }
        }

        __syncthreads();
    }
}
