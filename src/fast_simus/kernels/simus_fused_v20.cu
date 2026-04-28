/*
 * Fused TX+RX SIMUS kernel -- v20: 2-deep cmul pipelining.
 *
 * Based on v15. The cmul dependency chain (wait stall = 0.61) is broken
 * by running 2 independent frequency chains in the Phase 3 inner loop.
 *
 * Instead of: cv[n+1] = cmul(cv[n], sv)  (serial, 8 cycles/step)
 * We use:     cv_even[n] = cmul(cv_even[n-1], sv2)  // even frequencies
 *             cv_odd[n]  = cmul(cv_odd[n-1],  sv2)  // odd frequencies
 * Where sv2 = cmul(sv, sv) is the double-step.
 *
 * The two chains are independent: the scheduler can interleave them,
 * hiding the 4-cycle FMA latency behind the second chain's computation.
 *
 * Phase 2 is unchanged from v15 (TILE_SE=16 already has enough ILP).
 * Only Phase 3 (RX sweep) is pipelined because it dominates runtime.
 *
 * Shmem layout: same as v15 (geo fp32, TX fp16, delay fp32).
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, TILE_SE, TG_SIZE, MAX_FPT,
 *               B_SCAT, ELEM_TILE
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

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

        /* Phase 1+2: Geometry + TX sweep (same as v15) */
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

            float sum_re[MAX_FPT], sum_im[MAX_FPT];
            for (int i = 0; i < MAX_FPT; i++) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }

            for (int tile = 0; tile < N_TILES; tile++) {
                int ts = tile * TILE_SE;
                int te = ts + TILE_SE;
                if (te > N_ES) te = N_ES;
                int tl = te - ts;

                f2 cv[TILE_SE], sv_a[TILE_SE];
                #pragma unroll
                for (int j = 0; j < TILE_SE; j++) {
                    if (j >= tl) { cv[j] = {0.0f, 0.0f}; sv_a[j] = {1.0f, 0.0f}; continue; }
                    int se = ts + j, em = se / N_SUB;
                    float ph = GEO_KW_R(si)[se] + lid_f * GEO_KR_STEP(si)[se];
                    float av = GEO_ALPHA_R(si)[se] + lid_f * GEO_AR_STEP(si)[se];
                    float ai = GEO_AMP(si)[se] * expf(-av);
                    float vr, vi;
                    __sincosf(ph, &vi, &vr);
                    vr *= ai; vi *= ai;
                    float dp = lid_f * sh_dps_l[em];
                    float dr, di;
                    __sincosf(dp, &di, &dr);
                    float dvr = sh_da_init_re_l[em]*dr - sh_da_init_im_l[em]*di;
                    float dvi = sh_da_init_re_l[em]*di + sh_da_init_im_l[em]*dr;
                    cv[j] = {vr*dvr - vi*dvi, vr*dvi + vi*dvr};

                    float sp_re = GEO_STP_RX_RE(si)[se];
                    float sp_im = GEO_STP_RX_IM(si)[se];
                    float das_phase = stride_f * sh_dps_l[em];
                    float das_re, das_im;
                    __sincosf(das_phase, &das_im, &das_re);
                    sv_a[j] = {sp_re*das_re - sp_im*das_im, sp_re*das_im + sp_im*das_re};
                }

                for (int fi = 0; fi < my_n_freq; fi++) {
                    #pragma unroll
                    for (int j = 0; j < TILE_SE; j++) {
                        sum_re[fi] += cv[j].x; sum_im[fi] += cv[j].y;
                        cv[j] = cmul(cv[j], sv_a[j]);
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

        /* Phase 3: Element-tiled RX with 2-deep cmul pipelining.
         *
         * Two independent frequency chains:
         *   cv_even processes frequencies at stride*0, stride*2, stride*4, ...
         *   cv_odd  processes frequencies at stride*1, stride*3, stride*5, ...
         * Both use sv2 = cmul(sv, sv) as the double-step.
         */
        for (int eg = 0; eg < N_ELEM_GROUPS; eg++) {
            int se_base = eg * ELEM_TILE;
            int etl = ELEM_TILE;
            if (se_base + etl > N_ES) etl = N_ES - se_base;

            /* cv_even[si][et], cv_odd[si][et], sv2[si][et] */
            f2 cv_even[B_SCAT * ELEM_TILE];
            f2 cv_odd[B_SCAT * ELEM_TILE];
            f2 sv2[B_SCAT * ELEM_TILE];

            #pragma unroll
            for (int si = 0; si < B_SCAT; si++) {
                #pragma unroll
                for (int et = 0; et < ELEM_TILE; et++) {
                    int idx = si * ELEM_TILE + et;
                    if (si >= actual_b || out_flag[si] || et >= etl) {
                        cv_even[idx] = {0.0f, 0.0f};
                        cv_odd[idx] = {0.0f, 0.0f};
                        sv2[idx] = {1.0f, 0.0f};
                        continue;
                    }
                    int se = se_base + et;
                    float ph = GEO_KW_R(si)[se] + lid_f * GEO_KR_STEP(si)[se];
                    float av = GEO_ALPHA_R(si)[se] + lid_f * GEO_AR_STEP(si)[se];
                    float ai = GEO_AMP(si)[se] * expf(-av);
                    float vr, vi;
                    __sincosf(ph, &vi, &vr);
                    f2 cv0 = {vr * ai, vi * ai};

                    f2 sv1 = {GEO_STP_RX_RE(si)[se], GEO_STP_RX_IM(si)[se]};

                    cv_even[idx] = cv0;
                    cv_odd[idx] = cmul(cv0, sv1);
                    sv2[idx] = cmul(sv1, sv1);
                }
            }

            /* Process frequencies in pairs: (fi, fi+1) at a time.
             * cv_even handles fi=0,2,4,...; cv_odd handles fi=1,3,5,...
             * After each pair, advance both by double-step sv2. */
            int fi_pairs = my_n_freq / 2;
            int fi_remainder = my_n_freq % 2;

            for (int fip = 0; fip < fi_pairs; fip++) {
                int f_even = lid + (fip * 2) * TG_SIZE;
                int f_odd  = lid + (fip * 2 + 1) * TG_SIZE;

                float pf_even = (f_even < N_FREQ) ? probe[f_even] : 0.0f;
                float pf_odd  = (f_odd < N_FREQ)  ? probe[f_odd]  : 0.0f;

                float acc_re_e[ELEM_TILE], acc_im_e[ELEM_TILE];
                float acc_re_o[ELEM_TILE], acc_im_o[ELEM_TILE];
                #pragma unroll
                for (int et = 0; et < ELEM_TILE; et++) {
                    acc_re_e[et] = 0.0f; acc_im_e[et] = 0.0f;
                    acc_re_o[et] = 0.0f; acc_im_o[et] = 0.0f;
                }

                #pragma unroll
                for (int si = 0; si < B_SCAT; si++) {
                    float tk_re_e = h2f(sh_tx_re[si * N_FREQ + f_even]);
                    float tk_im_e = h2f(sh_tx_im[si * N_FREQ + f_even]);
                    float tk_re_o = (f_odd < N_FREQ) ? h2f(sh_tx_re[si * N_FREQ + f_odd]) : 0.0f;
                    float tk_im_o = (f_odd < N_FREQ) ? h2f(sh_tx_im[si * N_FREQ + f_odd]) : 0.0f;

                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        int idx = si * ELEM_TILE + et;

                        /* Even chain */
                        float rr_e = cv_even[idx].x * inv_nsub;
                        float ri_e = cv_even[idx].y * inv_nsub;
                        acc_re_e[et] += (tk_re_e*rr_e - tk_im_e*ri_e) * pf_even;
                        acc_im_e[et] += (tk_re_e*ri_e + tk_im_e*rr_e) * pf_even;

                        /* Odd chain */
                        float rr_o = cv_odd[idx].x * inv_nsub;
                        float ri_o = cv_odd[idx].y * inv_nsub;
                        acc_re_o[et] += (tk_re_o*rr_o - tk_im_o*ri_o) * pf_odd;
                        acc_im_o[et] += (tk_re_o*ri_o + tk_im_o*rr_o) * pf_odd;

                        /* Double-step both chains (INDEPENDENT!) */
                        cv_even[idx] = cmul(cv_even[idx], sv2[idx]);
                        cv_odd[idx]  = cmul(cv_odd[idx],  sv2[idx]);
                    }
                }

                /* Write even freq results */
                if (f_even < N_FREQ) {
                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        if (et >= etl) break;
                        int elem = (se_base + et) / N_SUB;
                        atomicAdd(&spect_re[elem * N_FREQ + f_even], acc_re_e[et]);
                        atomicAdd(&spect_im[elem * N_FREQ + f_even], acc_im_e[et]);
                    }
                }
                /* Write odd freq results */
                if (f_odd < N_FREQ) {
                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        if (et >= etl) break;
                        int elem = (se_base + et) / N_SUB;
                        atomicAdd(&spect_re[elem * N_FREQ + f_odd], acc_re_o[et]);
                        atomicAdd(&spect_im[elem * N_FREQ + f_odd], acc_im_o[et]);
                    }
                }
            }

            /* Handle remainder (if my_n_freq is odd). */
            if (fi_remainder) {
                int fi_last = my_n_freq - 1;
                int f_last = lid + fi_last * TG_SIZE;
                if (f_last < N_FREQ) {
                    float pf = probe[f_last];
                    float acc_re_r[ELEM_TILE], acc_im_r[ELEM_TILE];
                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        acc_re_r[et] = 0.0f; acc_im_r[et] = 0.0f;
                    }

                    #pragma unroll
                    for (int si = 0; si < B_SCAT; si++) {
                        float tk_re = h2f(sh_tx_re[si * N_FREQ + f_last]);
                        float tk_im = h2f(sh_tx_im[si * N_FREQ + f_last]);

                        #pragma unroll
                        for (int et = 0; et < ELEM_TILE; et++) {
                            int idx = si * ELEM_TILE + et;
                            float rr = cv_even[idx].x * inv_nsub;
                            float ri = cv_even[idx].y * inv_nsub;
                            acc_re_r[et] += (tk_re*rr - tk_im*ri) * pf;
                            acc_im_r[et] += (tk_re*ri + tk_im*rr) * pf;
                        }
                    }

                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        if (et >= etl) break;
                        int elem = (se_base + et) / N_SUB;
                        atomicAdd(&spect_re[elem * N_FREQ + f_last], acc_re_r[et]);
                        atomicAdd(&spect_im[elem * N_FREQ + f_last], acc_im_r[et]);
                    }
                }
            }
        }

        __syncthreads();
    }
}
