/*
 * Fused TX+RX SIMUS kernel -- v16: Grid-Y element group partitioning.
 *
 * Based on v11. Each block handles ONE element group (selected by blockIdx.y)
 * instead of iterating all groups. Grid is (N_BLOCKS_X, N_ELEM_GROUPS, 1).
 * Phase 1+2 unchanged; Phase 3 processes only blockIdx.y-th element group.
 *
 * Tradeoff: 8x redundant Phase 1+2 compute vs 8x fewer atomics per entry.
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, TILE_SE, TG_SIZE, MAX_FPT,
 *               B_SCAT, ELEM_TILE
 *
 * Shared memory: (7*B_SCAT*N_ES + 2*B_SCAT*N_FREQ + 3*N_ELEM) * 4 bytes
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

    float* sh_tx_re = shmem + 7 * B_SCAT * N_ES;
    float* sh_tx_im = sh_tx_re + B_SCAT * N_FREQ;
    float* sh_da_init_re_l = sh_tx_im + B_SCAT * N_FREQ;
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

    bool out_flag[B_SCAT];

    for (int scat_base = blockIdx.x * B_SCAT;
         scat_base < n_scat;
         scat_base += gridDim.x * B_SCAT)
    {
        int actual_b = B_SCAT;
        if (scat_base + B_SCAT > n_scat)
            actual_b = n_scat - scat_base;

        /* ---- Phase 1+2: geometry + TX for each scatterer in batch ---- */
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
                    sh_tx_re[si * N_FREQ + f] = 0.0f;
                    sh_tx_im[si * N_FREQ + f] = 0.0f;
                }
                __syncthreads();
                continue;
            }

            /* Phase 2: TX sweep */
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
                    sh_tx_re[si * N_FREQ + f] = (ppr*tr - ppi*ti) * rc;
                    sh_tx_im[si * N_FREQ + f] = (ppr*ti + ppi*tr) * rc;
                }
            }
            __syncthreads();
        }

        /* ---- Phase 3: SINGLE element group selected by blockIdx.y ---- */
        {
            int eg = blockIdx.y;
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
                    int se = se_base + et;
                    float ph = GEO_KW_R(si)[se] + lid_f * GEO_KR_STEP(si)[se];
                    float av = GEO_ALPHA_R(si)[se] + lid_f * GEO_AR_STEP(si)[se];
                    float ai = GEO_AMP(si)[se] * expf(-av);
                    float vr, vi;
                    __sincosf(ph, &vi, &vr);
                    cv[idx] = {vr * ai, vi * ai};
                    sv_arr[idx] = {GEO_STP_RX_RE(si)[se], GEO_STP_RX_IM(si)[se]};
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
                    float tk_re = sh_tx_re[si * N_FREQ + f];
                    float tk_im = sh_tx_im[si * N_FREQ + f];

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
                    atomicAdd(&spect_re[elem * N_FREQ + f], acc_re[et]);
                    atomicAdd(&spect_im[elem * N_FREQ + f], acc_im[et]);
                }
            }
        }

        __syncthreads();
    }
}
