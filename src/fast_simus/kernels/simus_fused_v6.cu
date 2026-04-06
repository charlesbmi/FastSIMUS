/*
 * Fused TX+RX SIMUS kernel -- v6: persistent + register TX + tuned ILP.
 *
 * Combines: persistent threads (reduced atomicAdd contention),
 * register-based TX (no sh_tx buffer → 3KB shmem → higher occupancy),
 * algebraic geometry (no asinf), loop unrolling hints.
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, TILE_SE, TG_SIZE, MAX_FPT
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct f2 { float x, y; };

__device__ __forceinline__ f2 cmul(f2 a, f2 b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

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
    float* sh_amp      = shmem;
    float* sh_kw_r     = sh_amp + N_ES;
    float* sh_kr_step  = sh_kw_r + N_ES;
    float* sh_alpha_r  = sh_kr_step + N_ES;
    float* sh_ar_step  = sh_alpha_r + N_ES;
    float* sh_stp_re   = sh_ar_step + N_ES;
    float* sh_stp_im   = sh_stp_re + N_ES;
    float* sh_stp_rx_re = sh_stp_im + N_ES;
    float* sh_stp_rx_im = sh_stp_rx_re + N_ES;
    float* sh_da_init_re_l = sh_stp_rx_im + N_ES;
    float* sh_da_init_im_l = sh_da_init_re_l + N_ELEM;
    float* sh_dps_l    = sh_da_init_im_l + N_ELEM;

    for (int e = lid; e < N_ELEM; e += TG_SIZE) {
        sh_da_init_re_l[e] = da_init_re[e];
        sh_da_init_im_l[e] = da_init_im[e];
        sh_dps_l[e]        = dps[e];
    }
    __syncthreads();

    int my_n_freq = 0;
    for (int f = lid; f < N_FREQ; f += TG_SIZE) my_n_freq++;

    for (int scat_idx = blockIdx.x; scat_idx < n_scat; scat_idx += gridDim.x) {
        float sx = scat_x[scat_idx];
        float sz = scat_z[scat_idx];
        float rc = rc_arr[scat_idx];

        bool is_out = (sz < 0.0f);
        if (radius < 1e30f) {
            float da = sx, db = sz + apex_offset;
            is_out = is_out || ((da*da + db*db) <= radius*radius);
        }

        /* Phase 1: geometry */
        for (int se = lid; se < N_ES; se += TG_SIZE) {
            int elem = se / N_SUB;
            float ex = elem_x[elem], ez = elem_z[elem];
            float ct = cos_te[elem], snt = sin_neg_te[elem];
            float dx = sx - ex - sub_dx[se];
            float dz = sz - ez - sub_dz[se];
            float r2 = dx*dx + dz*dz;
            float inv_r = rsqrtf(r2 + 1e-30f);
            float r = r2 * inv_r;
            float rc_ = fmaxf(r, min_dist);

            float sin_th = (dx*ct + dz*snt) * inv_r;
            float cos_th = (dz*ct - dx*snt) * inv_r;
            float obliq = (cos_th <= 0.0f) ? 1e-16f : cos_th;
            float sa = center_kw * seg_len * 0.5f * sin_th;
            float sv = (fabsf(sa) < 1e-8f) ? 1.0f : __fdividef(__sinf(sa), sa);

            sh_amp[se]     = obliq * sv * rsqrtf(rc_);
            sh_kw_r[se]    = kw_init * rc_;
            sh_kr_step[se] = kw_step * rc_;
            sh_alpha_r[se] = alpha_init * rc_;
            sh_ar_step[se] = alpha_step * rc_;

            float stp_phase = stride_f * kw_step * rc_;
            float stp_alpha = stride_f * alpha_step * rc_;
            float sm = expf(-stp_alpha);
            float sp_re, sp_im;
            __sincosf(stp_phase, &sp_im, &sp_re);
            sp_re *= sm; sp_im *= sm;

            float das_phase = stride_f * sh_dps_l[elem];
            float das_re, das_im;
            __sincosf(das_phase, &das_im, &das_re);

            sh_stp_re[se] = sp_re*das_re - sp_im*das_im;
            sh_stp_im[se] = sp_re*das_im + sp_im*das_re;
            sh_stp_rx_re[se] = sp_re;
            sh_stp_rx_im[se] = sp_im;
        }
        __syncthreads();

        if (is_out) { __syncthreads(); continue; }

        /* Phase 2: TX sweep (register accumulators) */
        const int N_TILES = (N_ES + TILE_SE - 1) / TILE_SE;

        float sum_re[MAX_FPT], sum_im[MAX_FPT];
        for (int i = 0; i < MAX_FPT; i++) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }

        for (int tile = 0; tile < N_TILES; tile++) {
            int ts = tile * TILE_SE;
            int te = ts + TILE_SE;
            if (te > N_ES) te = N_ES;
            int tl = te - ts;

            f2 ct[TILE_SE], st[TILE_SE];
            #pragma unroll
            for (int j = 0; j < TILE_SE; j++) {
                if (j >= tl) { ct[j] = {0.0f, 0.0f}; st[j] = {1.0f, 0.0f}; continue; }
                int se = ts + j, em = se / N_SUB;
                float ph = sh_kw_r[se] + lid_f * sh_kr_step[se];
                float av = sh_alpha_r[se] + lid_f * sh_ar_step[se];
                float ai = sh_amp[se] * expf(-av);
                float pr, pi;
                __sincosf(ph, &pi, &pr);
                pr *= ai; pi *= ai;
                float dp = lid_f * sh_dps_l[em];
                float dr, di;
                __sincosf(dp, &di, &dr);
                float dvr = sh_da_init_re_l[em]*dr - sh_da_init_im_l[em]*di;
                float dvi = sh_da_init_re_l[em]*di + sh_da_init_im_l[em]*dr;
                ct[j] = {pr*dvr - pi*dvi, pr*dvi + pi*dvr};
                st[j] = {sh_stp_re[se], sh_stp_im[se]};
            }

            for (int fi = 0; fi < my_n_freq; fi++) {
                #pragma unroll
                for (int j = 0; j < TILE_SE; j++) {
                    sum_re[fi] += ct[j].x; sum_im[fi] += ct[j].y;
                    ct[j] = cmul(ct[j], st[j]);
                }
            }
        }

        /* Finalize TX in registers (no shared memory write). */
        float tx_re[MAX_FPT], tx_im[MAX_FPT];
        {
            int fi = 0;
            for (int f = lid; f < N_FREQ; f += TG_SIZE, fi++) {
                float tr = sum_re[fi] * inv_nsub;
                float ti = sum_im[fi] * inv_nsub;
                float ppr = pp_re[f], ppi = pp_im[f];
                tx_re[fi] = (ppr*tr - ppi*ti) * rc;
                tx_im[fi] = (ppr*ti + ppi*tr) * rc;
            }
        }

        /* Phase 3: RX sweep reading TX from registers */
        for (int tile = 0; tile < N_TILES; tile++) {
            int ts = tile * TILE_SE;
            int te = ts + TILE_SE;
            if (te > N_ES) te = N_ES;
            int tl = te - ts;

            f2 ct[TILE_SE], st[TILE_SE];
            #pragma unroll
            for (int j = 0; j < TILE_SE; j++) {
                if (j >= tl) { ct[j] = {0.0f, 0.0f}; st[j] = {1.0f, 0.0f}; continue; }
                int se = ts + j;
                float ph = sh_kw_r[se] + lid_f * sh_kr_step[se];
                float av = sh_alpha_r[se] + lid_f * sh_ar_step[se];
                float ai = sh_amp[se] * expf(-av);
                float pr, pi;
                __sincosf(ph, &pi, &pr);
                pr *= ai; pi *= ai;
                ct[j] = {pr, pi};
                st[j] = {sh_stp_rx_re[se], sh_stp_rx_im[se]};
            }

            for (int fi = 0; fi < my_n_freq; fi++) {
                int f = lid + fi * TG_SIZE;
                if (f >= N_FREQ) break;
                float pk_re = tx_re[fi];
                float pk_im = tx_im[fi];
                float pf = probe[f];
                #pragma unroll
                for (int j = 0; j < TILE_SE; j++) {
                    if (j >= tl) break;
                    int se = ts + j;
                    int elem = se / N_SUB;
                    float rr = ct[j].x * inv_nsub;
                    float ri = ct[j].y * inv_nsub;
                    float cre = (pk_re*rr - pk_im*ri) * pf;
                    float cim = (pk_re*ri + pk_im*rr) * pf;
                    atomicAdd(&spect_re[elem * N_FREQ + f], cre);
                    atomicAdd(&spect_im[elem * N_FREQ + f], cim);
                    ct[j] = cmul(ct[j], st[j]);
                }
            }
        }

        __syncthreads();
    }
}
