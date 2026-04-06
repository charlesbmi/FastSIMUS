/*
 * TX-only kernel (Pass 1 of two-pass architecture).
 *
 * One block per scatterer. Computes TX[scat][freq] = sum_se(phasor * da) * spectrum * rc.
 * Writes to global memory. No atomics. Embarrassingly parallel.
 *
 * Shmem: only geometry + da arrays = (7*N_ES + 3*N_ELEM) * 4 bytes = ~2.5 KB.
 * Very low register pressure (no RX arrays).
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
void simus_tx_kernel(
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
    float* __restrict__ tx_out_re,
    float* __restrict__ tx_out_im,
    int   n_scat,
    float kw_init, float alpha_init,
    float kw_step, float alpha_step,
    float min_dist, float seg_len,
    float center_kw, float inv_nsub,
    float radius, float apex_offset
) {
    int scat_idx = blockIdx.x;
    if (scat_idx >= n_scat) return;

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
    float* sh_da_re    = sh_stp_im + N_ES;
    float* sh_da_im    = sh_da_re + N_ELEM;
    float* sh_dps_l    = sh_da_im + N_ELEM;

    float sx = scat_x[scat_idx];
    float sz = scat_z[scat_idx];
    float rc = rc_arr[scat_idx];

    bool is_out = (sz < 0.0f);
    if (radius < 1e30f) {
        float da = sx, db = sz + apex_offset;
        is_out = is_out || ((da*da + db*db) <= radius*radius);
    }

    for (int e = lid; e < N_ELEM; e += TG_SIZE) {
        sh_da_re[e] = da_init_re[e];
        sh_da_im[e] = da_init_im[e];
        sh_dps_l[e] = dps[e];
    }

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
        sh_stp_re[se] = sp_re;
        sh_stp_im[se] = sp_im;
    }
    __syncthreads();

    if (is_out) {
        for (int f = lid; f < N_FREQ; f += TG_SIZE) {
            tx_out_re[scat_idx * N_FREQ + f] = 0.0f;
            tx_out_im[scat_idx * N_FREQ + f] = 0.0f;
        }
        return;
    }

    /* TX frequency sweep with tiled sub-element progression */
    const int N_TILES = (N_ES + TILE_SE - 1) / TILE_SE;

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
            float ph = sh_kw_r[se] + lid_f * sh_kr_step[se];
            float av = sh_alpha_r[se] + lid_f * sh_ar_step[se];
            float ai = sh_amp[se] * expf(-av);
            float vr, vi;
            __sincosf(ph, &vi, &vr);
            vr *= ai; vi *= ai;

            float dp = lid_f * sh_dps_l[em];
            float dr, di;
            __sincosf(dp, &di, &dr);
            float dvr = sh_da_re[em]*dr - sh_da_im[em]*di;
            float dvi = sh_da_re[em]*di + sh_da_im[em]*dr;
            cv[j] = {vr*dvr - vi*dvi, vr*dvi + vi*dvr};

            float sp_re = sh_stp_re[se];
            float sp_im = sh_stp_im[se];
            float das_phase = stride_f * sh_dps_l[em];
            float das_re, das_im;
            __sincosf(das_phase, &das_im, &das_re);
            sv[j] = {sp_re*das_re - sp_im*das_im, sp_re*das_im + sp_im*das_re};
        }

        for (int fi = 0; fi < MAX_FPT; fi++) {
            int f = lid + fi * TG_SIZE;
            if (f >= N_FREQ) break;
            #pragma unroll
            for (int j = 0; j < TILE_SE; j++) {
                sum_re[fi] += cv[j].x; sum_im[fi] += cv[j].y;
                cv[j] = cmul(cv[j], sv[j]);
            }
        }
    }

    /* Write TX to global, including spectrum and rc */
    int fi = 0;
    for (int f = lid; f < N_FREQ; f += TG_SIZE, fi++) {
        float tr = sum_re[fi] * inv_nsub;
        float ti = sum_im[fi] * inv_nsub;
        float ppr = pp_re[f], ppi = pp_im[f];
        tx_out_re[scat_idx * N_FREQ + f] = (ppr*tr - ppi*ti) * rc;
        tx_out_im[scat_idx * N_FREQ + f] = (ppr*ti + ppi*tr) * rc;
    }
}
