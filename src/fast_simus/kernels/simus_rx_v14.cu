/*
 * RX kernel with warp-shuffle cross-scatterer reduction (Pass 2 of two-pass).
 *
 * Thread mapping: tid = elem_idx * SCAT_REDUCE + scat_batch
 * Adjacent threads handle DIFFERENT scatterers for the SAME element.
 * Warp shuffle reduces across SCAT_REDUCE scatterers before atomicAdd.
 * Result: SCAT_REDUCE fewer atomics than naive approach.
 *
 * Each thread processes ALL N_FREQ frequencies for its (element, scatterer).
 * Reads TX[scat][freq] from global memory (L1/L2 cached via broadcast).
 *
 * Block: N_ELEM * SCAT_REDUCE threads (e.g., 64*2 = 128)
 * Grid: ceil(N_SCAT / SCAT_REDUCE)
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, SCAT_REDUCE, TG_SIZE
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

extern "C" __global__
void simus_rx_kernel(
    const float* __restrict__ scat_x,
    const float* __restrict__ scat_z,
    const float* __restrict__ elem_x,
    const float* __restrict__ elem_z,
    const float* __restrict__ cos_te,
    const float* __restrict__ sin_neg_te,
    const float* __restrict__ sub_dx,
    const float* __restrict__ sub_dz,
    const float* __restrict__ tx_re,
    const float* __restrict__ tx_im,
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
    int scat_batch = lid % SCAT_REDUCE;
    int elem_idx = lid / SCAT_REDUCE;

    int tg_scat_base = blockIdx.x * SCAT_REDUCE;
    int scat_idx = tg_scat_base + scat_batch;

    bool valid = (scat_idx < n_scat && elem_idx < N_ELEM);

    float rx_re = 0.0f, rx_im = 0.0f;
    float step_re = 1.0f, step_im = 0.0f;

    if (valid) {
        float sx = scat_x[scat_idx], sz = scat_z[scat_idx];

        bool is_out = (sz < 0.0f);
        if (radius < 1e30f) {
            float da = sx, db = sz + apex_offset;
            is_out = is_out || ((da*da + db*db) <= radius*radius);
        }

        if (!is_out) {
            int se = elem_idx * N_SUB;
            float ex_ = elem_x[elem_idx], ez_ = elem_z[elem_idx];
            float ct = cos_te[elem_idx], snt = sin_neg_te[elem_idx];
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

            float amp = obliq * sv * rsqrtf(rc_);

            float ph = kw_init * rc_;
            float av = alpha_init * rc_;
            float ai = amp * expf(-av);
            float vr, vi;
            __sincosf(ph, &vi, &vr);
            rx_re = vr * ai;
            rx_im = vi * ai;

            float stp_phase = kw_step * rc_;
            float stp_alpha = alpha_step * rc_;
            float sm = expf(-stp_alpha);
            __sincosf(stp_phase, &step_im, &step_re);
            step_re *= sm;
            step_im *= sm;
        }
    }

    int tx_base = valid ? scat_idx * N_FREQ : 0;

    for (int f = 0; f < N_FREQ; f++) {
        float c_re = 0.0f, c_im = 0.0f;

        if (valid) {
            float rr = rx_re * inv_nsub;
            float ri = rx_im * inv_nsub;
            float tk_re = tx_re[tx_base + f];
            float tk_im = tx_im[tx_base + f];
            float pf = probe[f];
            c_re = (tk_re*rr - tk_im*ri) * pf;
            c_im = (tk_re*ri + tk_im*rr) * pf;

            float new_re = rx_re * step_re - rx_im * step_im;
            float new_im = rx_re * step_im + rx_im * step_re;
            rx_re = new_re;
            rx_im = new_im;
        }

#if SCAT_REDUCE >= 2
        c_re += __shfl_xor_sync(0xFFFFFFFF, c_re, 1);
        c_im += __shfl_xor_sync(0xFFFFFFFF, c_im, 1);
#endif
#if SCAT_REDUCE >= 4
        c_re += __shfl_xor_sync(0xFFFFFFFF, c_re, 2);
        c_im += __shfl_xor_sync(0xFFFFFFFF, c_im, 2);
#endif
#if SCAT_REDUCE >= 8
        c_re += __shfl_xor_sync(0xFFFFFFFF, c_re, 4);
        c_im += __shfl_xor_sync(0xFFFFFFFF, c_im, 4);
#endif
#if SCAT_REDUCE >= 16
        c_re += __shfl_xor_sync(0xFFFFFFFF, c_re, 8);
        c_im += __shfl_xor_sync(0xFFFFFFFF, c_im, 8);
#endif

        if (scat_batch == 0 && valid) {
            atomicAdd(&spect_re[elem_idx * N_FREQ + f], c_re);
            atomicAdd(&spect_im[elem_idx * N_FREQ + f], c_im);
        }
    }
}
