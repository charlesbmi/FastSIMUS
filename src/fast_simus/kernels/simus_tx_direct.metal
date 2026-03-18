// Kernel: Direct per-frequency TX computation.
// One thread per (scatterer, frequency) pair -- no thread-local arrays,
// zero register pressure, embarrassingly parallel.
//
// Uses metal::fast:: trig for phase computation (safe: no accumulated error).
// Geometry is recomputed per thread (traded for zero register pressure).
//
// Output: tx_re[N_SCAT * N_FREQ], tx_im[N_SCAT * N_FREQ]
//
// Compile-time constants:
//   N_ELEM, N_SUB, N_FREQ, N_ES, N_SCAT

    uint tid = thread_position_in_grid.x;
    if (tid >= N_SCAT * N_FREQ) return;

    int scat_idx = tid / N_FREQ;
    int freq_idx = tid % N_FREQ;

    float sx = scat_x[scat_idx];
    float sz = scat_z[scat_idx];
    float is_out_i = is_out[scat_idx];

    float kw_init    = scalars[0];
    float alpha_init = scalars[1];
    float kw_step    = scalars[2];
    float alpha_step = scalars[3];
    float min_dist   = scalars[4];
    float seg_len    = scalars[5];
    float center_kw  = scalars[6];
    float inv_nsub   = scalars[7];

    float kw_f = kw_init + freq_idx * kw_step;
    float alpha_f = alpha_init + freq_idx * alpha_step;
    float TWO_PI = 2.0f * M_PI_F;

    float tx_re_acc = 0.0f, tx_im_acc = 0.0f;

    for (int e = 0; e < N_ELEM; e++) {
        float ex = elem_x[e];
        float ez = elem_z[e];
        float te = theta_e[e];

        // da at this frequency: da_init * exp(j * freq_idx * delay_phase_step)
        float dp = float(freq_idx) * delay_phase_step[e];
        float ds_re = metal::fast::cos(dp);
        float ds_im = metal::fast::sin(dp);
        float da_re = da_init_re[e] * ds_re - da_init_im[e] * ds_im;
        float da_im = da_init_re[e] * ds_im + da_init_im[e] * ds_re;

        float sr = 0.0f, si = 0.0f;

        for (int s = 0; s < N_SUB; s++) {
            int idx = e * N_SUB + s;
            float dx = sx - ex - sub_dx[idx];
            float dz = sz - ez - sub_dz[idx];
            float r = metal::precise::sqrt(dx * dx + dz * dz);
            float rc_ = max(r, min_dist);

            float th = metal::precise::asin((dx + 1e-16f) / (r + 1e-16f)) - te;
            float obliq = (fabs(th) >= M_PI_2_F) ? 1e-16f : metal::fast::cos(th);

            float sa = center_kw * seg_len * 0.5f * metal::fast::sin(th);
            float sv = (fabs(sa) < 1e-8f) ? 1.0f : metal::fast::sin(sa) / sa;

            float kwr = kw_f * rc_;
            float ph_wrap = kwr - TWO_PI * metal::fast::floor(kwr / TWO_PI);
            float ai = obliq * sv / metal::precise::sqrt(rc_) * metal::fast::exp(-alpha_f * rc_);

            sr += ai * metal::fast::cos(ph_wrap);
            si += ai * metal::fast::sin(ph_wrap);
        }

        // Apply da and accumulate
        float elem_re = (sr * da_re - si * da_im) * inv_nsub;
        float elem_im = (sr * da_im + si * da_re) * inv_nsub;
        tx_re_acc += elem_re;
        tx_im_acc += elem_im;
    }

    // Apply pulse*probe spectrum
    float pp_re_f = pp_re[freq_idx], pp_im_f = pp_im[freq_idx];
    float pk_re = pp_re_f * tx_re_acc - pp_im_f * tx_im_acc;
    float pk_im = pp_re_f * tx_im_acc + pp_im_f * tx_re_acc;
    if (is_out_i > 0.5f) { pk_re = 0.0f; pk_im = 0.0f; }

    int out_idx = scat_idx * N_FREQ + freq_idx;
    tx_re[out_idx] = pk_re;
    tx_im[out_idx] = pk_im;
