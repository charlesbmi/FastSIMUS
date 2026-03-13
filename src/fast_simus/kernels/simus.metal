// Kernel body for simus RF spectrum computation via mx.fast.metal_kernel().
//
// One thread per scatterer. Fuses geometry, phase initialization, and full
// frequency sweep into a single kernel. At each frequency step, computes
// TX pressure at the scatterer (via acoustic reciprocity), scatters by RC,
// and back-propagates to each element via atomic add to the output spectrum.
//
// Compile-time constants (injected via header=):
//   N_ELEM  -- number of transducer elements
//   N_SUB   -- number of sub-elements per element
//   N_FREQ  -- number of selected frequency samples
//   N_ES    -- N_ELEM * N_SUB
//   N_SCAT  -- number of scatterers (for bounds check)

    uint i = thread_position_in_grid.x;
    if (i >= N_SCAT) return;

    float sx = scat_x[i];
    float sz = scat_z[i];
    float is_out_i = is_out[i];
    float rc_i = rc[i];

    float kw_init    = scalars[0];
    float alpha_init = scalars[1];
    float kw_step    = scalars[2];
    float alpha_step = scalars[3];
    float min_dist   = scalars[4];
    float seg_len    = scalars[5];
    float center_kw  = scalars[6];
    float inv_nsub   = scalars[7];

    float2 cur[N_ES];
    float2 stp[N_ES];
    float2 da[N_ELEM];
    float2 rp[N_ELEM];

    // Phase 1: Geometry + init geometric progressions
    for (int e = 0; e < N_ELEM; e++) {
        da[e] = float2(da_init_re[e], da_init_im[e]);
        float ex = elem_x[e];
        float ez = elem_z[e];
        float te = theta_e[e];

        for (int s = 0; s < N_SUB; s++) {
            int idx = e * N_SUB + s;
            float dx = sx - ex - sub_dx[idx];
            float dz = sz - ez - sub_dz[idx];
            float r = metal::precise::sqrt(dx * dx + dz * dz);
            float rc_ = max(r, min_dist);

            float th = metal::precise::asin((dx + 1e-16f) / (r + 1e-16f)) - te;
            float obliq = (fabs(th) >= M_PI_2_F) ? 1e-16f : metal::precise::cos(th);

            // Phase init: obliq/sqrt(r) * exp(-alpha*r + j*wrap(k*r, 2pi))
            float kwr = kw_init * rc_;
            float TWO_PI = 2.0f * M_PI_F;
            float ph_wrap = kwr - TWO_PI * metal::precise::floor(kwr / TWO_PI);
            float ai = obliq / metal::precise::sqrt(rc_) * metal::precise::exp(-alpha_init * rc_);
            float2 pi_ = float2(ai * metal::precise::cos(ph_wrap),
                                ai * metal::precise::sin(ph_wrap));

            // Phase step: exp((-alpha_step + j*k_step) * r)
            float as_ = metal::precise::exp(-alpha_step * rc_);
            float phs = kw_step * rc_;
            float2 ps_ = float2(as_ * metal::precise::cos(phs),
                                as_ * metal::precise::sin(phs));

            // Center-frequency sinc directivity
            float sa = center_kw * seg_len * 0.5f * metal::precise::sin(th);
            float sv = (fabs(sa) < 1e-8f) ? 1.0f : metal::precise::sin(sa) / sa;
            pi_ *= sv;

            cur[idx] = pi_;
            stp[idx] = ps_;
        }
    }

    // Phase 2: Frequency sweep with geometric progression
    for (int f = 0; f < N_FREQ; f++) {
        // Pass 1: Compute rp_mono per element, TX sum, step progressions
        float tx_re = 0.0f, tx_im = 0.0f;

        for (int e = 0; e < N_ELEM; e++) {
            float sr = 0.0f, si = 0.0f;
            for (int s = 0; s < N_SUB; s++) {
                int idx = e * N_SUB + s;
                sr += cur[idx].x;
                si += cur[idx].y;
                // Step geometric progression
                float cr = cur[idx].x, ci = cur[idx].y;
                float tr = stp[idx].x, ti = stp[idx].y;
                cur[idx] = float2(cr * tr - ci * ti, cr * ti + ci * tr);
            }
            rp[e] = float2(sr * inv_nsub, si * inv_nsub);

            // TX: rp_mono[e] * delay_apod[e]
            tx_re += rp[e].x * da[e].x - rp[e].y * da[e].y;
            tx_im += rp[e].x * da[e].y + rp[e].y * da[e].x;

            // Step delay-apod progression
            float dr = da[e].x, di = da[e].y;
            float dsr = da_step_re[e], dsi = da_step_im[e];
            da[e] = float2(dr * dsr - di * dsi, dr * dsi + di * dsr);
        }

        // TX pressure at scatterer: pulse_probe[f] * tx_sum
        float pp_re_f = pp_re[f], pp_im_f = pp_im[f];
        float pk_re = pp_re_f * tx_re - pp_im_f * tx_im;
        float pk_im = pp_re_f * tx_im + pp_im_f * tx_re;

        // Zero out-of-field scatterers
        if (is_out_i > 0.5f) { pk_re = 0.0f; pk_im = 0.0f; }

        // Pass 2: RX contribution to each element
        float w_re = rc_i * pk_re;
        float w_im = rc_i * pk_im;
        float probe_f = probe[f];

        for (int e = 0; e < N_ELEM; e++) {
            float c_re = (w_re * rp[e].x - w_im * rp[e].y) * probe_f;
            float c_im = (w_re * rp[e].y + w_im * rp[e].x) * probe_f;

            int offset = f * N_ELEM + e;
            atomic_fetch_add_explicit(&spect_re[offset], c_re, memory_order_relaxed);
            atomic_fetch_add_explicit(&spect_im[offset], c_im, memory_order_relaxed);
        }
    }
