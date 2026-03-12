// Kernel body for pfield pressure-field computation via mx.fast.metal_kernel().
//
// This file contains ONLY the kernel body -- the code that runs inside the
// auto-generated [[kernel]] void ...() { ... } wrapper.  mx.fast.metal_kernel()
// injects input/output buffer parameters automatically based on input_names
// and output_names.
//
// Compile-time constants (injected via header=):
//   N_ELEM  -- number of transducer elements
//   N_SUB   -- number of sub-elements per element
//   N_FREQ  -- number of frequency samples
//   N_ES    -- N_ELEM * N_SUB (total element-subelement pairs)

    uint g = thread_position_in_grid.x;

    float gx = grid_x[g];
    float gz = grid_z[g];

    float kw_init    = scalars[0];
    float alpha_init = scalars[1];
    float kw_step    = scalars[2];
    float alpha_step = scalars[3];
    float min_dist   = scalars[4];
    float seg_len    = scalars[5];
    float center_kw  = scalars[6];
    float eff_corr   = scalars[7];

    float2 cur[N_ES];
    float2 stp[N_ES];

    for (int e = 0; e < N_ELEM; e++) {
        float ex = elem_x[e];
        float ez = elem_z[e];
        float te = theta_e[e];
        float di_re = da_init_re[e], di_im = da_init_im[e];
        float ds_re = da_step_re[e], ds_im = da_step_im[e];

        for (int s = 0; s < N_SUB; s++) {
            int idx = e * N_SUB + s;

            float dx = gx - ex - sub_dx[idx];
            float dz = gz - ez - sub_dz[idx];
            float r = metal::precise::sqrt(dx * dx + dz * dz);
            float rc = max(r, min_dist);

            // Angle relative to element normal (unclipped distance for angle)
            float th = metal::precise::asin((dx + 1e-16f) / (r + 1e-16f)) - te;

            // Soft baffle obliquity
            float obliq = (fabs(th) >= M_PI_2_F) ? 1e-16f : metal::precise::cos(th);

            // Phase init: obliq/sqrt(r) * exp(-alpha*r + j*wrap(k*r, 2pi))
            float kwr = kw_init * rc;
            float TWO_PI = 2.0f * M_PI_F;
            float ph_wrap = kwr - TWO_PI * metal::precise::floor(kwr / TWO_PI);
            float ai = obliq / metal::precise::sqrt(rc) * metal::precise::exp(-alpha_init * rc);
            float2 pi_ = float2(ai * metal::precise::cos(ph_wrap),
                                ai * metal::precise::sin(ph_wrap));

            // Phase step: exp((-alpha_step + j*k_step) * r)
            float as_ = metal::precise::exp(-alpha_step * rc);
            float phs = kw_step * rc;
            float2 ps_ = float2(as_ * metal::precise::cos(phs),
                                as_ * metal::precise::sin(phs));

            // Center-frequency sinc directivity
            float sa = center_kw * seg_len * 0.5f * metal::precise::sin(th);
            float sv = (fabs(sa) < 1e-8f) ? 1.0f : metal::precise::sin(sa) / sa;
            pi_ *= sv;

            // Absorb delay+apodization (complex multiply)
            cur[idx] = float2(
                pi_.x * di_re - pi_.y * di_im,
                pi_.x * di_im + pi_.y * di_re
            );
            stp[idx] = float2(
                ps_.x * ds_re - ps_.y * ds_im,
                ps_.x * ds_im + ps_.y * ds_re
            );
        }
    }

    // Frequency sweep: accumulate sum_f |pulse_probe_f|^2 * |sum_es phase_es_f|^2
    float acc = 0.0f;
    for (int f = 0; f < N_FREQ; f++) {
        float sr = 0.0f, si = 0.0f;
        for (int j = 0; j < N_ES; j++) {
            sr += cur[j].x;
            si += cur[j].y;
            float cr = cur[j].x, ci = cur[j].y;
            float tr = stp[j].x, ti = stp[j].y;
            cur[j] = float2(cr * tr - ci * ti, cr * ti + ci * tr);
        }
        acc += pp_mag_sq[f] * (sr * sr + si * si);
    }

    pressure[g] = (is_out[g] > 0.5f) ? 0.0f : acc * eff_corr;
