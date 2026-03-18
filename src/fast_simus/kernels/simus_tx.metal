// Kernel A: TX phase -- one thread per scatterer.
// Computes TX pressure at each scatterer for each frequency.
// Delay+apodization (da) absorbed into cur/stp at init (pfield-style),
// eliminating da[N_ELEM] from registers and 12 flops/elem/freq from the sweep.
//
// Output: tx_re[N_SCAT * N_FREQ], tx_im[N_SCAT * N_FREQ]
//
// Compile-time constants:
//   N_ELEM, N_SUB, N_FREQ, N_ES, N_SCAT

    uint i = thread_position_in_grid.x;
    if (i >= N_SCAT) return;

    float sx = scat_x[i];
    float sz = scat_z[i];
    float is_out_i = is_out[i];

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

    // Phase 1: Geometry + init with da absorption
    for (int e = 0; e < N_ELEM; e++) {
        float di_re = da_init_re[e], di_im = da_init_im[e];
        float ds_re = da_step_re[e], ds_im = da_step_im[e];
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

            float kwr = kw_init * rc_;
            float TWO_PI = 2.0f * M_PI_F;
            float ph_wrap = kwr - TWO_PI * metal::precise::floor(kwr / TWO_PI);
            float ai = obliq / metal::precise::sqrt(rc_) * metal::precise::exp(-alpha_init * rc_);
            float2 pi_ = float2(ai * metal::precise::cos(ph_wrap),
                                ai * metal::precise::sin(ph_wrap));

            float as_ = metal::precise::exp(-alpha_step * rc_);
            float phs = kw_step * rc_;
            float2 ps_ = float2(as_ * metal::precise::cos(phs),
                                as_ * metal::precise::sin(phs));

            float sa = center_kw * seg_len * 0.5f * metal::precise::sin(th);
            float sv = (fabs(sa) < 1e-8f) ? 1.0f : metal::precise::sin(sa) / sa;
            pi_ *= sv;

            // Absorb delay+apodization into geometric progression
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

    // Phase 2: Frequency sweep -- TX only (da already in cur/stp)
    for (int f = 0; f < N_FREQ; f++) {
        float tx_re_acc = 0.0f, tx_im_acc = 0.0f;

        for (int e = 0; e < N_ELEM; e++) {
            float sr = 0.0f, si = 0.0f;
            for (int s = 0; s < N_SUB; s++) {
                int idx = e * N_SUB + s;
                sr += cur[idx].x;
                si += cur[idx].y;
                float cr = cur[idx].x, ci = cur[idx].y;
                float tr = stp[idx].x, ti = stp[idx].y;
                cur[idx] = float2(cr * tr - ci * ti, cr * ti + ci * tr);
            }
            tx_re_acc += sr * inv_nsub;
            tx_im_acc += si * inv_nsub;
        }

        float pp_re_f = pp_re[f], pp_im_f = pp_im[f];
        float pk_re = pp_re_f * tx_re_acc - pp_im_f * tx_im_acc;
        float pk_im = pp_re_f * tx_im_acc + pp_im_f * tx_re_acc;
        if (is_out_i > 0.5f) { pk_re = 0.0f; pk_im = 0.0f; }

        int out_idx = i * N_FREQ + f;
        tx_re[out_idx] = pk_re;
        tx_im[out_idx] = pk_im;
    }
