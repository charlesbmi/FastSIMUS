// Kernel: Shared-memory geometry + direct per-frequency TX computation.
// One threadgroup per scatterer; threads cooperatively compute geometry
// into shared memory, then each thread handles ceil(N_FREQ/TG) frequencies.
//
// Eliminates both register arrays AND redundant geometry recomputation.
// Uses metal::fast:: trig for frequency-domain phase (no accumulated error).
//
// Shared memory layout per sub-element:
//   amp[N_ES]       obliq * sinc / sqrt(r)  (frequency-independent amplitude)
//   kw_r[N_ES]      kw_init * r             (base phase)
//   kr_step[N_ES]   kw_step * r             (phase increment per freq index)
//   alpha_r[N_ES]   alpha_init * r           (base attenuation)
//   ar_step[N_ES]   alpha_step * r           (attenuation increment per freq)
//   elem_of[N_ES]   which element this sub-element belongs to (as float)
//
// Total: N_ES * 6 * 4 = 1536 bytes for N_ES=64. Well within 32KB limit.
//
// Output: tx_re[N_SCAT * N_FREQ], tx_im[N_SCAT * N_FREQ]
//
// Compile-time constants:
//   N_ELEM, N_SUB, N_FREQ, N_ES, N_SCAT

    threadgroup float shared_amp[N_ES];
    threadgroup float shared_kw_r[N_ES];
    threadgroup float shared_kr_step[N_ES];
    threadgroup float shared_alpha_r[N_ES];
    threadgroup float shared_ar_step[N_ES];
    threadgroup float shared_elem_idx[N_ES];

    uint scat_idx = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;

    if (scat_idx >= N_SCAT) return;

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

    // Phase 1: Cooperatively compute geometry into shared memory
    for (uint e = lid; e < (uint)N_ES; e += tpg) {
        int elem = e / N_SUB;
        int sub = e % N_SUB;
        int sub_global = elem * N_SUB + sub;

        float ex = elem_x[elem];
        float ez = elem_z[elem];
        float te = theta_e[elem];

        float dx = sx - ex - sub_dx[sub_global];
        float dz = sz - ez - sub_dz[sub_global];
        float r = metal::precise::sqrt(dx * dx + dz * dz);
        float rc_ = max(r, min_dist);

        float th = metal::precise::asin((dx + 1e-16f) / (r + 1e-16f)) - te;
        float obliq = (fabs(th) >= M_PI_2_F) ? 1e-16f : metal::fast::cos(th);

        float sa = center_kw * seg_len * 0.5f * metal::fast::sin(th);
        float sv = (fabs(sa) < 1e-8f) ? 1.0f : metal::fast::sin(sa) / sa;

        shared_amp[e] = obliq * sv / metal::precise::sqrt(rc_);
        shared_kw_r[e] = kw_init * rc_;
        shared_kr_step[e] = kw_step * rc_;
        shared_alpha_r[e] = alpha_init * rc_;
        shared_ar_step[e] = alpha_step * rc_;
        shared_elem_idx[e] = float(elem);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float TWO_PI = 2.0f * M_PI_F;

    // Phase 2: Each thread handles a stripe of frequencies
    for (uint f = lid; f < (uint)N_FREQ; f += tpg) {
        float f_float = float(f);

        float tx_re_acc = 0.0f, tx_im_acc = 0.0f;
        int prev_elem = -1;
        float da_re = 0.0f, da_im = 0.0f;
        float sr = 0.0f, si = 0.0f;

        for (int e = 0; e < N_ES; e++) {
            int cur_elem = int(shared_elem_idx[e]);

            // When we cross to a new element, flush the sub-element sum
            if (cur_elem != prev_elem) {
                if (prev_elem >= 0) {
                    tx_re_acc += (sr * da_re - si * da_im) * inv_nsub;
                    tx_im_acc += (sr * da_im + si * da_re) * inv_nsub;
                }
                sr = 0.0f;
                si = 0.0f;

                // Compute da for this element at this frequency
                float dp = f_float * delay_phase_step[cur_elem];
                float ds_re = metal::fast::cos(dp);
                float ds_im = metal::fast::sin(dp);
                da_re = da_init_re[cur_elem] * ds_re - da_init_im[cur_elem] * ds_im;
                da_im = da_init_re[cur_elem] * ds_im + da_init_im[cur_elem] * ds_re;
                prev_elem = cur_elem;
            }

            // Direct phase computation from precomputed geometry
            float total_phase = shared_kw_r[e] + f_float * shared_kr_step[e];
            float total_alpha = shared_alpha_r[e] + f_float * shared_ar_step[e];
            float ph_wrap = total_phase - TWO_PI * metal::fast::floor(total_phase / TWO_PI);
            float ai = shared_amp[e] * metal::fast::exp(-total_alpha);

            sr += ai * metal::fast::cos(ph_wrap);
            si += ai * metal::fast::sin(ph_wrap);
        }

        // Flush final element
        if (prev_elem >= 0) {
            tx_re_acc += (sr * da_re - si * da_im) * inv_nsub;
            tx_im_acc += (sr * da_im + si * da_re) * inv_nsub;
        }

        // Apply pulse*probe spectrum
        float pp_re_f = pp_re[f], pp_im_f = pp_im[f];
        float pk_re = pp_re_f * tx_re_acc - pp_im_f * tx_im_acc;
        float pk_im = pp_re_f * tx_im_acc + pp_im_f * tx_re_acc;
        if (is_out_i > 0.5f) { pk_re = 0.0f; pk_im = 0.0f; }

        int out_idx = scat_idx * N_FREQ + f;
        tx_re[out_idx] = pk_re;
        tx_im[out_idx] = pk_im;
    }
