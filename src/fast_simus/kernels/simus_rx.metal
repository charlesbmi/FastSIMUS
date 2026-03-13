// Kernel B: RX phase -- one thread per (scatterer, element) pair.
// Reads precomputed TX pressure, computes rp_mono per element,
// and writes to output spectrum via atomic add.
//
// VERY LOW register pressure: only N_SUB float2 values for cur/stp.
// Enables high GPU occupancy -> good latency hiding.
//
// Input: tx_re/tx_im[N_SCAT, N_FREQ] from kernel A
// Output: spect_re/spect_im[N_FREQ, N_ELEM] via atomic add
//
// Compile-time constants:
//   N_ELEM, N_SUB, N_FREQ, N_SCAT

    uint tid = thread_position_in_grid.x;
    if (tid >= N_SCAT * N_ELEM) return;

    int scat_idx = tid / N_ELEM;
    int elem_idx = tid % N_ELEM;

    float sx = scat_x[scat_idx];
    float sz = scat_z[scat_idx];
    float rc_i = rc[scat_idx];

    float kw_init    = scalars[0];
    float alpha_init = scalars[1];
    float kw_step    = scalars[2];
    float alpha_step = scalars[3];
    float min_dist   = scalars[4];
    float seg_len    = scalars[5];
    float center_kw  = scalars[6];
    float inv_nsub   = scalars[7];

    float ex = elem_x[elem_idx];
    float ez = elem_z[elem_idx];
    float te = theta_e[elem_idx];

    // Thread-local: only N_SUB float2 values (tiny!)
    float2 cur[N_SUB];
    float2 stp_arr[N_SUB];

    // Init geometric progression for this one element's sub-elements
    for (int s = 0; s < N_SUB; s++) {
        int sub_idx = elem_idx * N_SUB + s;
        float dx = sx - ex - sub_dx[sub_idx];
        float dz = sz - ez - sub_dz[sub_idx];
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

        cur[s] = pi_;
        stp_arr[s] = ps_;
    }

    // Frequency sweep: compute rp_mono, multiply by TX pressure, accumulate
    for (int f = 0; f < N_FREQ; f++) {
        float sr = 0.0f, si = 0.0f;
        for (int s = 0; s < N_SUB; s++) {
            sr += cur[s].x;
            si += cur[s].y;
            float cr = cur[s].x, ci = cur[s].y;
            float tr = stp_arr[s].x, ti = stp_arr[s].y;
            cur[s] = float2(cr * tr - ci * ti, cr * ti + ci * tr);
        }
        float rp_re = sr * inv_nsub;
        float rp_im = si * inv_nsub;

        // Read TX pressure from kernel A output
        int tx_idx = scat_idx * N_FREQ + f;
        float pk_re = tx_re[tx_idx];
        float pk_im = tx_im[tx_idx];

        float probe_f = probe[f];
        float c_re = rc_i * (pk_re * rp_re - pk_im * rp_im) * probe_f;
        float c_im = rc_i * (pk_re * rp_im + pk_im * rp_re) * probe_f;

        int offset = f * N_ELEM + elem_idx;
        atomic_fetch_add_explicit(&spect_re[offset], c_re, memory_order_relaxed);
        atomic_fetch_add_explicit(&spect_im[offset], c_im, memory_order_relaxed);
    }
