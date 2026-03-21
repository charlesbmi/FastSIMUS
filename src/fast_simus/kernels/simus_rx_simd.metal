// Kernel B: SIMD-reduce RX -- multiple scatterers per threadgroup with
// cross-scatterer SIMD reduction to cut atomic writes by SCAT_REDUCE.
//
// Thread layout: tid = elem_idx * SCAT_REDUCE + scat_batch
//   - Adjacent threads handle the SAME element from DIFFERENT scatterers
//   - Within a SIMD group (32 threads): 32/SR elements * SR scatterers
//   - simd_shuffle_xor reduces groups of SR threads (same element, different scat)
//   - Only scat_batch==0 threads write atomics -> SR fewer atomics
//
// Coalescing: writing threads (scat_batch==0) are at stride SR in the SIMD group.
// They write to consecutive element addresses -> coalesced atomics.
//
// TG = N_ELEM * SCAT_REDUCE (e.g., 64*2 = 128 for P4-2v with SR=2)
//
// Compile-time constants:
//   N_ELEM, N_SUB, N_FREQ, N_SCAT, SCAT_REDUCE

    uint tg_scat_base = threadgroup_position_in_grid.x * SCAT_REDUCE;
    uint lid = thread_position_in_threadgroup.x;
    uint scat_batch = lid % SCAT_REDUCE;
    uint elem_idx = lid / SCAT_REDUCE;
    uint scat_idx = tg_scat_base + scat_batch;

    bool valid = (scat_idx < (uint)N_SCAT && elem_idx < (uint)N_ELEM);

    float sx, sz, rc_i, ex, ez, te;
    float kw_init, alpha_init, kw_step, alpha_step, min_dist, seg_len, center_kw, inv_nsub;

    kw_init    = scalars[0];
    alpha_init = scalars[1];
    kw_step    = scalars[2];
    alpha_step = scalars[3];
    min_dist   = scalars[4];
    seg_len    = scalars[5];
    center_kw  = scalars[6];
    inv_nsub   = scalars[7];

    float2 cur[N_SUB];
    float2 stp_arr[N_SUB];

    if (valid) {
        sx = scat_x[scat_idx];
        sz = scat_z[scat_idx];
        rc_i = rc[scat_idx];
        ex = elem_x[elem_idx];
        ez = elem_z[elem_idx];
        te = theta_e[elem_idx];

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
    }

    for (int f = 0; f < N_FREQ; f++) {
        float c_re = 0.0f, c_im = 0.0f;

        if (valid) {
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

            int tx_idx = scat_idx * N_FREQ + f;
            float pk_re = tx_re[tx_idx];
            float pk_im = tx_im[tx_idx];

            float probe_f = probe[f];
            c_re = rc_i * (pk_re * rp_re - pk_im * rp_im) * probe_f;
            c_im = rc_i * (pk_re * rp_im + pk_im * rp_re) * probe_f;
        }

        // SIMD reduce across SCAT_REDUCE scatterers for the same element.
        // All threads participate (invalid threads contribute 0).
#if SCAT_REDUCE >= 2
        c_re += simd_shuffle_xor(c_re, 1);
        c_im += simd_shuffle_xor(c_im, 1);
#endif
#if SCAT_REDUCE >= 4
        c_re += simd_shuffle_xor(c_re, 2);
        c_im += simd_shuffle_xor(c_im, 2);
#endif
#if SCAT_REDUCE >= 8
        c_re += simd_shuffle_xor(c_re, 4);
        c_im += simd_shuffle_xor(c_im, 4);
#endif
#if SCAT_REDUCE >= 16
        c_re += simd_shuffle_xor(c_re, 8);
        c_im += simd_shuffle_xor(c_im, 8);
#endif

        if (scat_batch == 0 && valid) {
            int offset = f * N_ELEM + elem_idx;
            atomic_fetch_add_explicit(&spect_re[offset], c_re, memory_order_relaxed);
            atomic_fetch_add_explicit(&spect_im[offset], c_im, memory_order_relaxed);
        }
    }
