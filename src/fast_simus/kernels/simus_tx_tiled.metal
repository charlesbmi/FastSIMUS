// Kernel: Element-tiled progression with shared-memory geometry.
// One threadgroup per scatterer; threads cooperatively compute geometry
// AND da-absorbed stride steps into shared memory, then each thread
// processes sub-element tiles with geometric progression (ALU-only inner loop).
//
// Key advantage over shared-direct: inner loop has 0 SFU calls (vs 4-5).
// Key advantage over full progression: register pressure is TILE_SE*2 float2
// (e.g., 256 bytes for TILE_SE=16) vs N_ES*2 float2 (1024 bytes for N_ES=64).
//
// Shared memory layout:
//   amp[N_ES]         frequency-independent amplitude
//   kw_r[N_ES]        kw_init * r  (base phase)
//   kr_step[N_ES]     kw_step * r  (phase increment per freq index)
//   alpha_r[N_ES]     alpha_init * r  (base attenuation)
//   ar_step[N_ES]     alpha_step * r  (attenuation increment per freq)
//   stp[N_ES]         float2 stride step, da-absorbed (same for all threads)
//   da_init_re[N_ELEM] delay+apod init real part
//   da_init_im[N_ELEM] delay+apod init imag part
//   dps[N_ELEM]       delay_phase_step per element
//
//   Total: N_ES*(5*4 + 8) + N_ELEM*3*4 bytes
//          = 64*(20+8) + 64*12 = 1792 + 768 = 2560 bytes (N_ES=64)
//
// Output: tx_re[N_SCAT * N_FREQ], tx_im[N_SCAT * N_FREQ]
//
// Compile-time constants:
//   N_ELEM, N_SUB, N_FREQ, N_ES, N_SCAT, TILE_SE, TG_SIZE, MAX_FPT

    threadgroup float sh_amp[N_ES];
    threadgroup float sh_kw_r[N_ES];
    threadgroup float sh_kr_step[N_ES];
    threadgroup float sh_alpha_r[N_ES];
    threadgroup float sh_ar_step[N_ES];
    threadgroup float2 sh_stp[N_ES];
    threadgroup float sh_da_init_re[N_ELEM];
    threadgroup float sh_da_init_im[N_ELEM];
    threadgroup float sh_dps[N_ELEM];

    uint scat_idx = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;

    if (scat_idx >= N_SCAT) return;

    float sx = scat_x[scat_idx];
    float sz = scat_z[scat_idx];
    float is_out_i = is_out[scat_idx];

    float kw_init_v    = scalars[0];
    float alpha_init_v = scalars[1];
    float kw_step_v    = scalars[2];
    float alpha_step_v = scalars[3];
    float min_dist     = scalars[4];
    float seg_len      = scalars[5];
    float center_kw    = scalars[6];
    float inv_nsub     = scalars[7];

    float lid_f    = float(lid);
    float stride_f = float(TG_SIZE);

    // ---- Phase 1A: Cooperatively compute per-sub-element geometry ----
    for (uint se = lid; se < (uint)N_ES; se += tpg) {
        int elem = se / N_SUB;
        int sub_global = elem * N_SUB + (se % N_SUB);

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

        sh_amp[se]     = obliq * sv / metal::precise::sqrt(rc_);
        sh_kw_r[se]    = kw_init_v * rc_;
        sh_kr_step[se] = kw_step_v * rc_;
        sh_alpha_r[se] = alpha_init_v * rc_;
        sh_ar_step[se] = alpha_step_v * rc_;

        // Precompute da-absorbed stride step (same for ALL threads).
        // stp = exp((-alpha_step*stride + j*kw_step*stride) * r) * da_step^stride
        float stp_phase = stride_f * kw_step_v * rc_;
        float stp_alpha = stride_f * alpha_step_v * rc_;
        float sm = metal::fast::exp(-stp_alpha);
        float sp_re = sm * metal::fast::cos(stp_phase);
        float sp_im = sm * metal::fast::sin(stp_phase);

        float das_phase = stride_f * delay_phase_step[elem];
        float das_re = metal::fast::cos(das_phase);
        float das_im = metal::fast::sin(das_phase);
        sh_stp[se] = float2(sp_re * das_re - sp_im * das_im,
                            sp_re * das_im + sp_im * das_re);
    }

    // ---- Phase 1B: Cooperatively load per-element da info ----
    for (uint e = lid; e < (uint)N_ELEM; e += tpg) {
        sh_da_init_re[e] = da_init_re[e];
        sh_da_init_im[e] = da_init_im[e];
        sh_dps[e] = delay_phase_step[e];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 2: Tiled progression sweep ----
    constexpr int N_TILES = (N_ES + TILE_SE - 1) / TILE_SE;

    float sum_re[MAX_FPT];
    float sum_im[MAX_FPT];
    int my_n_freq = 0;
    for (uint f = lid; f < (uint)N_FREQ; f += tpg) my_n_freq++;
    for (int i = 0; i < MAX_FPT; i++) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }

    for (int tile = 0; tile < N_TILES; tile++) {
        int tile_start = tile * TILE_SE;
        int tile_end = min(tile_start + TILE_SE, N_ES);
        int tile_len = tile_end - tile_start;

        float2 cur_t[TILE_SE];
        float2 stp_t[TILE_SE];

        // Init cur at this thread's starting frequency, read stp from shared
        for (int te = 0; te < tile_len; te++) {
            int se = tile_start + te;
            int elem = se / N_SUB;

            float phase = sh_kw_r[se] + lid_f * sh_kr_step[se];
            float alpha_val = sh_alpha_r[se] + lid_f * sh_ar_step[se];
            float ai = sh_amp[se] * metal::fast::exp(-alpha_val);
            float pi_re = ai * metal::fast::cos(phase);
            float pi_im = ai * metal::fast::sin(phase);

            float da_ph = lid_f * sh_dps[elem];
            float da_cs_re = metal::fast::cos(da_ph);
            float da_cs_im = metal::fast::sin(da_ph);
            float da_re = sh_da_init_re[elem] * da_cs_re - sh_da_init_im[elem] * da_cs_im;
            float da_im = sh_da_init_re[elem] * da_cs_im + sh_da_init_im[elem] * da_cs_re;

            cur_t[te] = float2(pi_re * da_re - pi_im * da_im,
                               pi_re * da_im + pi_im * da_re);
            stp_t[te] = sh_stp[se];
        }

        // Sweep: ALU-only inner loop
        for (int fi = 0; fi < my_n_freq; fi++) {
            for (int te = 0; te < tile_len; te++) {
                sum_re[fi] += cur_t[te].x;
                sum_im[fi] += cur_t[te].y;
                float cr = cur_t[te].x, ci = cur_t[te].y;
                float tr = stp_t[te].x, ti = stp_t[te].y;
                cur_t[te] = float2(cr * tr - ci * ti, cr * ti + ci * tr);
            }
        }
    }

    // ---- Phase 3: Apply inv_nsub, pulse*probe spectrum, write output ----
    int fi = 0;
    for (uint f = lid; f < (uint)N_FREQ; f += tpg, fi++) {
        float tx_re_v = sum_re[fi] * inv_nsub;
        float tx_im_v = sum_im[fi] * inv_nsub;

        float pp_re_f = pp_re[f], pp_im_f = pp_im[f];
        float pk_re = pp_re_f * tx_re_v - pp_im_f * tx_im_v;
        float pk_im = pp_re_f * tx_im_v + pp_im_f * tx_re_v;
        if (is_out_i > 0.5f) { pk_re = 0.0f; pk_im = 0.0f; }

        int out_idx = scat_idx * N_FREQ + f;
        tx_re[out_idx] = pk_re;
        tx_im[out_idx] = pk_im;
    }
