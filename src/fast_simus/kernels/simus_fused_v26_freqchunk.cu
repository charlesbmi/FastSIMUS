/*
 * Fused TX+RX SIMUS kernel -- v26: freq-chunked register TX (correct, fp32).
 *
 * Built on v25b. KEY CHANGE: outer loop over freq chunks shrinks the live
 * tk window from B*MAX_FPT*2 (98 floats at B=7) to B*CHUNK_FPT*2
 * (56 floats at B=7, M=2). Phase 2 runs per-chunk; Phase 3 reinits cv at
 * chunk_start.
 *
 * Numerical safety: cv at chunk_start is computed by initing at `lid_f`
 * (small-arg __sincosf, accurate) and then chain-advancing
 * `chunk * CHUNK_FPT` cmul steps. Direct __sincosf at the chunk_start
 * phase argument loses ~1e-3 absolute precision for far scatterers
 * because the input `ph` reaches 5K-8K radians, where __sincosf's
 * argument reduction degrades. The cmul chain accumulates ~1 ulp per
 * step, matching v25b's drift profile.
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, TILE_SE, TG_SIZE, MAX_FPT,
 *               B_SCAT, ELEM_TILE, N_CHUNKS (default 2).
 *
 * Shared memory: (7*B_SCAT*N_ES + 3*N_ELEM) * 4 bytes (same as v25b).
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

#ifndef N_CHUNKS
#define N_CHUNKS 2
#endif

#define CHUNK_FPT ((MAX_FPT + N_CHUNKS - 1) / N_CHUNKS)

struct f2 { float x, y; };

__device__ __forceinline__ f2 cmul(f2 a, f2 b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

#define GEO_AMP(s)       (shmem + ((0*B_SCAT + (s)) * N_ES))
#define GEO_KW_R(s)      (shmem + ((1*B_SCAT + (s)) * N_ES))
#define GEO_KR_STEP(s)   (shmem + ((2*B_SCAT + (s)) * N_ES))
#define GEO_ALPHA_R(s)   (shmem + ((3*B_SCAT + (s)) * N_ES))
#define GEO_AR_STEP(s)   (shmem + ((4*B_SCAT + (s)) * N_ES))
#define GEO_STP_RX_RE(s) (shmem + ((5*B_SCAT + (s)) * N_ES))
#define GEO_STP_RX_IM(s) (shmem + ((6*B_SCAT + (s)) * N_ES))

extern "C" __global__
void simus_fused_kernel(
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
    float lid_f = (float)lid;
    float stride_f = (float)TG_SIZE;

    extern __shared__ float shmem[];

    /* TX: only one chunk lives in regs at a time. */
    float tk_re[B_SCAT * CHUNK_FPT];
    float tk_im[B_SCAT * CHUNK_FPT];

    float* sh_da_init_re_l = shmem + 7 * B_SCAT * N_ES;
    float* sh_da_init_im_l = sh_da_init_re_l + N_ELEM;
    float* sh_dps_l        = sh_da_init_im_l + N_ELEM;

    for (int e = lid; e < N_ELEM; e += TG_SIZE) {
        sh_da_init_re_l[e] = da_init_re[e];
        sh_da_init_im_l[e] = da_init_im[e];
        sh_dps_l[e]        = dps[e];
    }
    __syncthreads();

    const int N_TILES = (N_ES + TILE_SE - 1) / TILE_SE;
    const int N_ELEM_GROUPS = (N_ES + ELEM_TILE - 1) / ELEM_TILE;

    bool out_flag[B_SCAT];

    for (int scat_base = blockIdx.x * B_SCAT;
         scat_base < n_scat;
         scat_base += gridDim.x * B_SCAT)
    {
        int actual_b = B_SCAT;
        if (scat_base + B_SCAT > n_scat)
            actual_b = n_scat - scat_base;

        /* ---- Phase 1: geometry for each scatterer in batch (once) ---- */
        for (int si = 0; si < actual_b; si++) {
            int scat_idx = scat_base + si;
            float sx = scat_x[scat_idx];
            float sz = scat_z[scat_idx];

            bool is_out = (sz < 0.0f);
            if (radius < 1e30f) {
                float da = sx, db = sz + apex_offset;
                is_out = is_out || ((da*da + db*db) <= radius*radius);
            }
            out_flag[si] = is_out;

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

                GEO_AMP(si)[se]       = obliq * sv * rsqrtf(rc_);
                GEO_KW_R(si)[se]      = kw_init * rc_;
                GEO_KR_STEP(si)[se]   = kw_step * rc_;
                GEO_ALPHA_R(si)[se]   = alpha_init * rc_;
                GEO_AR_STEP(si)[se]   = alpha_step * rc_;

                float stp_phase = stride_f * kw_step * rc_;
                float stp_alpha = stride_f * alpha_step * rc_;
                float sm = expf(-stp_alpha);
                float sp_re, sp_im;
                __sincosf(stp_phase, &sp_im, &sp_re);
                sp_re *= sm; sp_im *= sm;
                GEO_STP_RX_RE(si)[se] = sp_re;
                GEO_STP_RX_IM(si)[se] = sp_im;
            }
        }
        __syncthreads();

        /* ---- Phase 2 + Phase 3: chunked over freqs ---- */
        #pragma unroll
        for (int chunk = 0; chunk < N_CHUNKS; chunk++) {
            const int chunk_fi_start = chunk * CHUNK_FPT;

            /* === Phase 2 chunk: produce tk_re/tk_im[B][CHUNK_FPT] === */
            for (int si = 0; si < actual_b; si++) {
                int scat_idx = scat_base + si;
                float rc = rc_arr[scat_idx];

                if (out_flag[si]) {
                    #pragma unroll
                    for (int fi_local = 0; fi_local < CHUNK_FPT; fi_local++) {
                        tk_re[si * CHUNK_FPT + fi_local] = 0.0f;
                        tk_im[si * CHUNK_FPT + fi_local] = 0.0f;
                    }
                    continue;
                }

                float sum_re[CHUNK_FPT], sum_im[CHUNK_FPT];
                #pragma unroll
                for (int i = 0; i < CHUNK_FPT; i++) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }

                for (int tile = 0; tile < N_TILES; tile++) {
                    int ts = tile * TILE_SE;
                    int te = ts + TILE_SE;
                    if (te > N_ES) te = N_ES;
                    int tl = te - ts;

                    /* Init cv at lid_f (small arg, accurate) then chain-advance
                     * chunk_fi_start steps to reach this chunk's start freq. */
                    f2 cv[TILE_SE], sv[TILE_SE];
                    #pragma unroll
                    for (int j = 0; j < TILE_SE; j++) {
                        if (j >= tl) { cv[j] = {0.0f, 0.0f}; sv[j] = {1.0f, 0.0f}; continue; }
                        int se = ts + j, em = se / N_SUB;
                        float ph = GEO_KW_R(si)[se] + lid_f * GEO_KR_STEP(si)[se];
                        float av = GEO_ALPHA_R(si)[se] + lid_f * GEO_AR_STEP(si)[se];
                        float ai = GEO_AMP(si)[se] * expf(-av);
                        float vr, vi;
                        __sincosf(ph, &vi, &vr);
                        vr *= ai; vi *= ai;
                        float dp = lid_f * sh_dps_l[em];
                        float dr, di;
                        __sincosf(dp, &di, &dr);
                        float dvr = sh_da_init_re_l[em]*dr - sh_da_init_im_l[em]*di;
                        float dvi = sh_da_init_re_l[em]*di + sh_da_init_im_l[em]*dr;
                        cv[j] = {vr*dvr - vi*dvi, vr*dvi + vi*dvr};

                        float sp_re = GEO_STP_RX_RE(si)[se];
                        float sp_im = GEO_STP_RX_IM(si)[se];
                        float das_phase = stride_f * sh_dps_l[em];
                        float das_re, das_im;
                        __sincosf(das_phase, &das_im, &das_re);
                        sv[j] = {sp_re*das_re - sp_im*das_im, sp_re*das_im + sp_im*das_re};
                    }

                    /* Advance cv from fi=0 to fi=chunk_fi_start via cmul.
                     * For chunk 0, this loop does zero iterations. */
                    #pragma unroll
                    for (int adv = 0; adv < N_CHUNKS * CHUNK_FPT; adv++) {
                        if (adv >= chunk_fi_start) break;
                        #pragma unroll
                        for (int j = 0; j < TILE_SE; j++) {
                            cv[j] = cmul(cv[j], sv[j]);
                        }
                    }

                    #pragma unroll
                    for (int fi_local = 0; fi_local < CHUNK_FPT; fi_local++) {
                        int fi_global = chunk_fi_start + fi_local;
                        int f_chk = lid + fi_global * TG_SIZE;
                        if (fi_global >= MAX_FPT || f_chk >= N_FREQ) break;
                        #pragma unroll
                        for (int j = 0; j < TILE_SE; j++) {
                            sum_re[fi_local] += cv[j].x;
                            sum_im[fi_local] += cv[j].y;
                            cv[j] = cmul(cv[j], sv[j]);
                        }
                    }
                }

                #pragma unroll
                for (int fi_local = 0; fi_local < CHUNK_FPT; fi_local++) {
                    int fi_global = chunk_fi_start + fi_local;
                    int f = lid + fi_global * TG_SIZE;
                    bool valid = (fi_global < MAX_FPT) && (f < N_FREQ);
                    float tr = sum_re[fi_local] * inv_nsub;
                    float ti = sum_im[fi_local] * inv_nsub;
                    float ppr = valid ? pp_re[f] : 0.0f;
                    float ppi = valid ? pp_im[f] : 0.0f;
                    tk_re[si * CHUNK_FPT + fi_local] = valid ? (ppr*tr - ppi*ti) * rc : 0.0f;
                    tk_im[si * CHUNK_FPT + fi_local] = valid ? (ppr*ti + ppi*tr) * rc : 0.0f;
                }
            }
            /* Pad tk for si in [actual_b, B_SCAT). */
            #pragma unroll
            for (int si = 0; si < B_SCAT; si++) {
                if (si < actual_b) continue;
                #pragma unroll
                for (int fi_local = 0; fi_local < CHUNK_FPT; fi_local++) {
                    tk_re[si * CHUNK_FPT + fi_local] = 0.0f;
                    tk_im[si * CHUNK_FPT + fi_local] = 0.0f;
                }
            }

            /* === Phase 3 chunk: RX accumulation over chunk freqs === */
            for (int eg = 0; eg < N_ELEM_GROUPS; eg++) {
                int se_base = eg * ELEM_TILE;
                int etl = ELEM_TILE;
                if (se_base + etl > N_ES) etl = N_ES - se_base;

                /* Init at lid_f (accurate) then chain-advance chunk_fi_start
                 * cmul steps to reach this chunk's start freq. */
                f2 cv[B_SCAT * ELEM_TILE];
                f2 sv_arr[B_SCAT * ELEM_TILE];

                #pragma unroll
                for (int si = 0; si < B_SCAT; si++) {
                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        int idx = si * ELEM_TILE + et;
                        if (si >= actual_b || out_flag[si] || et >= etl) {
                            cv[idx] = {0.0f, 0.0f};
                            sv_arr[idx] = {1.0f, 0.0f};
                            continue;
                        }
                        int se = se_base + et;
                        float ph = GEO_KW_R(si)[se] + lid_f * GEO_KR_STEP(si)[se];
                        float av = GEO_ALPHA_R(si)[se] + lid_f * GEO_AR_STEP(si)[se];
                        float ai = GEO_AMP(si)[se] * expf(-av);
                        float vr, vi;
                        __sincosf(ph, &vi, &vr);
                        cv[idx] = {vr * ai, vi * ai};
                        sv_arr[idx] = {GEO_STP_RX_RE(si)[se], GEO_STP_RX_IM(si)[se]};
                    }
                }

                /* Advance Phase 3 cv from fi=0 to fi=chunk_fi_start via cmul.
                 * Zero-iteration loop for chunk 0. */
                #pragma unroll
                for (int adv = 0; adv < N_CHUNKS * CHUNK_FPT; adv++) {
                    if (adv >= chunk_fi_start) break;
                    #pragma unroll
                    for (int idx = 0; idx < B_SCAT * ELEM_TILE; idx++) {
                        cv[idx] = cmul(cv[idx], sv_arr[idx]);
                    }
                }

                #pragma unroll
                for (int fi_local = 0; fi_local < CHUNK_FPT; fi_local++) {
                    int fi_global = chunk_fi_start + fi_local;
                    int f = lid + fi_global * TG_SIZE;
                    bool valid = (fi_global < MAX_FPT) && (f < N_FREQ);
                    float pf = valid ? probe[f] : 0.0f;

                    float acc_re[ELEM_TILE];
                    float acc_im[ELEM_TILE];
                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        acc_re[et] = 0.0f;
                        acc_im[et] = 0.0f;
                    }

                    #pragma unroll
                    for (int si = 0; si < B_SCAT; si++) {
                        float tkr = tk_re[si * CHUNK_FPT + fi_local];
                        float tki = tk_im[si * CHUNK_FPT + fi_local];

                        #pragma unroll
                        for (int et = 0; et < ELEM_TILE; et++) {
                            int idx = si * ELEM_TILE + et;
                            float rr = cv[idx].x * inv_nsub;
                            float ri = cv[idx].y * inv_nsub;
                            acc_re[et] += (tkr*rr - tki*ri) * pf;
                            acc_im[et] += (tkr*ri + tki*rr) * pf;
                            cv[idx] = cmul(cv[idx], sv_arr[idx]);
                        }
                    }

                    if (!valid) continue;
                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        if (et >= etl) break;
                        int elem = (se_base + et) / N_SUB;
                        atomicAdd(&spect_re[elem * N_FREQ + f], acc_re[et]);
                        atomicAdd(&spect_im[elem * N_FREQ + f], acc_im[et]);
                    }
                }
            }
        }

        __syncthreads();
    }
}
