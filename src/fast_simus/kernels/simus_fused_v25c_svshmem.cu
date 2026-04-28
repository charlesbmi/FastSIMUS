/*
 * Fused TX+RX SIMUS kernel -- v25c: register-resident TX, sv_arr in
 *  shmem (correct, fp32).
 *
 * v25b cached sv_arr[B*ELEM_TILE] in registers (56 floats at B=7 ET=4).
 * That competes with tk_re/tk_im for the 255-reg cap and forces 400 B
 * spill of tk into local memory -- which NCU showed saturates L2 at
 * 76.5 % throughput.
 *
 * v25c drops sv_arr from registers and reads GEO_STP_RX_RE/IM directly
 * from shmem inside the cmul, freeing those 56 regs for tk. Cost: an
 * extra shmem read per (si, et, fi) cmul advance -- bank-conflict free
 * since each thread reads its own row.
 *
 * v25 with one structural fix: every loop that indexes tk_re/tk_im is
 * now `for fi in 0..MAX_FPT` with `#pragma unroll` and predicated
 * validity, so fi is statically known. v25 spilled tk entirely
 * (576 B local mem at B=9 ET=4) because dynamic fi forced
 * tk_re[si*MAX_FPT + fi] off-register. Static fi unrolling makes tk
 * actually register-resident, eliminating the local-memory traffic.
 *
 * Why this is safe: in v11 each thread `lid` writes to sh_tx[si*N_FREQ + f]
 * for f in {lid, lid+TG_SIZE, lid+2*TG_SIZE, ...} during Phase 2, and reads
 * the same slots during Phase 3. There is no cross-thread sharing of TX --
 * the shmem allocation was a temporary, not a broadcast surface.
 *
 * Storing TX in per-thread register arrays tk_re[B_SCAT*MAX_FPT],
 * tk_im[B_SCAT*MAX_FPT] eliminates the dominant shmem cost
 * (2*B_SCAT*N_FREQ floats; 60 KB at B=9 N_FREQ=854) without changing
 * precision or arithmetic. Also lets us drop the pre-Phase-3 sync that
 * was only needed to publish sh_tx writes.
 *
 * Per-thread TX register cost: 2*B_SCAT*MAX_FPT floats. For
 * B=9 N_FREQ=854 TG=128 -> MAX_FPT=7 -> 126 floats. May force 1->something
 * trade vs spill; expected to come out well ahead given shmem savings
 * (76.5 KB -> 16.5 KB at B=9 ET=4 unlocks ~5 blk/SM vs v11's 1).
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, TILE_SE, TG_SIZE, MAX_FPT,
 *               B_SCAT, ELEM_TILE
 *
 * Shared memory: (7*B_SCAT*N_ES + 3*N_ELEM) * 4 bytes
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

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

    /* TX moved to per-thread registers; shmem only holds geometry + per-elem broadcast. */
    float tk_re[B_SCAT * MAX_FPT];
    float tk_im[B_SCAT * MAX_FPT];

    float* sh_da_init_re_l = shmem + 7 * B_SCAT * N_ES;
    float* sh_da_init_im_l = sh_da_init_re_l + N_ELEM;
    float* sh_dps_l        = sh_da_init_im_l + N_ELEM;

    for (int e = lid; e < N_ELEM; e += TG_SIZE) {
        sh_da_init_re_l[e] = da_init_re[e];
        sh_da_init_im_l[e] = da_init_im[e];
        sh_dps_l[e]        = dps[e];
    }
    __syncthreads();

    int my_n_freq = 0;
    for (int f = lid; f < N_FREQ; f += TG_SIZE) my_n_freq++;

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

        /* Zero-init register TX so si >= actual_b reads finite zeros in
         * Phase 3 (cv is also forced to 0 there, but 0*NaN would propagate). */
        #pragma unroll
        for (int si = 0; si < B_SCAT; si++) {
            #pragma unroll
            for (int fi = 0; fi < MAX_FPT; fi++) {
                tk_re[si * MAX_FPT + fi] = 0.0f;
                tk_im[si * MAX_FPT + fi] = 0.0f;
            }
        }

        /* ---- Phase 1+2: geometry + TX for each scatterer in batch ---- */
        for (int si = 0; si < actual_b; si++) {
            int scat_idx = scat_base + si;
            float sx = scat_x[scat_idx];
            float sz = scat_z[scat_idx];
            float rc = rc_arr[scat_idx];

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
            __syncthreads();

            if (is_out) {
                #pragma unroll
                for (int fi = 0; fi < MAX_FPT; fi++) {
                    tk_re[si * MAX_FPT + fi] = 0.0f;
                    tk_im[si * MAX_FPT + fi] = 0.0f;
                }
                continue;
            }

            /* Phase 2: TX sweep */
            float sum_re[MAX_FPT], sum_im[MAX_FPT];
            for (int i = 0; i < MAX_FPT; i++) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }

            for (int tile = 0; tile < N_TILES; tile++) {
                int ts = tile * TILE_SE;
                int te = ts + TILE_SE;
                if (te > N_ES) te = N_ES;
                int tl = te - ts;

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

                #pragma unroll
                for (int fi = 0; fi < MAX_FPT; fi++) {
                    int f_chk = lid + fi * TG_SIZE;
                    if (f_chk >= N_FREQ) break;
                    #pragma unroll
                    for (int j = 0; j < TILE_SE; j++) {
                        sum_re[fi] += cv[j].x; sum_im[fi] += cv[j].y;
                        cv[j] = cmul(cv[j], sv[j]);
                    }
                }
            }

            #pragma unroll
            for (int fi = 0; fi < MAX_FPT; fi++) {
                int f = lid + fi * TG_SIZE;
                bool valid = (f < N_FREQ);
                float tr = sum_re[fi] * inv_nsub;
                float ti = sum_im[fi] * inv_nsub;
                float ppr = valid ? pp_re[f] : 0.0f;
                float ppi = valid ? pp_im[f] : 0.0f;
                tk_re[si * MAX_FPT + fi] = valid ? (ppr*tr - ppi*ti) * rc : 0.0f;
                tk_im[si * MAX_FPT + fi] = valid ? (ppr*ti + ppi*tr) * rc : 0.0f;
            }
            /* No __syncthreads here -- TX is private to this thread. */
        }

        /* ---- Phase 3: element-tiled RX with B_SCAT accumulation ---- */
        for (int eg = 0; eg < N_ELEM_GROUPS; eg++) {
            int se_base = eg * ELEM_TILE;
            int etl = ELEM_TILE;
            if (se_base + etl > N_ES) etl = N_ES - se_base;

            /* Initialize B_SCAT * ELEM_TILE RX states. sv_arr stays in shmem
             * (re-read per cmul advance) to free registers for tk. */
            f2 cv[B_SCAT * ELEM_TILE];

            #pragma unroll
            for (int si = 0; si < B_SCAT; si++) {
                #pragma unroll
                for (int et = 0; et < ELEM_TILE; et++) {
                    int idx = si * ELEM_TILE + et;
                    if (si >= actual_b || out_flag[si] || et >= etl) {
                        cv[idx] = {0.0f, 0.0f};
                        continue;
                    }
                    int se = se_base + et;
                    float ph = GEO_KW_R(si)[se] + lid_f * GEO_KR_STEP(si)[se];
                    float av = GEO_ALPHA_R(si)[se] + lid_f * GEO_AR_STEP(si)[se];
                    float ai = GEO_AMP(si)[se] * expf(-av);
                    float vr, vi;
                    __sincosf(ph, &vi, &vr);
                    cv[idx] = {vr * ai, vi * ai};
                }
            }

            /* Sweep frequencies with B_SCAT * ELEM_TILE independent chains.
             * fi is statically unrolled so tk_re[si*MAX_FPT + fi] uses a
             * compile-time index, keeping tk truly register-resident. */
            #pragma unroll
            for (int fi = 0; fi < MAX_FPT; fi++) {
                int f = lid + fi * TG_SIZE;
                bool valid = (f < N_FREQ);
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
                    float tkr = tk_re[si * MAX_FPT + fi];
                    float tki = tk_im[si * MAX_FPT + fi];

                    #pragma unroll
                    for (int et = 0; et < ELEM_TILE; et++) {
                        int idx = si * ELEM_TILE + et;
                        int se = se_base + et;
                        float rr = cv[idx].x * inv_nsub;
                        float ri = cv[idx].y * inv_nsub;
                        acc_re[et] += (tkr*rr - tki*ri) * pf;
                        acc_im[et] += (tkr*ri + tki*rr) * pf;
                        f2 sv_local = {GEO_STP_RX_RE(si)[se], GEO_STP_RX_IM(si)[se]};
                        cv[idx] = cmul(cv[idx], sv_local);
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

        __syncthreads();
    }
}
