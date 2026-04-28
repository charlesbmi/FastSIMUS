/*
 * RX-only kernel with Grid-Y element group partitioning (v18).
 *
 * Designed for RTX 4090's 72MB L2: TX buffer streamed from L2, output (438KB)
 * fully L2-resident. Each y-block handles ONE element group, reducing atomic
 * contention on output by N_ELEM_GROUPS (8x with ELEM_TILE=8).
 *
 * Grid: (blocks_x, N_ELEM_GROUPS, 1)
 * Block: (TG_SIZE, 1, 1) = 128 threads
 *
 * Per block:
 *   - blockIdx.y selects element group (ELEM_TILE elements)
 *   - Persistent scatterer loop across blockIdx.x stride
 *   - Reads TX[scat][freq] from global memory (L2-resident via chunking)
 *   - RX geometric progression for ELEM_TILE elements
 *   - atomicAdd to output (only this element group's entries)
 *
 * Shmem: 5 * ELEM_TILE * 4 = 160 bytes (geometry cache per scatterer)
 *   Minimal shmem allows high occupancy.
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, TG_SIZE, MAX_FPT, ELEM_TILE
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct f2 { float x, y; };

__device__ __forceinline__ f2 cmul(f2 a, f2 b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

extern "C" __global__
void simus_rx_gridy_kernel(
    const float* __restrict__ scat_x,
    const float* __restrict__ scat_z,
    const float* __restrict__ elem_x,
    const float* __restrict__ elem_z,
    const float* __restrict__ cos_te,
    const float* __restrict__ sin_neg_te,
    const float* __restrict__ sub_dx,
    const float* __restrict__ sub_dz,
    const float* __restrict__ tx_re,
    const float* __restrict__ tx_im,
    const float* __restrict__ probe,
    float* __restrict__ spect_re,
    float* __restrict__ spect_im,
    int   n_scat,
    float kw_init, float alpha_init,
    float kw_step, float alpha_step,
    float min_dist, float seg_len,
    float center_kw, float inv_nsub,
    float radius, float apex_offset,
    int   scat_offset
) {
    const int lid = threadIdx.x;
    const float lid_f = (float)lid;
    const float stride_f = (float)TG_SIZE;

    const int eg = blockIdx.y;
    const int se_base = eg * ELEM_TILE;
    int etl = ELEM_TILE;
    if (se_base + etl > N_ES) etl = N_ES - se_base;

    int my_n_freq = 0;
    for (int f = lid; f < N_FREQ; f += TG_SIZE) my_n_freq++;

    for (int scat_idx = blockIdx.x + scat_offset;
         scat_idx < scat_offset + n_scat;
         scat_idx += gridDim.x)
    {
        float sx = scat_x[scat_idx];
        float sz = scat_z[scat_idx];

        bool is_out = (sz < 0.0f);
        if (radius < 1e30f) {
            float da = sx, db = sz + apex_offset;
            is_out = is_out || ((da*da + db*db) <= radius*radius);
        }
        if (is_out) continue;

        f2 cv[ELEM_TILE];
        f2 sv_arr[ELEM_TILE];

        #pragma unroll
        for (int et = 0; et < ELEM_TILE; et++) {
            if (et >= etl) {
                cv[et] = {0.0f, 0.0f};
                sv_arr[et] = {1.0f, 0.0f};
                continue;
            }
            int se = se_base + et;
            int elem = se / N_SUB;
            float ex_ = elem_x[elem], ez_ = elem_z[elem];
            float ct = cos_te[elem], snt = sin_neg_te[elem];
            float ddx = sx - ex_ - sub_dx[se];
            float ddz = sz - ez_ - sub_dz[se];
            float r2 = ddx*ddx + ddz*ddz;
            float inv_r = rsqrtf(r2 + 1e-30f);
            float r = r2 * inv_r;
            float rc_ = fmaxf(r, min_dist);

            float sin_th = (ddx*ct + ddz*snt) * inv_r;
            float cos_th = (ddz*ct - ddx*snt) * inv_r;
            float obliq = (cos_th <= 0.0f) ? 1e-16f : cos_th;
            float sa = center_kw * seg_len * 0.5f * sin_th;
            float svv = (fabsf(sa) < 1e-8f) ? 1.0f : __fdividef(__sinf(sa), sa);
            float amp = obliq * svv * rsqrtf(rc_);

            float ph = kw_init * rc_ + lid_f * kw_step * rc_;
            float av = alpha_init * rc_ + lid_f * alpha_step * rc_;
            float ai = amp * expf(-av);
            float vr, vi;
            __sincosf(ph, &vi, &vr);
            cv[et] = {vr * ai, vi * ai};

            float stp_ph = stride_f * kw_step * rc_;
            float stp_al = stride_f * alpha_step * rc_;
            float sm = expf(-stp_al);
            float spr, spi;
            __sincosf(stp_ph, &spi, &spr);
            sv_arr[et] = {spr * sm, spi * sm};
        }

        int tx_base = scat_idx * N_FREQ;

        for (int fi = 0; fi < my_n_freq; fi++) {
            int f = lid + fi * TG_SIZE;
            if (f >= N_FREQ) break;
            float pf = probe[f];
            float tk_re = tx_re[tx_base + f];
            float tk_im = tx_im[tx_base + f];

            #pragma unroll
            for (int et = 0; et < ELEM_TILE; et++) {
                if (et >= etl) break;
                float rr = cv[et].x * inv_nsub;
                float ri = cv[et].y * inv_nsub;
                float c_re = (tk_re*rr - tk_im*ri) * pf;
                float c_im = (tk_re*ri + tk_im*rr) * pf;
                int elem = (se_base + et) / N_SUB;
                atomicAdd(&spect_re[elem * N_FREQ + f], c_re);
                atomicAdd(&spect_im[elem * N_FREQ + f], c_im);
                cv[et] = cmul(cv[et], sv_arr[et]);
            }
        }
    }
}
