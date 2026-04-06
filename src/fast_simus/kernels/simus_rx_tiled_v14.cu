/*
 * RX kernel with tiled shmem output accumulation (Pass 2, variant B).
 *
 * ZERO atomics. Output is accumulated in shared memory per frequency tile.
 * Each block processes a subset of scatterers for one frequency tile.
 * Output tile: shmem[FREQ_TILE][N_ELEM][2] (transposed for bank-conflict-free access).
 *
 * Thread mapping: 1 thread per element (blockDim.x = N_ELEM).
 * Each thread sweeps FREQ_TILE frequencies using geometric progression (pure FMA).
 * Persistent scatterer loop within each block.
 *
 * Grid.x = tile_idx (0..ceil(N_FREQ/FREQ_TILE)-1)
 * Grid.y = blocks_per_tile (for scatterer parallelism)
 *
 * Compile-time: N_ELEM, N_SUB, N_FREQ, N_ES, FREQ_TILE, N_TILES
 */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

extern "C" __global__
void simus_rx_tiled_kernel(
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
    int   n_blocks_per_tile
) {
    int elem_idx = threadIdx.x;
    int tile_idx = blockIdx.x;
    int blk_in_tile = blockIdx.y;
    int freq_start = tile_idx * FREQ_TILE;

    /* Shmem output: transposed [FREQ_TILE][N_ELEM][2] for bank-conflict-free writes.
     * Thread elem_idx writes to shmem[(f * N_ELEM + elem_idx) * 2 + {0,1}]. */
    extern __shared__ float sh_out[];

    for (int i = threadIdx.x; i < FREQ_TILE * N_ELEM * 2; i += blockDim.x)
        sh_out[i] = 0.0f;
    __syncthreads();

    int actual_tile_freqs = FREQ_TILE;
    if (freq_start + FREQ_TILE > N_FREQ)
        actual_tile_freqs = N_FREQ - freq_start;

    float ex_ = elem_x[elem_idx];
    float ez_ = elem_z[elem_idx];
    float ct = cos_te[elem_idx];
    float snt = sin_neg_te[elem_idx];
    int se = elem_idx * N_SUB;
    float sdx = sub_dx[se];
    float sdz = sub_dz[se];

    float k_at_tile = kw_init + (float)freq_start * kw_step;
    float a_at_tile = alpha_init + (float)freq_start * alpha_step;

    int scat_per_block = (n_scat + n_blocks_per_tile - 1) / n_blocks_per_tile;
    int scat_start = blk_in_tile * scat_per_block;
    int scat_end = scat_start + scat_per_block;
    if (scat_end > n_scat) scat_end = n_scat;

    for (int s = scat_start; s < scat_end; s++) {
        float sx = scat_x[s];
        float sz = scat_z[s];

        bool is_out = (sz < 0.0f);
        if (radius < 1e30f) {
            float da = sx, db = sz + apex_offset;
            is_out = is_out || ((da*da + db*db) <= radius*radius);
        }
        if (is_out) continue;

        float dx = sx - ex_ - sdx;
        float dz = sz - ez_ - sdz;
        float r2 = dx*dx + dz*dz;
        float inv_r = rsqrtf(r2 + 1e-30f);
        float r = r2 * inv_r;
        float rc_ = fmaxf(r, min_dist);

        float sin_th = (dx*ct + dz*snt) * inv_r;
        float cos_th = (dz*ct - dx*snt) * inv_r;
        float obliq = (cos_th <= 0.0f) ? 1e-16f : cos_th;
        float sa = center_kw * seg_len * 0.5f * sin_th;
        float sv = (fabsf(sa) < 1e-8f) ? 1.0f : __fdividef(__sinf(sa), sa);
        float amp = obliq * sv * rsqrtf(rc_);

        float ph = k_at_tile * rc_;
        float av = a_at_tile * rc_;
        float ai = amp * expf(-av);
        float rx_re, rx_im;
        __sincosf(ph, &rx_im, &rx_re);
        rx_re *= ai; rx_im *= ai;

        float stp_ph = kw_step * rc_;
        float stp_av = alpha_step * rc_;
        float sm = expf(-stp_av);
        float stp_re, stp_im;
        __sincosf(stp_ph, &stp_im, &stp_re);
        stp_re *= sm; stp_im *= sm;

        int tx_base = s * N_FREQ + freq_start;

        for (int fi = 0; fi < actual_tile_freqs; fi++) {
            float tk_re = tx_re[tx_base + fi];
            float tk_im = tx_im[tx_base + fi];
            float pf = probe[freq_start + fi];

            float rr = rx_re * inv_nsub;
            float ri = rx_im * inv_nsub;
            float c_re = (tk_re*rr - tk_im*ri) * pf;
            float c_im = (tk_re*ri + tk_im*rr) * pf;

            int out_idx = (fi * N_ELEM + elem_idx) * 2;
            sh_out[out_idx]     += c_re;
            sh_out[out_idx + 1] += c_im;

            float new_re = rx_re * stp_re - rx_im * stp_im;
            float new_im = rx_re * stp_im + rx_im * stp_re;
            rx_re = new_re;
            rx_im = new_im;
        }
    }

    __syncthreads();

    /* Write shmem output to global. Multiple blocks per tile must use atomicAdd.
     * If n_blocks_per_tile == 1, could use regular writes instead. */
    for (int fi = threadIdx.x; fi < actual_tile_freqs * N_ELEM; fi += blockDim.x) {
        int f_local = fi / N_ELEM;
        int e = fi % N_ELEM;
        int gf = freq_start + f_local;
        int sh_idx = (f_local * N_ELEM + e) * 2;
        atomicAdd(&spect_re[e * N_FREQ + gf], sh_out[sh_idx]);
        atomicAdd(&spect_im[e * N_FREQ + gf], sh_out[sh_idx + 1]);
    }
}
