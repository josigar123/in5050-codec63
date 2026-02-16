#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "me_device.h"

/*
block1 and block2 points to the beginning of a 8x8 block in the entire frame.
stride is the widht of the entire frame in pixels (bytes).

v * stride is used to calculate the beginning of the next row. u is then used to
index the columns of the current row.

Two strides since orig is 8x8 contiguous in memory NOT the entire frame.
ref is the entire frame.
*/
__device__ __forceinline__ static void
sad_block_8x8_device(const uint8_t *block1, const uint8_t *block2, int stride1,
                     int stride2, int *result) {
  *result = 0;
#pragma unroll
  for (int v = 0; v < 8; ++v) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      *result += abs(block2[v * stride2 + u] - block1[v * stride1 + u]);
    }
  }
}

/* Motion estimation for 8x8 block */
/*
  We take in all necessary parameters so we dont dereference directly from cm as
  that will cause segfaults upon launching kernels.
*/
__global__ static void me_block_8x8_kernel(const uint8_t *orig,
                                           const uint8_t *ref,
                                           struct macroblock *mbs, int mb_cols,
                                           int mb_rows, int w, int h,
                                           int range) {

  // Warp based indexing
  int lane = threadIdx.x & 31;    // 0..31 in warp
  int warp_id = threadIdx.x >> 5; // warp index inside block
  int warps_per_block = blockDim.x >> 5;

  int num_mbs = mb_cols * mb_rows;
  int mb_linear = blockIdx.x * warps_per_block + warp_id;
  if (mb_linear >= num_mbs)
    return;

  int mb_x = mb_linear % mb_cols;
  int mb_y = mb_linear / mb_cols;
  macroblock *mb = &mbs[mb_linear];

  // Potential bug, might need to divide by 8 when calculating bounding box
  // below
  int mx = mb_x * 8;
  int my = mb_y * 8;

  // Two pixels in 8x8 block
  int p0 = lane;
  int p1 = lane + 32;

  int u0 = p0 & 7, v0 = p0 >> 3;
  int u1 = p1 & 7, v1 = p1 >> 3;

  // Cache orig pixels for current lane
  int o0 = (int)orig[(my + v0) * w + (mx + u0)];
  int o1 = (int)orig[(my + v1) * w + (mx + u1)];

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  if (left < 0) {
    left = 0;
  }
  if (top < 0) {
    top = 0;
  }
  if (right > (w - 8)) {
    right = w - 8;
  }
  if (bottom > (h - 8)) {
    bottom = h - 8;
  }

  int x, y;

  int best_sad = INT_MAX;
  // Write to stack memory, then write to managed memory at the end
  int best_mv_x = 0;
  int best_mv_y = 0;

  unsigned mask = 0xffffffffu;
  for (y = top; y < bottom; ++y) {
    for (x = left; x < right; ++x) {
      const uint8_t *ref_block = ref + y * w + x;

      // Calculate SAD
      int local_sad = abs((int)ref_block[v0 * w + u0] - o0) +
                      abs((int)ref_block[v1 * w + u1] - o1);

      // Reduce with warp shuffle across lanes
      int sad = local_sad;
      sad += __shfl_down_sync(mask, sad, 16);
      sad += __shfl_down_sync(mask, sad, 8);
      sad += __shfl_down_sync(mask, sad, 4);
      sad += __shfl_down_sync(mask, sad, 2);
      sad += __shfl_down_sync(mask, sad, 1);

      if (lane == 0 && sad < best_sad) {
        best_sad = sad;
        best_mv_x = x - mx;
        best_mv_y = y - my;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  if (lane == 0) {
    mb->mv_x = (int8_t)best_mv_x;
    mb->mv_y = (int8_t)best_mv_y;
    mb->use_mv = 1;
  }
}

// DANGER: Is also used by the decoder
/* Motion compensation for 8x8 block */
__global__ static void mc_block_8x8_kernel(uint8_t *predicted,
                                           const uint8_t *ref,
                                           const struct macroblock *mbs,
                                           int mb_cols, int mb_rows, int w) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_mbs = mb_cols * mb_rows;
  if (tid >= num_mbs)
    return;

  int mb_x = tid % mb_cols;
  int mb_y = tid / mb_cols;

  const struct macroblock *mb = &mbs[tid];
  if (!mb->use_mv)
    return;

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y) {
    for (x = left; x < right; ++x) {
      predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void launch_motion_inter(const motion_inter_args &a, cudaStream_t stream_y,
                         cudaStream_t stream_u, cudaStream_t stream_v) {

  const int warps_per_block = 1;
  const int tpb = 32 * warps_per_block;
  size_t mbs_Y = (size_t)a.mb_cols_Y * a.mb_rows_Y;
  size_t mbs_C = (size_t)a.mb_cols_C * a.mb_rows_C;
  size_t blocks_Y = (mbs_Y + tpb - 1) / tpb;
  size_t blocks_C = (mbs_C + tpb - 1) / tpb;

  size_t blocks_Y1 = (mbs_Y + warps_per_block - 1) / warps_per_block;
  size_t blocks_C1 = (mbs_C + warps_per_block - 1) / warps_per_block;

  nvtxRangePushA("motion_estimation");
  me_block_8x8_kernel<<<blocks_Y1, tpb, 0, stream_y>>>(
      a.orig_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y, a.h_Y,
      a.range_Y);

  me_block_8x8_kernel<<<blocks_C1, tpb, 0, stream_u>>>(
      a.orig_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U, a.h_U,
      a.range_C);

  me_block_8x8_kernel<<<blocks_C1, tpb, 0, stream_v>>>(
      a.orig_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V, a.h_V,
      a.range_C);

  nvtxRangePop();

  nvtxRangePushA("motion_compensation");
  mc_block_8x8_kernel<<<blocks_Y, 64, 0, stream_y>>>(
      a.pred_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y);

  mc_block_8x8_kernel<<<blocks_C, 64, 0, stream_u>>>(
      a.pred_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U);

  mc_block_8x8_kernel<<<blocks_C, 64, 0, stream_v>>>(
      a.pred_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V);
  nvtxRangePop();
}