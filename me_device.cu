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
  // Map 1D thread index to 2D macroblock row-major matrix index
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_mbs = mb_cols * mb_rows;
  if (tid >= num_mbs)
    return;

  int mb_x = tid % mb_cols;
  int mb_y = tid / mb_cols;
  struct macroblock *mb = &mbs[tid];

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

  int mx = mb_x * 8;
  int my = mb_y * 8;

  // Load the single orig block into stack memory
  uint8_t orig_block[64];
#pragma unroll
  for (int v = 0; v < 8; ++v) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      orig_block[v * 8 + u] = orig[(my + v) * w + (mx + u)];
    }
  }

  int best_sad = INT_MAX;
  // Write to stack memory, then write to managed memory at the end
  int best_mv_x = 0;
  int best_mv_y = 0;

  for (y = top; y < bottom; ++y) {
    for (x = left; x < right; ++x) {
      int sad;
      sad_block_8x8_device(orig_block, ref + y * w + x, 8, w, &sad);

      if (sad < best_sad) {
        best_mv_x = x - mx;
        best_mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->mv_x = (int8_t)best_mv_x;
  mb->mv_y = (int8_t)best_mv_y;
  mb->use_mv = 1;
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
  const int tpb = 64;
  size_t nY = (size_t)a.mb_cols_Y * a.mb_rows_Y;
  size_t nC = (size_t)a.mb_cols_C * a.mb_rows_C;
  size_t bY = (nY + tpb - 1) / tpb;
  size_t bC = (nC + tpb - 1) / tpb;

  nvtxRangePushA("motion_estimation");
  me_block_8x8_kernel<<<bY, tpb, 0, stream_y>>>(a.orig_Y, a.recons_Y, a.mbs_Y,
                                                a.mb_cols_Y, a.mb_rows_Y, a.w_Y,
                                                a.h_Y, a.range_Y);

  me_block_8x8_kernel<<<bC, tpb, 0, stream_u>>>(a.orig_U, a.recons_U, a.mbs_U,
                                                a.mb_cols_C, a.mb_rows_C, a.w_U,
                                                a.h_U, a.range_C);

  me_block_8x8_kernel<<<bC, tpb, 0, stream_v>>>(a.orig_V, a.recons_V, a.mbs_V,
                                                a.mb_cols_C, a.mb_rows_C, a.w_V,
                                                a.h_V, a.range_C);

  nvtxRangePop();

  nvtxRangePushA("motion_compensation");
  mc_block_8x8_kernel<<<bY, tpb, 0, stream_y>>>(
      a.pred_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y);

  mc_block_8x8_kernel<<<bC, tpb, 0, stream_u>>>(
      a.pred_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U);

  mc_block_8x8_kernel<<<bC, tpb, 0, stream_v>>>(
      a.pred_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V);
  nvtxRangePop();
}