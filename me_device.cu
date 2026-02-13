#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "me.h"

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

void c63_motion_inter(struct c63_common *cm) {
  nvtxRangePushA("c63_motion_inter");

  const int threads_per_block = 64;

  // Extract all necessary data from cm
  const uint8_t *orig_Y = cm->curframe->orig->Y;
  const uint8_t *orig_U = cm->curframe->orig->U;
  const uint8_t *orig_V = cm->curframe->orig->V;

  const uint8_t *recons_Y = cm->refframe->recons->Y;
  const uint8_t *recons_U = cm->refframe->recons->U;
  const uint8_t *recons_V = cm->refframe->recons->V;

  struct macroblock *mbs_Y = cm->curframe->mbs[Y_COMPONENT];
  struct macroblock *mbs_U = cm->curframe->mbs[U_COMPONENT];
  struct macroblock *mbs_V = cm->curframe->mbs[V_COMPONENT];

  uint8_t *pred_Y = cm->curframe->predicted->Y;
  uint8_t *pred_U = cm->curframe->predicted->U;
  uint8_t *pred_V = cm->curframe->predicted->V;

  int mb_cols_Y = cm->mb_cols;
  int mb_rows_Y = cm->mb_rows;
  int mb_cols_C = cm->mb_cols / 2;
  int mb_rows_C = cm->mb_rows / 2;

  int w_Y = cm->padw[Y_COMPONENT], h_Y = cm->padh[Y_COMPONENT];
  int w_U = cm->padw[U_COMPONENT], h_U = cm->padh[U_COMPONENT];
  int w_V = cm->padw[V_COMPONENT], h_V = cm->padh[V_COMPONENT];

  int range_Y = cm->me_search_range;
  int range_C = cm->me_search_range / 2;

  size_t nY = (size_t)mb_cols_Y * mb_rows_Y;
  size_t nC = (size_t)mb_cols_C * mb_rows_C;
  size_t bY = (nY + threads_per_block - 1) / threads_per_block;
  size_t bC = (nC + threads_per_block - 1) / threads_per_block;

  // Motion Estimation
  nvtxRangePushA("motion_estimation");
  // Luma
  me_block_8x8_kernel<<<bY, threads_per_block>>>(
      orig_Y, recons_Y, mbs_Y, mb_cols_Y, mb_rows_Y, w_Y, h_Y, range_Y);

  // Chroma
  me_block_8x8_kernel<<<bC, threads_per_block>>>(
      orig_U, recons_U, mbs_U, mb_cols_C, mb_rows_C, w_U, h_U, range_C);
  me_block_8x8_kernel<<<bC, threads_per_block>>>(
      orig_V, recons_V, mbs_V, mb_cols_C, mb_rows_C, w_V, h_V, range_C);

  nvtxRangePop();

  nvtxRangePushA("motion_compensation");
  // Motion Compensation
  // Luma
  mc_block_8x8_kernel<<<bY, threads_per_block>>>(pred_Y, recons_Y, mbs_Y,
                                                 mb_cols_Y, mb_rows_Y, w_Y);
  // Chroma
  mc_block_8x8_kernel<<<bC, threads_per_block>>>(pred_U, recons_U, mbs_U,
                                                 mb_cols_C, mb_rows_C, w_U);
  mc_block_8x8_kernel<<<bC, threads_per_block>>>(pred_V, recons_V, mbs_V,
                                                 mb_cols_C, mb_rows_C, w_V);
  nvtxRangePop();

  nvtxRangePop();
}