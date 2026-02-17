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

/* Motion estimation for 8x8 block */
/*
  We take in all necessary parameters so we dont dereference directly from cm as
  that will cause segfaults upon launching kernels.
*/
__global__ static void me_block_8x8_kernel(const uint8_t *__restrict__ orig,
                                           const uint8_t *__restrict__ ref,
                                           struct macroblock *__restrict__ mbs,
                                           int mb_cols, int mb_rows, int w,
                                           int h, int range) {

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

  // Mask, all lanes are participating
  unsigned mask = __activemask();
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
  int mb_idx = blockIdx.x; // one block per MB
  if (mb_idx >= mb_cols * mb_rows)
    return;

  int lane = threadIdx.x; // 0..31
  int mb_x = mb_idx % mb_cols;
  int mb_y = mb_idx / mb_cols;

  const macroblock *mb = &mbs[mb_idx];
  if (!mb->use_mv) {
    return;
  }

  int left = mb_x * 8;
  int top = mb_y * 8;

  // lane handles 2 pixels in the 8x8 block
  int p0 = lane;      // 0..31
  int p1 = lane + 32; // 32..63

  int u0 = p0 & 7, v0 = p0 >> 3;
  int u1 = p1 & 7, v1 = p1 >> 3;

  int x0 = left + u0, y0 = top + v0;
  int x1 = left + u1, y1 = top + v1;

  predicted[y0 * w + x0] = ref[(y0 + mb->mv_y) * w + (x0 + mb->mv_x)];
  predicted[y1 * w + x1] = ref[(y1 + mb->mv_y) * w + (x1 + mb->mv_x)];
}

void launch_motion_inter(const motion_inter_args &a, cudaStream_t stream_y,
                         cudaStream_t stream_u, cudaStream_t stream_v) {

  const int warps_per_block = 1;
  const int tpb = 32 * warps_per_block;
  size_t mbs_Y = (size_t)a.mb_cols_Y * a.mb_rows_Y;
  size_t mbs_C = (size_t)a.mb_cols_C * a.mb_rows_C;
  size_t blocks_Y = (mbs_Y + warps_per_block - 1) / warps_per_block;
  size_t blocks_C = (mbs_C + warps_per_block - 1) / warps_per_block;

  nvtxRangePushA("motion_estimation");
  me_block_8x8_kernel<<<blocks_Y, tpb, 0, stream_y>>>(
      a.orig_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y, a.h_Y,
      a.range_Y);

  me_block_8x8_kernel<<<blocks_C, tpb, 0, stream_u>>>(
      a.orig_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U, a.h_U,
      a.range_C);

  me_block_8x8_kernel<<<blocks_C, tpb, 0, stream_v>>>(
      a.orig_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V, a.h_V,
      a.range_C);

  nvtxRangePop();

  nvtxRangePushA("motion_compensation");
  mc_block_8x8_kernel<<<blocks_Y, tpb, 0, stream_y>>>(
      a.pred_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y);

  mc_block_8x8_kernel<<<blocks_C, tpb, 0, stream_u>>>(
      a.pred_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U);

  mc_block_8x8_kernel<<<blocks_C, tpb, 0, stream_v>>>(
      a.pred_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V);
  nvtxRangePop();
}

motion_inter_args create_motion_inter_args(struct c63_common *cm) {
  motion_inter_args a{};

  a.is_keyframe = cm->curframe->keyframe;

  // ME inputs
  a.orig_Y = cm->curframe->orig->Y;
  a.orig_U = cm->curframe->orig->U;
  a.orig_V = cm->curframe->orig->V;

  // Only read reference frame when inter-frame
  if (!a.is_keyframe && cm->refframe) {
    a.recons_Y = cm->refframe->recons->Y;
    a.recons_U = cm->refframe->recons->U;
    a.recons_V = cm->refframe->recons->V;
  } else {
    a.recons_Y = nullptr;
    a.recons_U = nullptr;
    a.recons_V = nullptr;
  }

  // MV outputs / MC inputs
  a.mbs_Y = cm->curframe->mbs[Y_COMPONENT];
  a.mbs_U = cm->curframe->mbs[U_COMPONENT];
  a.mbs_V = cm->curframe->mbs[V_COMPONENT];

  // MC outputs
  a.pred_Y = cm->curframe->predicted->Y;
  a.pred_U = cm->curframe->predicted->U;
  a.pred_V = cm->curframe->predicted->V;

  // Geometry
  a.mb_cols_Y = cm->mb_cols;
  a.mb_rows_Y = cm->mb_rows;
  a.mb_cols_C = cm->mb_cols / 2;
  a.mb_rows_C = cm->mb_rows / 2;

  a.w_Y = cm->padw[Y_COMPONENT];
  a.h_Y = cm->padh[Y_COMPONENT];
  a.w_U = cm->padw[U_COMPONENT];
  a.h_U = cm->padh[U_COMPONENT];
  a.w_V = cm->padw[V_COMPONENT];
  a.h_V = cm->padh[V_COMPONENT];

  a.range_Y = cm->me_search_range;
  a.range_C = cm->me_search_range / 2;

  return a;
}