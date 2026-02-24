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

  extern __shared__ uint8_t ref_tile[];

  // Warp based indexing
  int lane = threadIdx.x & 31;    // 0..31 in warp
  int warp_id = threadIdx.x >> 5; // warp index inside block (which warp am i
                                  // in, if we have multiple warps per block)
  int warps_per_block =
      blockDim.x >> 5; // How many warps are there in the block

  int num_mbs = mb_cols * mb_rows;
  int mb_linear = blockIdx.x * warps_per_block + warp_id;
  if (mb_linear >= num_mbs)
    return;

  int mb_x = mb_linear % mb_cols;
  int mb_y = mb_linear / mb_cols;
  macroblock *mb = &mbs[mb_linear];

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

  // Calculate bounding box for search area and clamp if necessary
  int left = max(0, mx - range);
  int top = max(0, my - range);
  int right = min(w - 8, mx + range);
  int bottom = min(h - 8, my + range);

  // Shared tile dimensions
  int tile_w = 8 + 2 * range;

  // Pointer offset per warp
  uint8_t *warp_tile = ref_tile + warp_id * tile_w * tile_w;

  // Load ref tile into shared mem
  for (int i = lane; i < tile_w * tile_w; i += 32) {
    int tx = i % tile_w;
    int ty = i / tile_w;

    int gx = left + tx;
    int gy = top + ty;

    warp_tile[i] = ref[gy * w + gx];
  }

  // Sync the warp lanes
  __syncwarp();

  int best_sad = INT_MAX;
  // Write to register memory, then write to managed memory at the end
  int best_mv_x = 0;
  int best_mv_y = 0;

  // Mask, all lanes are participating
  unsigned mask = __activemask();

  for (int y = top; y <= bottom; ++y) {
    for (int x = left; x <= right; ++x) {
      int tile_x = x - left;
      int tile_y = y - top;

      // Read from shared memory instead of RAM
      uint8_t r0 = warp_tile[(tile_y + v0) * tile_w + (tile_x + u0)];
      uint8_t r1 = warp_tile[(tile_y + v1) * tile_w + (tile_x + u1)];

      int sad = __sad((int)r0, o0, 0);
      sad = __sad((int)r1, o1, sad);

      // Reduce with warp shuffle across lanes
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

  if (lane == 0) {
    mb->mv_x = (int8_t)best_mv_x;
    mb->mv_y = (int8_t)best_mv_y;
    mb->use_mv = 1;
  }
}

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

  const int warps_per_block = 4;
  const int tpb = 32 * warps_per_block;
  size_t mbs_Y = (size_t)a.mb_cols_Y * a.mb_rows_Y;
  size_t mbs_C = (size_t)a.mb_cols_C * a.mb_rows_C;
  size_t blocks_Y = (mbs_Y + warps_per_block - 1) / warps_per_block;
  size_t blocks_C = (mbs_C + warps_per_block - 1) / warps_per_block;
  int tile_w_Y = 8 + 2 * a.range_Y;
  size_t shm_Y = warps_per_block * tile_w_Y * tile_w_Y * sizeof(uint8_t);

  int tile_w_C = 8 + 2 * a.range_C;
  size_t shm_C = warps_per_block * tile_w_C * tile_w_C * sizeof(uint8_t);

  nvtxRangePushA("me_mc_inter");
  me_block_8x8_kernel<<<blocks_Y, tpb, shm_Y, stream_y>>>(
      a.orig_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y, a.h_Y,
      a.range_Y);

  me_block_8x8_kernel<<<blocks_C, tpb, shm_C, stream_u>>>(
      a.orig_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U, a.h_U,
      a.range_C);

  me_block_8x8_kernel<<<blocks_C, tpb, shm_C, stream_v>>>(
      a.orig_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V, a.h_V,
      a.range_C);

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