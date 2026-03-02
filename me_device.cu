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

  // A tile in shared memory holding the search window, allocated by the kernel
  // launch params
  extern __shared__ uint8_t ref_tile[];

  // Warp based indexing
  int lane =
      threadIdx.x & 31; // What lane am I in this warp? Same as threadIdx.x % 32
  int warp_id = threadIdx.x >>
                5; // What warp am I in this block? Same as threadIdx.x / 32
  int warps_per_block =
      blockDim.x >>
      5; // How many warps are there in the block? Same as blockDim.x / 32

  int num_mbs = mb_cols * mb_rows; // Total number of macroblocks in the frame
  int mb_linear = blockIdx.x * warps_per_block +
                  warp_id; // Linear index of the macroblock, standard formula
                           // for indexing into a 1D array

  // If the linear index is greater than the total number of macroblocks, return
  if (mb_linear >= num_mbs)
    return;

  int mb_x = mb_linear % mb_cols;   // Column index of the macroblock
  int mb_y = mb_linear / mb_cols;   // Row index of the macroblock
  macroblock *mb = &mbs[mb_linear]; // Pointer to the macroblock

  int mx = mb_x * 8; // Starting x-coordinate of the macroblock in pixels
  int my = mb_y * 8; // Starting y-coordinate of the macroblock in pixels

  // Two pixels in 8x8 block
  int p0 = lane;      // What pixel am I in this lane? Same as lane
  int p1 = lane + 32; // What pixel am I in this lane? Same as lane + 32

  int u0 = p0 & 7, v0 = p0 >> 3; // What column and row am I in this pixel? Same
                                 // as p0 % 8 and p0 / 8
  int u1 = p1 & 7, v1 = p1 >> 3; // What column and row am I in this pixel? Same
                                 // as p1 % 8 and p1 / 8

  // Cache orig pixels for current lane
  // each lane does two reads from global memory for caching
  int o0 =
      (int)orig[(my + v0) * w + (mx + u0)]; // Pixel 0 in the original frame
  int o1 =
      (int)orig[(my + v1) * w + (mx + u1)]; // Pixel 1 in the original frame

  // Calculate bounding box for search area and clamp if necessary
  int left = max(0, mx - range);
  int top = max(0, my - range);
  int right = min(w - 8, mx + range);
  int bottom = min(h - 8, my + range);

  // Shared tile dimensions
  int tile_w = 8 + 2 * range; // Width of the search window in pixels

  // Pointer offset per warp
  uint8_t *warp_tile =
      ref_tile +
      warp_id * tile_w *
          tile_w; // Pointer to the start of the search window for this warp

  // Load ref tile into shared mem where warp_tile is the pointer base for where
  // to load the ref pixels for this warp
  for (int i = lane; i < tile_w * tile_w; i += 32) {
    int tx = i % tile_w; // What column am I in this pixel? Same as i % tile_w
    int ty = i / tile_w; // What row am I in this pixel? Same as i / tile_w

    int gx = left + tx; // What x-coordinate am I in this pixel?
    int gy = top + ty;  // What y-coordinate am I in this pixel?

    warp_tile[i] =
        ref[gy * w +
            gx]; // Load the reference pixel into shared memory, row major order
  }

  // Sync the warp lanes, ensure all lanes have loaded the search window before
  // continuing
  __syncwarp();

  int best_sad = INT_MAX;
  // Write to register memory, then write to managed memory at the end
  int best_mv_x = 0;
  int best_mv_y = 0;

  // Mask, all lanes are participating
  unsigned mask =
      __activemask(); // A dynamic mask for all active lanes in the warp

// Iterate over the search window
#pragma unroll
  for (int y = top; y <= bottom; ++y) {
#pragma unroll
    for (int x = left; x <= right; ++x) {
      int tile_x = x - left; // What column?
      int tile_y = y - top;  // What row?

      // Read from shared memory instead of RAM, reading from the warp tile
      // where we loaded the ref for this wapr
      uint8_t r0 = warp_tile[(tile_y + v0) * tile_w + (tile_x + u0)];
      uint8_t r1 = warp_tile[(tile_y + v1) * tile_w + (tile_x + u1)];

      // Perform SAD with intrinsic
      int sad = __sad((int)r0, o0, 0);
      sad = __sad((int)r1, o1, sad);

      // Reduce with warp shuffle across lanes so that lane 0 holds the correct
      // SAD value Reduce the offsets for each shuffle so we bubble the value
      // into lane 0
      sad += __shfl_down_sync(mask, sad, 16);
      sad += __shfl_down_sync(mask, sad, 8);
      sad += __shfl_down_sync(mask, sad, 4);
      sad += __shfl_down_sync(mask, sad, 2);
      sad += __shfl_down_sync(mask, sad, 1);

      // Only lane zero performs the comparison and updates the best candidate
      // so far since it holds the correct SAD value out of all the lanes from
      // the reduction above.
      if (lane == 0 && sad < best_sad) {
        best_sad = sad;
        best_mv_x = x - mx;
        best_mv_y = y - my;
      }
    }
  }

  // Lane zero writes the best candidate vector to global memory at the end
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
  int lane =
      threadIdx.x & 31; // What lane am I in this warp? Same as threadIdx.x % 32
  int warp_id = threadIdx.x >>
                5; // What warp am I in this block? Same as threadIdx.x / 32
  int warps_per_block =
      blockDim.x >>
      5; // How many warps are there in the block? Same as blockDim.x / 32

  int num_mbs = mb_cols * mb_rows; // Total number of macroblocks in the frame
  int mb_linear = blockIdx.x * warps_per_block +
                  warp_id; // Linear index of the macroblock, standard formula
                           // for indexing into a 1D array

  // If the linear index is greater than the total number of macroblocks, return
  if (mb_linear >= num_mbs)
    return;

  int mb_x = mb_linear % mb_cols; // Column index of the macroblock
  int mb_y = mb_linear / mb_cols; // Row index of the macroblock

  // Return if we are not using a motion vector
  const macroblock *mb = &mbs[mb_linear]; // Pointer to the macroblock
  if (!mb->use_mv) {
    return;
  }

  int left = mb_x * 8; // Starting x-coordinate of the macroblock in pixels
  int top = mb_y * 8;  // Starting y-coordinate of the macroblock in pixels

  // lane handles 2 pixels in the 8x8 block
  int p0 = lane;      // What pixel am I in this lane?
  int p1 = lane + 32; // What pixel am I in this lane?

  int u0 = p0 & 7, v0 = p0 >> 3; // What column and row am I in this pixel? Same
                                 // as p0 % 8 and p0 / 8
  int u1 = p1 & 7, v1 = p1 >> 3; // What column and row am I in this pixel? Same
                                 // as p1 % 8 and p1 / 8

  int x0 = left + u0,
      y0 = top + v0; // What x-coordinate and y-coordinate am I in this pixel?
                     // Same as left + u0 and top + v0
  int x1 = left + u1,
      y1 = top + v1; // What x-coordinate and y-coordinate am I in this pixel?
                     // Same as left + u1 and top + v1

  // Write the predicted pixels from the reference frame to the predicted frame
  // (global memory)
  predicted[y0 * w + x0] = ref[(y0 + mb->mv_y) * w + (x0 + mb->mv_x)];
  predicted[y1 * w + x1] = ref[(y1 + mb->mv_y) * w + (x1 + mb->mv_x)];
}

void launch_motion_inter(const motion_inter_args &a, cudaStream_t stream_y,
                         cudaStream_t stream_u, cudaStream_t stream_v) {

  const int warps_per_block = 1;
  const int tpb = 32 * warps_per_block;

  // Total number of macroblocks in the frame
  size_t mbs_Y = (size_t)a.mb_cols_Y * a.mb_rows_Y;
  size_t mbs_C = (size_t)a.mb_cols_C * a.mb_rows_C;

  // How many blocks we want for each plane
  size_t blocks_Y = (mbs_Y + warps_per_block - 1) / warps_per_block;
  size_t blocks_C = (mbs_C + warps_per_block - 1) / warps_per_block;

  // Allocate shared memory for the search window for each plane
  int tile_w_Y = 8 + 2 * a.range_Y;
  size_t shm_Y = warps_per_block * tile_w_Y * tile_w_Y * sizeof(uint8_t);

  int tile_w_C = 8 + 2 * a.range_C;
  size_t shm_C = warps_per_block * tile_w_C * tile_w_C * sizeof(uint8_t);

  // ME
  me_block_8x8_kernel<<<blocks_Y, tpb, shm_Y, stream_y>>>(
      a.orig_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y, a.h_Y,
      a.range_Y);

  me_block_8x8_kernel<<<blocks_C, tpb, shm_C, stream_u>>>(
      a.orig_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U, a.h_U,
      a.range_C);

  me_block_8x8_kernel<<<blocks_C, tpb, shm_C, stream_v>>>(
      a.orig_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V, a.h_V,
      a.range_C);

  // MC
  mc_block_8x8_kernel<<<blocks_Y, tpb, 0, stream_y>>>(
      a.pred_Y, a.recons_Y, a.mbs_Y, a.mb_cols_Y, a.mb_rows_Y, a.w_Y);

  mc_block_8x8_kernel<<<blocks_C, tpb, 0, stream_u>>>(
      a.pred_U, a.recons_U, a.mbs_U, a.mb_cols_C, a.mb_rows_C, a.w_U);

  mc_block_8x8_kernel<<<blocks_C, tpb, 0, stream_v>>>(
      a.pred_V, a.recons_V, a.mbs_V, a.mb_cols_C, a.mb_rows_C, a.w_V);
}

// Create a snapshot of the necessary arguments for the motion estimation and
// compensation
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