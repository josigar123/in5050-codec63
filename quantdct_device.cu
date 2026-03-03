#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "quantdct_device.h"

#define ISQRT2                                                                 \
  0.70710678118654f // 1/sqrt(2), used to scale the DC (zero-frequency) terms in
                    // the DCT

// These tables live in constant memory, which is fast read-only memory cached
// and shared across all threads. Ideal for lookup tables that every thread
// reads.
__constant__ uint8_t c_zigzag_U[64]; // Column coordinates for each DCT
                                     // coefficient in zigzag scan order
__constant__ uint8_t c_zigzag_V[64]; // Row coordinates for each DCT coefficient
                                     // in zigzag scan order
__constant__ float c_dctlookup[8][8]; // Pre-computed cosine values used in the
                                      // DCT/IDCT formula

__constant__ uint8_t c_quant_Y[64]; // Quantization table for Y (luma)
__constant__ uint8_t c_quant_U[64]; // Quantization table for U (chroma)
__constant__ uint8_t c_quant_V[64]; // Quantization table for V (chroma)

// Returns the right quantization table for the given color component (Y, U, or
// V). Using a template means the choice is made at compile time, not at
// runtime.
template <int C> __device__ __forceinline__ const uint8_t *get_quant_tbl() {
  return (C == Y_COMPONENT)   ? c_quant_Y
         : (C == U_COMPONENT) ? c_quant_U
                              : c_quant_V;
}

// Uploads all lookup tables and quantization tables to GPU constant memory.
// Must be called once before any kernel is launched.
void init_quantdct_constants(const c63_common *cm) {
  cudaMemcpyToSymbol(c_zigzag_U, zigzag_U, sizeof(zigzag_U));
  cudaMemcpyToSymbol(c_zigzag_V, zigzag_V, sizeof(zigzag_V));
  cudaMemcpyToSymbol(c_dctlookup, dctlookup, sizeof(dctlookup));

  cudaMemcpyToSymbol(c_quant_Y, cm->quanttbl[Y_COMPONENT],
                     sizeof(cm->quanttbl[Y_COMPONENT]));
  cudaMemcpyToSymbol(c_quant_U, cm->quanttbl[U_COMPONENT],
                     sizeof(cm->quanttbl[U_COMPONENT]));
  cudaMemcpyToSymbol(c_quant_V, cm->quanttbl[V_COMPONENT],
                     sizeof(cm->quanttbl[V_COMPONENT]));
}

// Reverses the DCT+Quantize step: dequantizes the coefficients and applies IDCT
// to reconstruct pixels. One warp per 8x8 block; each lane handles two
// coefficients (z0 and z1 = z0+32).
template <int C>
__global__ void dequantize_idct_kernel(const int16_t *__restrict__ in_data,
                                       const uint8_t *__restrict__ prediction,
                                       uint32_t width, uint32_t height,
                                       uint8_t *__restrict__ out_data) {

  // Shared memory stores the dequantized 8x8 coefficient block for this warp,
  // so all lanes can access each other's values during the IDCT summation.
  extern __shared__ float shared[];

  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;

  int blocks_x = (int)width / 8;
  int blocks_y = (int)height / 8;
  int total_blocks = blocks_x * blocks_y;

  int block_linear = blockIdx.x * warps_per_block + warp_id;
  if (block_linear >= total_blocks)
    return;

  int bx = block_linear % blocks_x;
  int by = block_linear / blocks_x;

  int px = bx * 8; // Top-left x pixel of this block
  int py = by * 8; // Top-left y pixel of this block
  int pixel_base = py * (int)width + px;
  int coeff_base = by * ((int)width * 8) +
                   bx * 64; // Offset into the coefficient array for this block

  float *warp_coeff =
      shared +
      warp_id * 64; // Each warp gets its own 64-float slice of shared memory
  const uint8_t *quant_tbl = get_quant_tbl<C>();

  // Each lane handles two zigzag positions: z0 and z1, covering all 64
  // coefficients across the 32 lanes
  int z0 = lane;
  int z1 = lane + 32;

  // Dequantize coefficient z0 and write to shared memory in 2D frequency layout
  // (v*8 + u)
  uint8_t u0 = c_zigzag_U[z0];
  uint8_t v0 = c_zigzag_V[z0];
  float dct0 = (float)in_data[coeff_base + z0];
  float deq0 = (float)round((dct0 * quant_tbl[z0]) / 4.0); // Undo quantization
  float a10 = !u0 ? ISQRT2
                  : 1.0f; // Apply DC scaling if this is the zero-frequency term
  float a20 = !v0 ? ISQRT2 : 1.0f;
  warp_coeff[v0 * 8 + u0] = deq0 * a10 * a20;

  // Same for coefficient z1
  uint8_t u1 = c_zigzag_U[z1];
  uint8_t v1 = c_zigzag_V[z1];
  float dct1 = (float)in_data[coeff_base + z1];
  float deq1 = (float)round((dct1 * quant_tbl[z1]) / 4.0);
  float a11 = !u1 ? ISQRT2 : 1.0f;
  float a21 = !v1 ? ISQRT2 : 1.0f;
  warp_coeff[v1 * 8 + u1] = deq1 * a11 * a21;

  __syncwarp(); // Wait for all lanes to finish writing before reading in IDCT

  // Re-use lane as pixel index. Each lane reconstructs two output pixels.
  int p0 = lane;
  int p1 = lane + 32;

  int out_u0 = p0 & 7;  // Column of pixel p0 in the 8x8 block
  int out_v0 = p0 >> 3; // Row of pixel p0
  float idct0 = 0.0f;

  int out_u1 = p1 & 7;
  int out_v1 = p1 >> 3;
  float idct1 = 0.0f;

  // IDCT: sum all frequency contributions to reconstruct each pixel
#pragma unroll
  for (int y = 0; y < 8; ++y) {
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      idct0 += warp_coeff[y * 8 + x] * c_dctlookup[out_u0][x] *
               c_dctlookup[out_v0][y];

      idct1 += warp_coeff[y * 8 + x] * c_dctlookup[out_u1][x] *
               c_dctlookup[out_v1][y];
    }
  }

  // Add prediction back (residual + prediction = reconstructed pixel), clamp to
  // [0, 255]
  int pred_idx0 = pixel_base + out_v0 * (int)width + out_u0;
  int16_t tmp0 = (int16_t)idct0 + (int16_t)prediction[pred_idx0];
  if (tmp0 < 0)
    tmp0 = 0;
  else if (tmp0 > 255)
    tmp0 = 255;
  out_data[pred_idx0] = (uint8_t)tmp0;

  int pred_idx1 = pixel_base + out_v1 * (int)width + out_u1;
  int16_t tmp1 = (int16_t)idct1 + (int16_t)prediction[pred_idx1];
  if (tmp1 < 0)
    tmp1 = 0;
  else if (tmp1 > 255)
    tmp1 = 255;
  out_data[pred_idx1] = (uint8_t)tmp1;
}

// Computes residuals (current - predicted), applies DCT to convert to frequency
// domain, then quantizes the coefficients (divides by quant table, killing
// small high-frequency values). Output is written in zigzag order, ready for
// entropy coding. One warp per 8x8 block; each lane handles two pixels (p0 and
// p1 = p0+32).
template <int C>
__global__ void dct_quantize_kernel(const uint8_t *__restrict__ in_data,
                                    uint8_t *__restrict__ prediction,
                                    uint32_t width, uint32_t height,
                                    int16_t *__restrict__ out_data) {
  // Shared memory stores the residual pixel block for this warp,
  // so all lanes can read any pixel during the DCT summation.
  extern __shared__ float shared[];

  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;

  int blocks_x = (int)width / 8;
  int blocks_y = (int)height / 8;
  int total_blocks = blocks_x * blocks_y;

  int block_linear = blockIdx.x * warps_per_block + warp_id;
  if (block_linear >= total_blocks)
    return;

  int bx = block_linear % blocks_x;
  int by = block_linear / blocks_x;

  int px = bx * 8; // Top-left x pixel of this block
  int py = by * 8; // Top-left y pixel of this block
  int pixel_base = py * (int)width + px;
  int coeff_base = by * ((int)width * 8) +
                   bx * 64; // Offset into the coefficient array for this block

  float *warp_block =
      shared +
      warp_id * 64; // Each warp gets its own 64-float slice of shared memory
  const uint8_t *quant_tbl = get_quant_tbl<C>();

  // Each lane handles two pixels: p0 and p1, covering all 64 pixels across the
  // 32 lanes
  int p0 = lane;
  int p1 = lane + 32;

  // Compute residual (current pixel - predicted pixel) and write to shared
  // memory
  int u0 = p0 & 7;
  int v0 = p0 >> 3;
  int in_idx0 = pixel_base + v0 * (int)width + u0;
  warp_block[p0] = (float)((int16_t)in_data[in_idx0] -
                           (int16_t)prediction[in_idx0]); // residual for p0

  int u1 = p1 & 7;
  int v1 = p1 >> 3;
  int in_idx1 = pixel_base + v1 * (int)width + u1;
  warp_block[p1] = (float)((int16_t)in_data[in_idx1] -
                           (int16_t)prediction[in_idx1]); // residual for p1

  __syncwarp(); // Wait for all lanes to finish writing residuals before DCT
                // reads them

  // Each lane computes two DCT coefficients in zigzag order
  int z0 = lane;
  int z1 = lane + 32;

  uint8_t coeff_u0 = c_zigzag_U[z0]; // Frequency column for coefficient z0
  uint8_t coeff_v0 = c_zigzag_V[z0]; // Frequency row for coefficient z0
  float dct0 = 0.0f;

  uint8_t coeff_u1 = c_zigzag_U[z1];
  uint8_t coeff_v1 = c_zigzag_V[z1];
  float dct1 = 0.0f;

  // DCT: sum all pixel contributions to compute each frequency coefficient
#pragma unroll
  for (int y = 0; y < 8; ++y) {
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      dct0 += warp_block[y * 8 + x] * c_dctlookup[x][coeff_u0] *
              c_dctlookup[y][coeff_v0];

      dct1 += warp_block[y * 8 + x] * c_dctlookup[x][coeff_u1] *
              c_dctlookup[y][coeff_v1];
    }
  }

  // Apply DC scaling and quantize, then write to global memory
  float a10 = !coeff_u0 ? ISQRT2
                        : 1.0f; // Scale if this is the zero-frequency (DC) term
  float a20 = !coeff_v0 ? ISQRT2 : 1.0f;
  float scaled0 = dct0 * a10 * a20;
  out_data[coeff_base + z0] =
      (int16_t)round((scaled0 / 4.0) / quant_tbl[z0]); // Quantize

  float a11 = !coeff_u1 ? ISQRT2 : 1.0f;
  float a21 = !coeff_v1 ? ISQRT2 : 1.0f;
  float scaled1 = dct1 * a11 * a21;
  out_data[coeff_base + z1] =
      (int16_t)round((scaled1 / 4.0) / quant_tbl[z1]); // Quantize
}

/* Launches the DCT/Quant/DeQuant/IDCT pipeline for inter-coded frames
 * Each color component (Y, U, V) is processed in its own CUDA stream
 * -> concurrent execution and better GPU utilization.
 */
void launch_quantdct_inter(const quant_inter_args &a, cudaStream_t stream_y,
                           cudaStream_t stream_u, cudaStream_t stream_v) {
  const int warps_per_block_dct = 4;
  const int warps_per_block_idct = 1;
  const int tpb_dct = 32 * warps_per_block_dct;
  const int tpb_idct = 32 * warps_per_block_idct;

  size_t blocksY_8x8 = (size_t)(a.wY / 8) * (size_t)(a.hY / 8);
  size_t blocksU_8x8 = (size_t)(a.wU / 8) * (size_t)(a.hU / 8);
  size_t blocksV_8x8 = (size_t)(a.wV / 8) * (size_t)(a.hV / 8);

  size_t gridY_dct =
      (blocksY_8x8 + warps_per_block_dct - 1) / warps_per_block_dct;
  size_t gridU_dct =
      (blocksU_8x8 + warps_per_block_dct - 1) / warps_per_block_dct;
  size_t gridV_dct =
      (blocksV_8x8 + warps_per_block_dct - 1) / warps_per_block_dct;

  size_t gridY_idct =
      (blocksY_8x8 + warps_per_block_idct - 1) / warps_per_block_idct;
  size_t gridU_idct =
      (blocksU_8x8 + warps_per_block_idct - 1) / warps_per_block_idct;
  size_t gridV_idct =
      (blocksV_8x8 + warps_per_block_idct - 1) / warps_per_block_idct;

  size_t shm_bytes_dct = (size_t)warps_per_block_dct * 64 * sizeof(float);
  size_t shm_bytes_idct = (size_t)warps_per_block_idct * 64 * sizeof(float);

  dct_quantize_kernel<Y_COMPONENT>
      <<<gridY_dct, tpb_dct, shm_bytes_dct, stream_y>>>(a.inY, a.predY, a.wY,
                                                        a.hY, a.resY);

  dct_quantize_kernel<U_COMPONENT>
      <<<gridU_dct, tpb_dct, shm_bytes_dct, stream_u>>>(a.inU, a.predU, a.wU,
                                                        a.hU, a.resU);

  dct_quantize_kernel<V_COMPONENT>
      <<<gridV_dct, tpb_dct, shm_bytes_dct, stream_v>>>(a.inV, a.predV, a.wV,
                                                        a.hV, a.resV);

  dequantize_idct_kernel<Y_COMPONENT>
      <<<gridY_idct, tpb_idct, shm_bytes_idct, stream_y>>>(a.resY, a.predY,
                                                           a.wY, a.hY, a.recY);

  dequantize_idct_kernel<U_COMPONENT>
      <<<gridU_idct, tpb_idct, shm_bytes_idct, stream_u>>>(a.resU, a.predU,
                                                           a.wU, a.hU, a.recU);

  dequantize_idct_kernel<V_COMPONENT>
      <<<gridV_idct, tpb_idct, shm_bytes_idct, stream_v>>>(a.resV, a.predV,
                                                           a.wV, a.hV, a.recV);
}

// Create a snapshot of the necessary arguments for the DCT/Quant/DeQuant/IDCT
// pipeline
quant_inter_args create_quant_inter_args(struct c63_common *cm, yuv_t *image) {
  quant_inter_args a{};

  a.inY = image->Y;
  a.inU = image->U;
  a.inV = image->V;

  a.predY = cm->curframe->predicted->Y;
  a.predU = cm->curframe->predicted->U;
  a.predV = cm->curframe->predicted->V;

  a.resY = cm->curframe->residuals->Ydct;
  a.resU = cm->curframe->residuals->Udct;
  a.resV = cm->curframe->residuals->Vdct;

  a.recY = cm->curframe->recons->Y;
  a.recU = cm->curframe->recons->U;
  a.recV = cm->curframe->recons->V;

  a.wY = cm->padw[Y_COMPONENT];
  a.hY = cm->padh[Y_COMPONENT];
  a.wU = cm->padw[U_COMPONENT];
  a.hU = cm->padh[U_COMPONENT];
  a.wV = cm->padw[V_COMPONENT];
  a.hV = cm->padh[V_COMPONENT];

  return a;
}