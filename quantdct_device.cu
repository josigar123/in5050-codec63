#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <nvtx3/nvToolsExt.h>

#include "quantdct_device.h"

#define ISQRT2 0.70710678118654f

// Forward declarations for tables in constant memory
__constant__ uint8_t c_zigzag_U[64];
__constant__ uint8_t c_zigzag_V[64];
__constant__ float c_dctlookup[8][8];

__constant__ uint8_t c_quant_Y[64];
__constant__ uint8_t c_quant_U[64];
__constant__ uint8_t c_quant_V[64];

// We need a template to get the correct quant table for kernel runs
template <int C> __device__ __forceinline__ const uint8_t *get_quant_tbl() {
  return (C == Y_COMPONENT)   ? c_quant_Y
         : (C == U_COMPONENT) ? c_quant_U
                              : c_quant_V;
}

// Called once at startup from init_c63_enc
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

__device__ __forceinline__ static void dct_2d_device(const float *in,
                                                     float *out) {

  for (int v = 0; v < 8; v++) {
    for (int u = 0; u < 8; u++) {
      /* Compute the DCT */
      float dct = 0;
      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          dct += in[y * 8 + x] * c_dctlookup[x][u] * c_dctlookup[y][v];
        }
      }

      out[v * 8 + u] = dct;
    }
  }
}

// Same as dct_2d, but reverse lookup, same optimization could work
__device__ __forceinline__ static void idct_2d_device(const float *in,
                                                      float *out) {
  // Loop through all elements of the block
  for (int v = 0; v < 8; v++) {
    for (int u = 0; u < 8; u++) {
      /* Compute the iDCT */
      float dct = 0;
      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          dct += in[y * 8 + x] * c_dctlookup[u][x] * c_dctlookup[v][y];
        }
      }

      out[v * 8 + u] = dct;
    }
  }
}

__device__ __forceinline__ static void scale_block_device(float *in_data,
                                                          float *out_data) {
  int u, v;

  for (v = 0; v < 8; ++v) {
    for (u = 0; u < 8; ++u) {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
    }
  }
}

__device__ __forceinline__ static void
quantize_block_device(float *in_data, float *out_data,
                      const uint8_t *quant_tbl) {
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = c_zigzag_U[zigzag];
    uint8_t v = c_zigzag_V[zigzag];

    float dct = in_data[v * 8 + u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float)round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

__device__ __forceinline__ static void
dequantize_block_device(float *in_data, float *out_data,
                        const uint8_t *quant_tbl) {
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = c_zigzag_U[zigzag];
    uint8_t v = c_zigzag_V[zigzag];

    float dct = in_data[zigzag];

    out_data[v * 8 + u] = (float)round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

__device__ __forceinline__ static void
dct_quant_block_8x8_device(int16_t *in_data, int16_t *out_data,
                           const uint8_t *quant_tbl) {

  float mb[64], mb2[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    mb[i] = (float)in_data[i];
  }

  dct_2d_device(mb, mb2);

  scale_block_device(mb2, mb);

  quantize_block_device(mb, mb2, quant_tbl);
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    out_data[i] = (int16_t)mb2[i];
  }
}

__device__ __forceinline__ static void
dequant_idct_block_8x8_device(const int16_t *in_data, int16_t *out_data,
                              const uint8_t *quant_tbl) {
  float mb[64], mb2[64];

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    mb[i] = (float)in_data[i];
  }

  dequantize_block_device(mb, mb2, quant_tbl);
  scale_block_device(mb2, mb);
  idct_2d_device(mb, mb2);
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    out_data[i] = (int16_t)mb2[i];
  }
}

template <int C>
__global__ void dequantize_idct_kernel(const int16_t *__restrict__ in_data,
                                       const uint8_t *__restrict__ prediction,
                                       uint32_t width, uint32_t height,
                                       uint8_t *__restrict__ out_data) {

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

  int px = bx * 8;
  int py = by * 8;
  int pixel_base = py * (int)width + px;
  int coeff_base = by * ((int)width * 8) + bx * 64;

  float *warp_coeff = shared + warp_id * 64;
  const uint8_t *quant_tbl = get_quant_tbl<C>();

  int z0 = lane;
  int z1 = lane + 32;

  uint8_t u0 = c_zigzag_U[z0];
  uint8_t v0 = c_zigzag_V[z0];
  float dct0 = (float)in_data[coeff_base + z0];
  float deq0 = (float)round((dct0 * quant_tbl[z0]) / 4.0);
  float a10 = !u0 ? ISQRT2 : 1.0f;
  float a20 = !v0 ? ISQRT2 : 1.0f;
  warp_coeff[v0 * 8 + u0] = deq0 * a10 * a20;

  uint8_t u1 = c_zigzag_U[z1];
  uint8_t v1 = c_zigzag_V[z1];
  float dct1 = (float)in_data[coeff_base + z1];
  float deq1 = (float)round((dct1 * quant_tbl[z1]) / 4.0);
  float a11 = !u1 ? ISQRT2 : 1.0f;
  float a21 = !v1 ? ISQRT2 : 1.0f;
  warp_coeff[v1 * 8 + u1] = deq1 * a11 * a21;

  __syncwarp();

  int p0 = lane;
  int p1 = lane + 32;

  int out_u0 = p0 & 7;
  int out_v0 = p0 >> 3;
  float idct0 = 0.0f;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      idct0 += warp_coeff[y * 8 + x] * c_dctlookup[out_u0][x] *
               c_dctlookup[out_v0][y];
    }
  }
  int pred_idx0 = pixel_base + out_v0 * (int)width + out_u0;
  int16_t tmp0 = (int16_t)idct0 + (int16_t)prediction[pred_idx0];
  if (tmp0 < 0)
    tmp0 = 0;
  else if (tmp0 > 255)
    tmp0 = 255;
  out_data[pred_idx0] = (uint8_t)tmp0;

  int out_u1 = p1 & 7;
  int out_v1 = p1 >> 3;
  float idct1 = 0.0f;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      idct1 += warp_coeff[y * 8 + x] * c_dctlookup[out_u1][x] *
               c_dctlookup[out_v1][y];
    }
  }
  int pred_idx1 = pixel_base + out_v1 * (int)width + out_u1;
  int16_t tmp1 = (int16_t)idct1 + (int16_t)prediction[pred_idx1];
  if (tmp1 < 0)
    tmp1 = 0;
  else if (tmp1 > 255)
    tmp1 = 255;
  out_data[pred_idx1] = (uint8_t)tmp1;
}

template <int C>
__global__ void dct_quantize_kernel(const uint8_t *__restrict__ in_data,
                                    uint8_t *__restrict__ prediction,
                                    uint32_t width, uint32_t height,
                                    int16_t *__restrict__ out_data) {
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

  int px = bx * 8;
  int py = by * 8;
  int pixel_base = py * (int)width + px;
  int coeff_base = by * ((int)width * 8) + bx * 64;

  float *warp_block = shared + warp_id * 64;
  const uint8_t *quant_tbl = get_quant_tbl<C>();

  int p0 = lane;
  int p1 = lane + 32;

  int u0 = p0 & 7;
  int v0 = p0 >> 3;
  int in_idx0 = pixel_base + v0 * (int)width + u0;
  warp_block[p0] =
      (float)((int16_t)in_data[in_idx0] - (int16_t)prediction[in_idx0]);

  int u1 = p1 & 7;
  int v1 = p1 >> 3;
  int in_idx1 = pixel_base + v1 * (int)width + u1;
  warp_block[p1] =
      (float)((int16_t)in_data[in_idx1] - (int16_t)prediction[in_idx1]);

  __syncwarp();

  int z0 = lane;
  int z1 = lane + 32;

  uint8_t coeff_u0 = c_zigzag_U[z0];
  uint8_t coeff_v0 = c_zigzag_V[z0];
  float dct0 = 0.0f;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      dct0 += warp_block[y * 8 + x] * c_dctlookup[x][coeff_u0] *
              c_dctlookup[y][coeff_v0];
    }
  }
  float a10 = !coeff_u0 ? ISQRT2 : 1.0f;
  float a20 = !coeff_v0 ? ISQRT2 : 1.0f;
  float scaled0 = dct0 * a10 * a20;
  out_data[coeff_base + z0] = (int16_t)round((scaled0 / 4.0) / quant_tbl[z0]);

  uint8_t coeff_u1 = c_zigzag_U[z1];
  uint8_t coeff_v1 = c_zigzag_V[z1];
  float dct1 = 0.0f;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      dct1 += warp_block[y * 8 + x] * c_dctlookup[x][coeff_u1] *
              c_dctlookup[y][coeff_v1];
    }
  }
  float a11 = !coeff_u1 ? ISQRT2 : 1.0f;
  float a21 = !coeff_v1 ? ISQRT2 : 1.0f;
  float scaled1 = dct1 * a11 * a21;
  out_data[coeff_base + z1] = (int16_t)round((scaled1 / 4.0) / quant_tbl[z1]);
}

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

  nvtxRangePushA("dct_idct_inter");
  nvtxRangePushA("dct_quantize");
  dct_quantize_kernel<Y_COMPONENT>
      <<<gridY_dct, tpb_dct, shm_bytes_dct, stream_y>>>(a.inY, a.predY, a.wY,
                                                        a.hY, a.resY);

  dct_quantize_kernel<U_COMPONENT>
      <<<gridU_dct, tpb_dct, shm_bytes_dct, stream_u>>>(a.inU, a.predU, a.wU,
                                                        a.hU, a.resU);

  dct_quantize_kernel<V_COMPONENT>
      <<<gridV_dct, tpb_dct, shm_bytes_dct, stream_v>>>(a.inV, a.predV, a.wV,
                                                        a.hV, a.resV);

  nvtxRangePop();
  nvtxRangePushA("dequantize_idct");
  dequantize_idct_kernel<Y_COMPONENT>
      <<<gridY_idct, tpb_idct, shm_bytes_idct, stream_y>>>(a.resY, a.predY,
                                                           a.wY, a.hY, a.recY);

  dequantize_idct_kernel<U_COMPONENT>
      <<<gridU_idct, tpb_idct, shm_bytes_idct, stream_u>>>(a.resU, a.predU,
                                                           a.wU, a.hU, a.recU);

  dequantize_idct_kernel<V_COMPONENT>
      <<<gridV_idct, tpb_idct, shm_bytes_idct, stream_v>>>(a.resV, a.predV,
                                                           a.wV, a.hV, a.recV);
  nvtxRangePop();

  nvtxRangePop();
}

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