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

// Called once at startup from init_c63_enc
void init_quantdct_constants(void) {
  cudaMemcpyToSymbol(c_zigzag_U, zigzag_U, sizeof(zigzag_U));
  cudaMemcpyToSymbol(c_zigzag_V, zigzag_V, sizeof(zigzag_V));
  cudaMemcpyToSymbol(c_dctlookup, dctlookup, sizeof(dctlookup));
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
// DANGER: Also used by the decoder
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
quantize_block_device(float *in_data, float *out_data, uint8_t *quant_tbl) {
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
dequantize_block_device(float *in_data, float *out_data, uint8_t *quant_tbl) {
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
                           uint8_t *quant_tbl) {

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
dequant_idct_block_8x8_device(int16_t *in_data, int16_t *out_data,
                              uint8_t *quant_tbl) {
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

__global__ void dequantize_idct_kernel(int16_t *__restrict__ in_data,
                                       uint8_t *__restrict__ prediction,
                                       uint32_t width, uint32_t height,
                                       uint8_t *__restrict__ out_data,
                                       uint8_t *__restrict__ quantization) {

  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;

  int blocks_x = width / 8;
  int blocks_y = height / 8;
  if (bx >= blocks_x || by >= blocks_y)
    return;

  int px = bx * 8;
  int py = by * 8;
  int pixel_base = py * (int)width + px;
  int coeff_base = by * ((int)width * 8) + bx * 64;

  int16_t block[64];
  dequant_idct_block_8x8_device(in_data + coeff_base, block, quantization);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int p = i * (int)width + j;
      int16_t tmp = block[i * 8 + j] + (int16_t)prediction[pixel_base + p];
      if (tmp < 0)
        tmp = 0;
      else if (tmp > 255)
        tmp = 255;
      out_data[pixel_base + p] = (uint8_t)tmp;
    }
  }
}

__global__ void dct_quantize_kernel(uint8_t *__restrict__ in_data,
                                    uint8_t *__restrict__ prediction,
                                    uint32_t width, uint32_t height,
                                    int16_t *__restrict__ out_data,
                                    uint8_t *__restrict__ quantization) {
  int bx =
      blockIdx.x * blockDim.x + threadIdx.x; // block index in x (8x8 blocks)
  int by = blockIdx.y * blockDim.y + threadIdx.y; // block index in y

  int blocks_x = width / 8;
  int blocks_y = height / 8;
  if (bx >= blocks_x || by >= blocks_y)
    return;

  int px = bx * 8;
  int py = by * 8;
  int pixel_base = py * (int)width + px;

  // Preserve your existing coeff layout:
  // row stride in coeff space = width * 8, each 8x8 coeff block = 64 entries
  int coeff_base = by * ((int)width * 8) + bx * 64;

  int16_t block[64];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int p = i * (int)width + j;
      block[i * 8 + j] = (int16_t)in_data[pixel_base + p] -
                         (int16_t)prediction[pixel_base + p];
    }
  }

  dct_quant_block_8x8_device(block, out_data + coeff_base, quantization);
}

void launch_quantdct_inter(const quant_inter_args &a, cudaStream_t stream_y,
                           cudaStream_t stream_u, cudaStream_t stream_v) {
  dim3 block(16, 8);
  dim3 gridY((a.wY / 8 + block.x - 1) / block.x,
             (a.hY / 8 + block.y - 1) / block.y);
  dim3 gridU((a.wU / 8 + block.x - 1) / block.x,
             (a.hU / 8 + block.y - 1) / block.y);
  dim3 gridV((a.wV / 8 + block.x - 1) / block.x,
             (a.hV / 8 + block.y - 1) / block.y);

  nvtxRangePushA("dct_quantize");
  dct_quantize_kernel<<<gridY, block, 0, stream_y>>>(a.inY, a.predY, a.wY, a.hY,
                                                     a.resY, a.qY);
  dct_quantize_kernel<<<gridU, block, 0, stream_u>>>(a.inU, a.predU, a.wU, a.hU,
                                                     a.resU, a.qU);

  dct_quantize_kernel<<<gridV, block, 0, stream_v>>>(a.inV, a.predV, a.wV, a.hV,
                                                     a.resV, a.qV);

  nvtxRangePop();

  nvtxRangePushA("dequantize_idct");
  dequantize_idct_kernel<<<gridY, block, 0, stream_y>>>(a.resY, a.predY, a.wY,
                                                        a.hY, a.recY, a.qY);

  dequantize_idct_kernel<<<gridU, block, 0, stream_u>>>(a.resU, a.predU, a.wU,
                                                        a.hU, a.recU, a.qU);

  dequantize_idct_kernel<<<gridV, block, 0, stream_v>>>(a.resV, a.predV, a.wV,
                                                        a.hV, a.recV, a.qV);

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

  a.qY = cm->quanttbl[Y_COMPONENT];
  a.qU = cm->quanttbl[U_COMPONENT];
  a.qV = cm->quanttbl[V_COMPONENT];

  a.wY = cm->padw[Y_COMPONENT];
  a.hY = cm->padh[Y_COMPONENT];
  a.wU = cm->padw[U_COMPONENT];
  a.hU = cm->padh[U_COMPONENT];
  a.wV = cm->padw[V_COMPONENT];
  a.hV = cm->padh[V_COMPONENT];

  return a;
}