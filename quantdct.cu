#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <nvtx3/nvToolsExt.h>

#include "tables.h"

#define ISQRT2 0.70710678118654f

static void dct_2d(const float *in, float *out) {

  /*
    u,v are frequency componets
    x, y are spatial coordinates

    In the two inner loops we loop through all spatial coordinates of the block
    which weighs the cosine contributions. So each coefficient is a weighted sum
    of the basis contributions weighted by the spatial coordiantes

    OPTIMIZATION?: For a given block we iterate over it 64 times which would
    yield the same values, these could maybe be cached in shared memory?
  */
  // Loop through all elements of the block
  for (int v = 0; v < 8; v++) {
    for (int u = 0; u < 8; u++) {
      /* Compute the DCT */
      float dct = 0;
      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          dct +=
              in[y * 8 + x] * dctlookup[x][u] *
              dctlookup[y][v]; // Table lookup are precomputed basis for cosine
        }
      }

      out[v * 8 + u] = dct; // 8x8 coefficient block
    }
  }
}

// Same as dct_2d, but reverse lookup, same optimization could work
// DANGER: Also used by the decoder
static void idct_2d(const float *in, float *out) {
  // Loop through all elements of the block
  for (int v = 0; v < 8; v++) {
    for (int u = 0; u < 8; u++) {
      /* Compute the iDCT */
      float dct = 0;
      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          dct += in[y * 8 + x] * dctlookup[u][x] * dctlookup[v][y];
        }
      }

      out[v * 8 + u] = dct;
    }
  }
}

/*
  scales with 1/sqrt(2) if u or v is 0, otherwise 1 nothing special going on

  OPTIMIZATION?:  we could coalesce the scaling into the original dct loop so we
  dont do 64 extra multiplications per block
*/
static void scale_block(float *in_data, float *out_data) {
  int u, v;

  for (v = 0; v < 8; ++v) {
    for (u = 0; u < 8; ++u) {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
    }
  }
}

/*
  in_data is the DCT coefficients
  out_data is the quantized block
  quant_tbl is the quantization table for a given color component
*/
static void quantize_block(float *in_data, float *out_data,
                           uint8_t *quant_tbl) {
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = zigzag_U[zigzag]; // Table for horizontal frequency component
    uint8_t v = zigzag_V[zigzag]; // Table for vertical frequency component

    float dct = in_data[v * 8 + u]; // Get the coefficient

    /* Zig-zag and quantize */
    out_data[zigzag] = (float)round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

/*
  reconstructed lossy dct coefficient block, same logic as quantize_block but
  reverse
*/
// DANGER: Also used by the decoder
static void dequantize_block(float *in_data, float *out_data,
                             uint8_t *quant_tbl) {
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v * 8 + u] = (float)round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

/*
  Just runs the pipeline: dct -> scale -> quantize

  OPTIMIZATION?: Instead of the last for-loop copy, we could pass out_data
  directly into quantize_block instead of mb2
*/
static void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
                                uint8_t *quant_tbl) {
  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  for (int i = 0; i < 64; i++) {
    mb[i] = in_data[i];
  }

  dct_2d(mb, mb2);
  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (int i = 0; i < 64; i++) {
    out_data[i] = mb2[i];
  }
}

/*
 Same logic as above, but the inverse operations:

 OPTIMIZATION?: Same as above
*/
// DANGER: Also used by the decoder
static void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
                                   uint8_t *quant_tbl) {
  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  for (int i = 0; i < 64; i++) {
    mb[i] = in_data[i];
  }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);
  idct_2d(mb, mb2);

  for (int i = 0; i < 64; i++) {
    out_data[i] = mb2[i];
  }
}

/*
  Function performs dequant + idct row-by-row on the residual frame

 in_data is the DCT + quantized rows of the residual frame
*/
// DANGER: Also used by the decoder
static void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w,
                                int h, int y, uint8_t *out_data,
                                uint8_t *quantization) {
  int x;

  int16_t block[8 * 8];

  /* Perform the dequantization and iDCT */
  for (x = 0; x < w; x += 8) {
    int i, j;

    // Write the dequantized and iDCTed block to the block array
    dequant_idct_block_8x8(in_data + (x * 8), block, quantization);

    for (i = 0; i < 8; ++i) {
      for (j = 0; j < 8; ++j) {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        // By adding the prediction block to the residual we get an
        // approximation of the original frame
        int16_t tmp =
            block[i * 8 + j] +
            (int16_t)prediction[i * w + j + x]; // x index gives the position of
                                                // the block in the frame

        if (tmp < 0) {
          tmp = 0;
        } else if (tmp > 255) {
          tmp = 255;
        }

        // Write the reconstructed pixel to the output frame
        out_data[i * w + j + x] = tmp;
      }
    }
  }
}

/*
 Calculates the residual frame from prediction and original frame.
 Then it quantizes the residual row-by-row
*/
static void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w,
                             int h, int16_t *out_data, uint8_t *quantization) {
  int x;

  int16_t block[8 * 8];

  /* Perform the DCT and quantization */
  for (x = 0; x < w; x += 8) {
    int i, j;

    // Calculate residual frame from prediction and original frame
    for (i = 0; i < 8; ++i) {
      for (j = 0; j < 8; ++j) {
        block[i * 8 + j] =
            ((int16_t)in_data[i * w + j + x] - prediction[i * w + j + x]);
      }
    }

    /* Store MBs linear in memory, i.e. the 64 coefficients are stored
       continous. This allows us to ignore stride in DCT/iDCT and other
       functions.

       out_data + (x * 8) points to the beginning of the next block of 8x8
       quantized coefficients
    */
    dct_quant_block_8x8(block, out_data + (x * 8), quantization);
  }
}

// DANGER: Also used by the decoder
void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
                     uint32_t height, uint8_t *out_data,
                     uint8_t *quantization) {
  nvtxRangePushA("dequantize_idct");
  int y;

  // Dequantize all rows of the residual frame moving vertically, reconstructing
  // the frame row-by-row
  for (y = 0; y < height; y += 8) {
    dequantize_idct_row(in_data + y * width, prediction + y * width, width,
                        height, y, out_data + y * width, quantization);
  }
  nvtxRangePop();
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
                  uint32_t height, int16_t *out_data, uint8_t *quantization) {
  nvtxRangePushA("dct_quantize");
  int y;

  // Quantize all rows of the frame moving vertically
  for (y = 0; y < height; y += 8) {
    dct_quantize_row(in_data + y * width, prediction + y * width, width, height,
                     out_data + y * width, quantization);
  }
  nvtxRangePop();
}
