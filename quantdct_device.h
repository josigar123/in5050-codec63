#ifndef QUANTDCT_DEVICE_H_
#define QUANTDCT_DEVICE_H_

#include "c63.h"
#include "tables.h"
#include <cuda_runtime.h>
#include <stdint.h>

struct quant_inter_args {
  uint8_t *inY, *inU, *inV;
  uint8_t *predY, *predU, *predV;
  int16_t *resY, *resU, *resV;
  uint8_t *recY, *recU, *recV;
  int wY, hY, wU, hU, wV, hV;
};

void init_quantdct_constants(const c63_common *cm);

void launch_quantdct_inter(const quant_inter_args &a, cudaStream_t stream_y,
                           cudaStream_t stream_u, cudaStream_t stream_v);

struct quant_inter_args create_quant_inter_args(struct c63_common *cm,
                                                yuv_t *image);
#endif /* QUANTDCT_DEVICE_H_ */