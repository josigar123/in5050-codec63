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
  uint8_t *qY, *qU, *qV;
  int wY, hY, wU, hU, wV, hV;
};

void init_quantdct_constants(void);

void launch_quantdct_inter(const quant_inter_args &a, cudaStream_t stream_y,
                           cudaStream_t stream_u, cudaStream_t stream_v);

#endif /* QUANTDCT_DEVICE_H_ */