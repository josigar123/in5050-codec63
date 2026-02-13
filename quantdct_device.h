#ifndef QUANTDCT_DEVICE_H_
#define QUANTDCT_DEVICE_H_

#include <cuda_runtime.h>

#include "quantdct.h"
#include "tables.h"

void quantize_dct_dequantize_idct_inter(struct c63_common *cm, yuv_t *image);

#endif /* QUANTDCT_DEVICE_H_ */