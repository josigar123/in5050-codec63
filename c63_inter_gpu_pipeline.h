#ifndef C63_INTER_GPU_PIPELINE_H_
#define C63_INTER_GPU_PIPELINE_H_

#include "c63.h"
#include <cuda_runtime.h>
#include <stdint.h>

void c63_inter_gpu_pipeline(struct c63_common *cm, yuv_t *image,
                            cudaStream_t stream_y, cudaStream_t stream_u,
                            cudaStream_t stream_v);

#endif