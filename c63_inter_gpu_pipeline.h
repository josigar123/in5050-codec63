#ifndef C63_INTER_GPU_PIPELINE_H_
#define C63_INTER_GPU_PIPELINE_H_

#include "c63.h"
#include "me_device.h"
#include "quantdct_device.h"
#include <cuda_runtime.h>
#include <stdint.h>

void c63_inter_gpu_pipeline(quant_inter_args &q, motion_inter_args &m,
                            cudaStream_t stream_y, cudaStream_t stream_u,
                            cudaStream_t stream_v);

#endif