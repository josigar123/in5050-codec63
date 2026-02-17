
#include "c63_inter_gpu_pipeline.h"
#include "me_device.h"
#include "quantdct_device.h"
#include <nvtx3/nvToolsExt.h>

void c63_inter_gpu_pipeline(quant_inter_args &q, motion_inter_args &m,
                            cudaStream_t stream_y, cudaStream_t stream_u,
                            cudaStream_t stream_v) {
  nvtxRangePushA("c63_inter_gpu_pipeline");

  if (!m.is_keyframe) {
    // Launch the motion estimation and compensation
    launch_motion_inter(m, stream_y, stream_u, stream_v);
  }

  // Launch the DCT/Quant/DeQuant/IDCT pipeline
  launch_quantdct_inter(q, stream_y, stream_u, stream_v);

  nvtxRangePop();
}