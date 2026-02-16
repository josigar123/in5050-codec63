
#include "c63_inter_gpu_pipeline.h"
#include "me_device.h"
#include "quantdct_device.h"
#include <nvtx3/nvToolsExt.h>

void c63_inter_gpu_pipeline(struct c63_common *cm, yuv_t *image,
                            cudaStream_t stream) {
  nvtxRangePushA("c63_inter_gpu_pipeline");

  // Snapshot the quant arguments before first launch, always create a snapshot
  // of these
  quant_inter_args q = {image->Y,
                        image->U,
                        image->V,
                        cm->curframe->predicted->Y,
                        cm->curframe->predicted->U,
                        cm->curframe->predicted->V,
                        cm->curframe->residuals->Ydct,
                        cm->curframe->residuals->Udct,
                        cm->curframe->residuals->Vdct,
                        cm->curframe->recons->Y,
                        cm->curframe->recons->U,
                        cm->curframe->recons->V,
                        cm->quanttbl[Y_COMPONENT],
                        cm->quanttbl[U_COMPONENT],
                        cm->quanttbl[V_COMPONENT],
                        cm->padw[Y_COMPONENT],
                        cm->padh[Y_COMPONENT],
                        cm->padw[U_COMPONENT],
                        cm->padh[U_COMPONENT],
                        cm->padw[V_COMPONENT],
                        cm->padh[V_COMPONENT]};

  if (!cm->curframe->keyframe) {
    // Capture snapshot only if its not a keyframe
    motion_inter_args m = {cm->curframe->orig->Y,
                           cm->curframe->orig->U,
                           cm->curframe->orig->V,
                           cm->refframe->recons->Y,
                           cm->refframe->recons->U,
                           cm->refframe->recons->V,
                           cm->curframe->mbs[Y_COMPONENT],
                           cm->curframe->mbs[U_COMPONENT],
                           cm->curframe->mbs[V_COMPONENT],
                           cm->curframe->predicted->Y,
                           cm->curframe->predicted->U,
                           cm->curframe->predicted->V,
                           cm->mb_cols,
                           cm->mb_rows,
                           cm->mb_cols / 2,
                           cm->mb_rows / 2,
                           cm->padw[Y_COMPONENT],
                           cm->padh[Y_COMPONENT],
                           cm->padw[U_COMPONENT],
                           cm->padh[U_COMPONENT],
                           cm->padw[V_COMPONENT],
                           cm->padh[V_COMPONENT],
                           cm->me_search_range,
                           cm->me_search_range / 2};

    // Launch the motion estimation and compensation
    launch_motion_inter(m, stream);
  }

  // Launch the DCT/Quant/DeQuant/IDCT pipeline
  launch_quantdct_inter(q, stream);

  nvtxRangePop();
}