#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "common.h"

void destroy_frame(struct frame *f) {
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) {
    return;
  }

  cudaFree(f->recons->Y);
  cudaFree(f->recons->U);
  cudaFree(f->recons->V);
  cudaFree(f->recons);

  cudaFree(f->residuals->Ydct);
  cudaFree(f->residuals->Udct);
  cudaFree(f->residuals->Vdct);
  cudaFree(f->residuals);

  cudaFree(f->predicted->Y);
  cudaFree(f->predicted->U);
  cudaFree(f->predicted->V);
  cudaFree(f->predicted);

  cudaFree(f->mbs[Y_COMPONENT]);
  cudaFree(f->mbs[U_COMPONENT]);
  cudaFree(f->mbs[V_COMPONENT]);

  cudaFree(f);
}

struct frame *create_frame(struct c63_common *cm, yuv_t *image) {
  frame *f;
  cudaMallocManaged(&f, sizeof(struct frame));

  f->orig = image;

  cudaMallocManaged(&f->recons, sizeof(yuv_t));
  cudaMallocManaged(&f->recons->Y, cm->ypw * cm->yph);
  cudaMallocManaged(&f->recons->U, cm->upw * cm->uph);
  cudaMallocManaged(&f->recons->V, cm->vpw * cm->vph);

  cudaMallocManaged(&f->predicted, sizeof(yuv_t));
  cudaMallocManaged(&f->predicted->Y, cm->ypw * cm->yph * sizeof(uint8_t));
  cudaMemset(f->predicted->Y, 0, (cm->ypw * cm->yph) * sizeof(uint8_t));

  cudaMallocManaged(&f->predicted->U, cm->upw * cm->uph * sizeof(uint8_t));
  cudaMemset(f->predicted->U, 0, (cm->upw * cm->uph) * sizeof(uint8_t));

  cudaMallocManaged(&f->predicted->V, cm->vpw * cm->vph * sizeof(uint8_t));
  cudaMemset(f->predicted->V, 0, (cm->vpw * cm->vph) * sizeof(uint8_t));

  cudaMallocManaged(&f->residuals, sizeof(dct_t));
  cudaMallocManaged(&f->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
  cudaMemset(f->residuals->Ydct, 0, (cm->ypw * cm->yph) * sizeof(int16_t));

  cudaMallocManaged(&f->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
  cudaMemset(f->residuals->Udct, 0, (cm->upw * cm->uph) * sizeof(int16_t));

  cudaMallocManaged(&f->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));
  cudaMemset(f->residuals->Vdct, 0, (cm->vpw * cm->vph) * sizeof(int16_t));

  cudaMallocManaged(&f->mbs[Y_COMPONENT],
                    cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
  cudaMemset(f->mbs[Y_COMPONENT], 0,
             (cm->mb_rows * cm->mb_cols) * sizeof(struct macroblock));

  cudaMallocManaged(&f->mbs[U_COMPONENT], cm->mb_rows / 2 * cm->mb_cols / 2 *
                                              sizeof(struct macroblock));
  cudaMemset(f->mbs[U_COMPONENT], 0,
             (cm->mb_rows / 2 * cm->mb_cols / 2) * sizeof(struct macroblock));

  cudaMallocManaged(&f->mbs[V_COMPONENT], cm->mb_rows / 2 * cm->mb_cols / 2 *
                                              sizeof(struct macroblock));
  cudaMemset(f->mbs[V_COMPONENT], 0,
             (cm->mb_rows / 2 * cm->mb_cols / 2) * sizeof(struct macroblock));

  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp) {
  fwrite(image->Y, 1, w * h, fp);
  fwrite(image->U, 1, w * h / 4, fp);
  fwrite(image->V, 1, w * h / 4, fp);
}

void reset_frame_work(reset_frame_work_args &a) {

  cudaMemsetAsync(a.predicted_Y, 0, a.ypw * a.yph * sizeof(uint8_t),
                  a.stream_y);
  cudaMemsetAsync(a.residuals_Ydct, 0, a.ypw * a.yph * sizeof(int16_t),
                  a.stream_y);
  cudaMemsetAsync(a.mbs_Y, 0, a.mb_rows * a.mb_cols * sizeof(struct macroblock),
                  a.stream_y);

  cudaMemsetAsync(a.predicted_U, 0, a.upw * a.uph * sizeof(uint8_t),
                  a.stream_u);
  cudaMemsetAsync(a.residuals_Udct, 0, a.upw * a.uph * sizeof(int16_t),
                  a.stream_u);
  cudaMemsetAsync(a.mbs_U, 0,
                  (a.mb_rows / 2) * (a.mb_cols / 2) * sizeof(struct macroblock),
                  a.stream_u);

  cudaMemsetAsync(a.predicted_V, 0, a.vpw * a.vph * sizeof(uint8_t),
                  a.stream_v);
  cudaMemsetAsync(a.residuals_Vdct, 0, a.vpw * a.vph * sizeof(int16_t),
                  a.stream_v);
  cudaMemsetAsync(a.mbs_V, 0,
                  (a.mb_rows / 2) * (a.mb_cols / 2) * sizeof(struct macroblock),
                  a.stream_v);
}

reset_frame_work_args create_reset_frame_work_args(struct c63_common *cm,
                                                   cudaStream_t stream_y,
                                                   cudaStream_t stream_u,
                                                   cudaStream_t stream_v) {
  reset_frame_work_args a{};

  a.predicted_Y = cm->curframe->predicted->Y;
  a.residuals_Ydct = cm->curframe->residuals->Ydct;
  a.mbs_Y = cm->curframe->mbs[Y_COMPONENT];
  a.ypw = cm->ypw;
  a.yph = cm->yph;

  a.predicted_U = cm->curframe->predicted->U;
  a.residuals_Udct = cm->curframe->residuals->Udct;
  a.mbs_U = cm->curframe->mbs[U_COMPONENT];
  a.upw = cm->upw;
  a.uph = cm->uph;

  a.predicted_V = cm->curframe->predicted->V;
  a.residuals_Vdct = cm->curframe->residuals->Vdct;
  a.mbs_V = cm->curframe->mbs[V_COMPONENT];
  a.vpw = cm->vpw;
  a.vph = cm->vph;

  a.mb_rows = cm->mb_rows;
  a.mb_cols = cm->mb_cols;

  a.stream_y = stream_y;
  a.stream_u = stream_u;
  a.stream_v = stream_v;

  return a;
}