#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include "c63.h"
#include <cuda_runtime.h>
#include <inttypes.h>

struct reset_frame_work_args {
  uint8_t *predicted_Y;
  int16_t *residuals_Ydct;
  struct macroblock *mbs_Y;
  int ypw;
  int yph;

  uint8_t *predicted_U;
  int16_t *residuals_Udct;
  struct macroblock *mbs_U;
  int upw;
  int uph;

  uint8_t *predicted_V;
  int16_t *residuals_Vdct;
  struct macroblock *mbs_V;
  int vpw;
  int vph;

  int mb_rows;
  int mb_cols;

  cudaStream_t stream_y;
  cudaStream_t stream_u;
  cudaStream_t stream_v;
};

// Declarations
struct frame *create_frame(struct c63_common *cm, yuv_t *image);

void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

void reset_frame_work(reset_frame_work_args &a);

reset_frame_work_args create_reset_frame_work_args(struct c63_common *cm,
                                                   cudaStream_t stream_y,
                                                   cudaStream_t stream_u,
                                                   cudaStream_t stream_v);

#endif /* C63_COMMON_H_ */
