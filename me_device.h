#ifndef C63_ME_H_
#define C63_ME_H_

#include "c63.h"
#include <cuda_runtime.h>
#include <stdint.h>

struct motion_inter_args {
  // ME inputs
  const uint8_t *orig_Y, *orig_U, *orig_V;
  const uint8_t *recons_Y, *recons_U, *recons_V;

  // MV outputs / MC inputs
  macroblock *mbs_Y, *mbs_U, *mbs_V;

  // MC outputs
  uint8_t *pred_Y, *pred_U, *pred_V;

  // geometry
  int mb_cols_Y, mb_rows_Y;
  int mb_cols_C, mb_rows_C;
  int w_Y, h_Y, w_U, h_U, w_V, h_V;
  int range_Y, range_C;
};

void launch_motion_inter(const motion_inter_args &a, cudaStream_t stream_y,
                         cudaStream_t stream_u, cudaStream_t stream_v);

#endif /* C63_ME_H_ */
