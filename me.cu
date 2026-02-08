#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "me.h"

/*
block1 and block2 points to the beginning of a 8x8 block in the entire frame.
stride is the widht of the entire frame in pixels (bytes).

v * stride is used to calculate the beginning of the next row. u is then used to
index the columns of the current row.
*/
static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride,
                          int *result) {
  int u, v;

  *result = 0;

  for (v = 0; v < 8; ++v) {
    for (u = 0; u < 8; ++u) {
      *result += abs(block2[v * stride + u] - block1[v * stride + u]);
    }
  }
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
                         uint8_t *orig, uint8_t *ref, int color_component) {

  /*
    For a given color_component (Y, U, V) mbs[color_component] will return a
    pointer to the components macroblocks for the current frame.

  The second index indexed the actual macroblock to motion estimate on.

  mb_y = the row index of the macroblock
  cm->padw[color_component] / 8 = the width of the macroblock in pixels
  mb_x = the column index of the macroblock
  */
  struct macroblock *mb =
      &cm->curframe
           ->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. color_component == 0 means Y. It is
   * quarted since we half the resolution of the U and V channels both*/
  if (color_component > 0) {
    range /= 2;
  }

  /*Bounding box for the search window for the reference frame. This is needed
   * so that we do not exceed the bounds of the reference frame in any
   * direction, so that when we calculate SAD with a macroblock in orig with
   * ref, the macroblock in ref will fully be in ref and not exceed it*/
  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) {
    left = 0;
  }
  if (top < 0) {
    top = 0;
  }
  if (right > (w - 8)) {
    right = w - 8;
  }
  if (bottom > (h - 8)) {
    bottom = h - 8;
  }

  int x, y;

  /* Points to the top-left corner of the macroblock (the start of each
   * macroblock to iterate over)*/
  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;

  for (y = top; y < bottom; ++y) {
    for (x = left; x < right; ++x) {
      int sad;
      /*
        We multiply my * w and y * w since this is how we get from one row to
        another, since the frames are stored in a row major order.
      */
      sad_block_8x8(orig + my * w + mx, ref + y * w + x, w, &sad);

      /* printf("(%4d,%4d) - %d\n", x, y, sad); */

      if (sad < best_sad) {
        /*
          Store the cooridnates of the motion vector. e.g the displacement in 2D
          that tells the us where to ind the macroblock in the reference framce

          The vector is the offset (e.g how to reach the best predicted
          macroblock in the reference frame) from the current macroblock. When
          using the offset we get the top-left corner of the best predicted
          macroblock in the reference frame.
        */
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm) {
  /* Compare this frame with previous reconstructed frame (e.g the reference
   * frame)*/
  int mb_x, mb_y;

  /* Luma */
  /* For each macroblock in the luma frame estimate the motion vector from the
   * reconstructed reference frame (after iDCT/iQuant)*/
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y) {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x) {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
                   cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  /* For each macroblock in the chroma frames (U, V) estimate the motion
   * vector*/
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y) {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x) {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
                   cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
                   cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

// DANGER: Is also used by the decoder
/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
                         uint8_t *predicted, uint8_t *ref,
                         int color_component) {

  /*
    The function simply copies the block from the reference frame to the
    predicted block

    It is calculated by current position + motion vector offset.
  */
  struct macroblock *mb =
      &cm->curframe
           ->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  // If we do not use a motion vector, just return
  if (!mb->use_mv) {
    return;
  }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y) {
    for (x = left; x < right; ++x) {
      predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

/*
  This function copies every macroblock in the current framte to a predicted
  frame. We use the motion vectors to copy the macroblocks from the reference
  framte to the predicted frame.

  For U and V channels we have half the block in each direction (x, y)
*/
// DANGER: Is also used by the decoder
void c63_motion_compensate(struct c63_common *cm) {
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y) {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x) {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
                   cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y) {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x) {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
                   cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
                   cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
