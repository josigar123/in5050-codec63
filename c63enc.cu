#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "c63.h"
#include "c63_inter_gpu_pipeline.h"
#include "c63_write.h"
#include "common.h"
#include "nvtx3/nvToolsExt.h"
#include "quantdct_device.h"
#include "tables.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

// Allocates memory for the input image
static yuv_t *alloc_input_image(struct c63_common *cm) {
  yuv_t *image;
  cudaMallocHost(&image, sizeof(yuv_t));
  cudaMallocHost(&image->Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] *
                                sizeof(uint8_t));
  cudaMallocHost(&image->U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] *
                                sizeof(uint8_t));
  cudaMallocHost(&image->V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] *
                                sizeof(uint8_t));
  return image;
}

// Frees the memory for the input image
static void free_input_image(yuv_t *image) {
  cudaFreeHost(image->Y);
  cudaFreeHost(image->U);
  cudaFreeHost(image->V);
  cudaFreeHost(image);
}

// This function reads the YUV file into memory, memory has already been
// allocated (See helpers above)
/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static int read_yuv_into(FILE *file, struct c63_common *cm, yuv_t *image,
                         cudaStream_t stream_y, cudaStream_t stream_u,
                         cudaStream_t stream_v) {
  size_t len = 0;
  size_t y_bytes =
      cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t);
  size_t u_bytes =
      cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t);
  size_t v_bytes =
      cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t);

  cudaMemset(image->Y, 0, y_bytes);
  cudaMemset(image->U, 0, u_bytes);
  cudaMemset(image->V, 0, v_bytes);

  len += fread(image->Y, 1, width * height, file);
  len += fread(image->U, 1, (width * height) / 4, file);
  len += fread(image->V, 1, (width * height) / 4, file);

  if (ferror(file)) {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
    return 0;
  if (len != (size_t)(width * height * 1.5f)) {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);
    return 0;
  }

  return 1;
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image,
                             cudaStream_t stream_y, cudaStream_t stream_u,
                             cudaStream_t stream_v) {
  /* Advance to next frame */
  /* ping pong between the two allocated frames*/
  struct frame *tmp = cm->refframe;
  cm->refframe = cm->curframe;
  cm->curframe = tmp;

  cm->curframe->orig = image;

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    fprintf(stderr, " (keyframe) ");
  } else {
    cm->curframe->keyframe = 0;
  }

  // Take a snapshot of arguments for passing to the pipeline and work reset
  reset_frame_work_args a =
      create_reset_frame_work_args(cm, stream_y, stream_u, stream_v);

  quant_inter_args q = create_quant_inter_args(cm, image);

  motion_inter_args m = create_motion_inter_args(cm);

  reset_frame_work(a);

  // Run the ME/MC and DCT/Quant/DeQuant/IDCT pipeline, the test for chceking if
  // we have a keyframe is inside the pipeline
  c63_inter_gpu_pipeline(q, m, stream_y, stream_u, stream_v);

  /* Function dump_image(), found in common.c, can be used here to check if the
   prediction is correct */
}

struct c63_common *init_c63_enc(int width, int height) {
  int i;

  c63_common *cm;
  cudaMallocManaged(&cm, sizeof(struct c63_common));
  cudaMemset(cm, 0, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width / 16.0f) * 16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height / 16.0f) * 16);
  cm->padw[U_COMPONENT] = cm->upw =
      (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
  cm->padh[U_COMPONENT] = cm->uph =
      (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
  cm->padw[V_COMPONENT] = cm->vpw =
      (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
  cm->padh[V_COMPONENT] = cm->vph =
      (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                 // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;    // Pixels in every direction
  cm->keyframe_interval = 100; // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i) {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  // Copy tables into constant memory
  init_quantdct_constants(cm);

  return cm;
}

void free_c63_enc(struct c63_common *cm) {
  destroy_frame(cm->curframe);
  cudaFree(cm);
}

static void print_help() {
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int c;
  yuv_t *image;

  if (argc == 1) {
    print_help();
  }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1) {
    switch (c) {
    case 'h':
      height = atoi(optarg);
      break;
    case 'w':
      width = atoi(optarg);
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'f':
      limit_numframes = atoi(optarg);
      break;
    default:
      print_help();
      break;
    }
  }

  if (optind >= argc) {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");

  if (outfile == NULL) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];

  if (limit_numframes) {
    printf("Limited to %d frames.\n", limit_numframes);
  }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;

  // Create a streams for the encoding process
  cudaStream_t stream_y;
  cudaStream_t stream_u;
  cudaStream_t stream_v;
  cudaStreamCreate(&stream_y);
  cudaStreamCreate(&stream_u);
  cudaStreamCreate(&stream_v);

  // Here we allocate memory for the input image once, with managed memory
  image = alloc_input_image(cm);

  // Here we create two frames, one for the reference frame and one for the
  // current frame We allocate and memset to zero, allocation happens once, a
  // helper will be used to reset frame work
  struct frame *frame_a = create_frame(cm, image);
  struct frame *frame_b = create_frame(cm, image);

  // Here we set the reference frame and current frame, and initialize the frame
  // number and frames since keyframe
  cm->refframe = frame_a;
  cm->curframe = frame_b;
  cm->framenum = 0;
  cm->frames_since_keyframe = 0;

  while (1) {
    if (!read_yuv_into(infile, cm, image, stream_y, stream_u, stream_v)) {
      break;
    }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image, stream_y, stream_u, stream_v);

    cudaStreamSynchronize(stream_y);
    cudaStreamSynchronize(stream_u);
    cudaStreamSynchronize(stream_v);

    write_frame(cm);

    printf("Done!\n");

    ++numframes;

    ++cm->framenum;
    ++cm->frames_since_keyframe;

    if (limit_numframes && numframes >= limit_numframes) {
      break;
    }
  }

  cudaStreamDestroy(stream_y);
  cudaStreamDestroy(stream_u);
  cudaStreamDestroy(stream_v);

  // We only free input image once at the end of encoding, since it is used for
  // every frame
  free_input_image(image);
  destroy_frame(frame_a);
  destroy_frame(frame_b);
  cudaFree(cm);
  // free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  return EXIT_SUCCESS;
}
