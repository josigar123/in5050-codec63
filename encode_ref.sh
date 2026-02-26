#!/bin/bash
# encode_ref.sh
# Purpose: Encode video using the reference build (CUDA branch)

set -e

# Variables
BUILD_DIR="build-ref"
INPUT_VIDEO="foreman.yuv"
OUTPUT_FILE="foremanout-ref.c63"
WIDTH=352
HEIGHT=288
FPS=10

echo "Encoding $INPUT_VIDEO using reference branch..."

cd $BUILD_DIR
../c63enc -w $WIDTH -h $HEIGHT -f $FPS -o $OUTPUT_FILE ../$INPUT_VIDEO

echo "Encoding complete. Output: $BUILD_DIR/$OUTPUT_FILE"

