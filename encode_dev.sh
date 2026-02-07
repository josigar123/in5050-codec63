#!/bin/bash
# encode_dev.sh
# Purpose: Encode video using the development build (CUDA-dev branch)

set -e

# Variables
BUILD_DIR="build-dev"
INPUT_VIDEO="/mnt/sdcard/cipr/foreman.yuv"
OUTPUT_FILE="foremanout-dev.c63"
WIDTH=352
HEIGHT=288
FPS=10

echo "Encoding $INPUT_VIDEO using development branch..."

cd $BUILD_DIR
../c63enc -w $WIDTH -h $HEIGHT -f $FPS -o $OUTPUT_FILE ../$INPUT_VIDEO

echo "Encoding complete. Output: $BUILD_DIR/$OUTPUT_FILE"

