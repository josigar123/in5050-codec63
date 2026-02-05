#!/bin/bash
# build_dev.sh
# Purpose: Clean and build the experimental branch (CUDA-dev)

set -e

# Variables
BRANCH="CUDA-dev"
BUILD_DIR="build-dev"
SRC_DIR=$(pwd)

echo "Building development branch: $BRANCH"

# Checkout dev branch
git checkout $BRANCH
git pull origin $BRANCH

# Clean old build
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

# Run CMake and build
cmake ..
make -j

echo "Development build complete. Binaries are in: $SRC_DIR"

