#!/bin/bash
# build_ref.sh
# Purpose: Clean and build the reference branch (CUDA)

set -e  # Exit on error

# Variables
BRANCH="CUDA"
BUILD_DIR="build-ref"
SRC_DIR=$(pwd)

echo "Building reference branch: $BRANCH"

# Checkout reference branch
git checkout $BRANCH
git pull origin $BRANCH

# Clean old build
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

# Run CMake and build
cmake ..
make -j

echo "Reference build complete. Binaries are in: $SRC_DIR"
