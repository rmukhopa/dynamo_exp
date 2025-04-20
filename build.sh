#!/bin/bash

# TODO: Seems this image is x86 only and won't work on ARM
#BASE_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release"
#BASE_IMAGE_TAG="main"

# TODO: apt install libopenmpi-dev + pip install mpi4py has issues finding mpi.h (on ARM only?)
BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
BASE_IMAGE_TAG="25.03-cuda12.8-devel-ubuntu24.04"

BASE_IMAGE="ubuntu"
BASE_IMAGE_TAG="24.04"
PLATFORM="linux/arm64"
TAG="dynamo_trtllm_arm64"
docker pull --platform ${PLATFORM} "${BASE_IMAGE}:${BASE_IMAGE_TAG}"
./container/build.sh --framework TENSORRTLLM --base-image ${BASE_IMAGE} --base-image-tag ${BASE_IMAGE_TAG} --platform ${PLATFORM} --tag ${TAG}

REMOTE_TAG="gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci:rmccormick-${TAG}"
docker tag ${TAG} ${REMOTE_TAG}
docker push ${REMOTE_TAG}
