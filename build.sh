#!/bin/bash

# TODO: Seems this image is x86 only and won't work on ARM
#BASE_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release"
#BASE_IMAGE_TAG="main"

BASE_IMAGE="gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci"
BASE_IMAGE_TAG="tensorrt_llm_release_48db263_aarch64"
LOCAL_TAG="dynamo_trtllm_arm64"
PLATFORM="linux/arm64"
docker pull --platform ${PLATFORM} "${BASE_IMAGE}:${BASE_IMAGE_TAG}"
./container/build.sh --framework TENSORRTLLM --base-image ${BASE_IMAGE} --base-image-tag ${BASE_IMAGE_TAG} --platform ${PLATFORM} --tag ${LOCAL_TAG}


REMOTE_IMAGE="gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci"
REMOTE_TAG="${REMOTE_IMAGE}:rmccormick_dynamo_${BASE_IMAGE_TAG}"
docker tag ${LOCAL_TAG} ${REMOTE_TAG}
docker push ${REMOTE_TAG}
