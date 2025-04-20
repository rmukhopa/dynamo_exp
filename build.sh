#!/bin/bash
docker pull --platform linux/arm64 urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main
./container/build.sh --framework TENSORRTLLM --target dev --base-image urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release --base-image-tag main
