#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-chimera-task1-router:latest}"

echo "=+= Building Docker Image: ${DOCKER_IMAGE_TAG}"
docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  "$SCRIPT_DIR"
