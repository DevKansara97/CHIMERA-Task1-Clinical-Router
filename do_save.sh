#!/usr/bin/env bash
set -e

DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-chimera-task1-router:latest}"
OUTPUT_FILE="chimera-task1-router.tar.gz"

echo "=+= Saving Docker image to ${OUTPUT_FILE} for Grand Challenge submission ..."
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$OUTPUT_FILE"
echo "Saved ${OUTPUT_FILE} ($(du -h "$OUTPUT_FILE" | cut -f1))"
