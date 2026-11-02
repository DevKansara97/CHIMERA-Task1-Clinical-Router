#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-chimera-task1-router:latest}"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"
chmod -R 777 "${OUTPUT_DIR}"

echo "=+= Testing container inference offline on sample case"
docker run --rm \
  --network none \
  --volume "${INPUT_DIR}":/input:ro \
  --volume "${OUTPUT_DIR}":/output \
  "$DOCKER_IMAGE_TAG"

echo "=+= Prediction Output:"
cat "${OUTPUT_DIR}/prostate-biopsy-decision.json"
echo ""
cat "${OUTPUT_DIR}/prostate-biopsy-decision-reasoning.json"
echo ""
