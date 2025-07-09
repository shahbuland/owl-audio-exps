#!/bin/bash

# Build, tag, and push Docker image script

set -e  # Exit on any error

# Configuration
PROJECT_ID="openworld-main"
REGISTRY="us-central1-docker.pkg.dev"
REPOSITORY="skypilot"
IMAGE_NAME="owl-wm-cond"
TAG=${1:-latest}

# Constructed image names
LOCAL_TAG="${REGISTRY}/owl-wms/cond:${TAG}"
REMOTE_TAG="${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image..."
docker build -t $LOCAL_TAG .

echo "Tagging image for remote repository..."
docker tag $LOCAL_TAG $REMOTE_TAG

echo "Pushing image to remote repository..."
docker push $REMOTE_TAG

echo "Build and push completed successfully!"
echo "Local tag: $LOCAL_TAG"
echo "Remote tag: $REMOTE_TAG"