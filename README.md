# owl_wms
Basic world models

## Docker Build & Deploy (For multinode training)

### Setup
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Configure your Docker registry settings in `.env`:
   ```bash
   # Docker Registry Configuration
   PROJECT_ID=your-project-id
   REGISTRY=us-central1-docker.pkg.dev
   REPOSITORY=your-repository
   IMAGE_NAME=your-image-name
   DEFAULT_TAG=latest
   
   # Local build configuration
   LOCAL_REGISTRY=us-central1-docker.pkg.dev
   LOCAL_PROJECT=your-local-project
   LOCAL_REPOSITORY=your-local-repo
   ```

### Build and Deploy
```bash
# Build, tag, and push with default tag
./build_and_push.sh

# Build, tag, and push with custom tag
./build_and_push.sh v1.0.0
```
