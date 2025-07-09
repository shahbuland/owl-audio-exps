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

The script will:
1. Build the Docker image locally
2. Tag it for your remote registry
3. Push to the configured registry
4. Update `skypilot/config.yaml` with the new image tag

## Multi-Node Training with SkyPilot

### Setup
1. Edit `skypilot/config.yaml` to specify your training configuration:
   ```yaml
   # Change this line to point to your config file
   train.py --config_path configs/YOUR_CONFIG.yml
   ```

2. Optionally adjust the number of nodes and GPU type:
   ```yaml
   resources:
     accelerators: H200:8  # 8 H200s per node
   num_nodes: 2            # Number of nodes
   ```

### Launch Training
```bash
# Build and push your container
./build_and_push.sh

# Launch multi-node training on SkyPilot
sky launch skypilot/config.yaml

# Check job status
sky status

# View logs
sky logs <cluster_name>
```

