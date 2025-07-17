# owl_wms
Basic world models

## Training

### Deploy Training Codebase and Train on SkyPilot

Setup SkyPilot
```bash
# Install SkyPilot
python3 -m pip install -U skypilot

# Authenticate
sky api login -e https://owlskypilot:<password>@cluster.openworldlabs.ai
```

Create Docker Image for your Current Codebase

Ensure you've configured your Docker Registry settings first. See "Docker Build & Deploy (For multinode training)"
```bash
# dockerizes and sets the $IMAGE_REF environment variable used by SkyPilot
./build_and_push.sh
```

Launch your Trainer
```bash
export EXPERIMENT_NAME=new-attention-pattern-v2
export TRAIN_CONFIG=skypilot/config.yaml

# Provision Single Node
sky launch --infra kubernetes --gpus H200:8 --num-nodes 1 --name $EXPERIMENT_NAME $TRAIN_CONFIG

# **OR** Provision Multiple Nodes
sky launch --infra kubernetes --gpus H200:8 --num-nodes 2 --name $EXPERIMENT_NAME $TRAIN_CONFIG
```

SkyPilot Basic Commands
```bash
# Launch multi-node training on SkyPilot
sky launch skypilot/config.yaml

# Check job status
sky status

# View logs
sky logs <cluster_name>
```


### Train On Other Hosts

Setup the model / trainer / requirements
```bash
export REPO=https://github.com/wayfarer-labs/owl-wms
export EXPERIMENT_COMMIT=20bb0973336ab8696ad60e26bf1a7d5004191c70

git clone --recursive -j8 $REPO
cd owl-wms
git fetch && git checkout $EXPERIMENT_COMMIT
pip install -r requirements.txt
pip install -r owl-vaes/requirements.txt

# Configure WandB
wandb login
```

Then run training:
```bash
# Single GPU
python train.py --config_path configs/basic.yml

# Multi-GPU (single node)
torchrun --nproc_per_node=8 train.py --config_path configs/basic.yml
```

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

### Prerequisites
1. Make sure you're authenticated with Google Cloud:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. Make sure you set your Project ID for google cloud.
