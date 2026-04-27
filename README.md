# Abot Explorer

[![arXiv](https://img.shields.io/badge/arXiv-2604.19034-b31b1b.svg)](http://arxiv.org/abs/2604.19034)

**Paper:** http://arxiv.org/abs/2604.19034

Abot Explorer is an autonomous exploration agent powered by vision-language models for 3D scene understanding and navigation. It supports inference and evaluation on both **3D Gaussian Splatting (3DGS)** scenes (via PGSR) and **Habitat** simulated environments.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Evaluation Environment Setup](#3-evaluation-environment-setup)
4. [Inference on 3DGS Environment](#4-inference-on-3dgs-environment)
5. [Inference on Habitat Environment](#5-inference-on-habitat-environment)

---


## 1. Prerequisites

- **CUDA**: Version **12.8** has been tested and confirmed to run stably
- **Python**: Version 3.10
- **Conda**: For environment management

---

## 2. Download Code

```bash
git clone http://github/abot-explorer/abot_explorer.git
cd abot_explorer
```

---

## 3. Models and Datasets

- Download the finetuned Qwen2.5_vl model weights from HuggingFace (**[coming soon]**).
- Evaluation scenes include InteriorGS, Matterport3D, and HM3D.
- Scene assets should be downloaded according to the official instructions of each dataset:
  - [InteriorGS](https://github.com/manycore-research/InteriorGS)
  - [Matterport3D](https://niessner.github.io/Matterport/#download)
  - [Habitat-Matterport-3D](https://github.com/matterport/habitat-matterport-3dresearch)

  For InteriorGS, there is an additional set of ground-truth map data for evaluation metrics, including occ map, object labels, room polygons, and annotated roadmaps.
  InteriorGS, MP3D, and HM3D all include a set of fixed exploration starting points.
  These can be downloaded from HuggingFace (**[comming soon]**).

### Data Structure Example

```text
assets_data/
	interiorgs/
		0001_839920/
			gs/
				floor_0_block_0/
					point_cloud_rotated.ply
					polygon_f0_0.txt
					polygon_z_f0.txt
		0002_839955/
		...
	mp3d/
		v1/
			tasks/
				mp3d/
					1LXtFkjw3qL/
						1LXtFkjw3qL_semantic.ply
						1LXtFkjw3qL.glb
						1LXtFkjw3qL.house
						1LXtFkjw3qL.navmesh
					...
	hm3d/
		00009-vLpv2VX547B/
			vLpv2VX547B.basis.glb
			vLpv2VX547B.basis.navmesh
		...

map_data/
	interiorgs/
		0001_839920/
			link_map.json
			occ_map_height.tiff
			occ_map_meta.txt
			occ_map.png
			structure.json
		0002_839955/
		...
	hm3d/
		00009-vLpv2VX547B/
			map/
				building.json
				floor_0/
					link_map.json
					occ_map_height.tiff
					occ_map_meta.txt
					occ_map.png
				floor_1/
				...
		...

start_pose_data/
	interiorgs/
		val_seen/
			0010_840154/
				node_0010_840154_0.json
				node_0010_840154_1.json
				...
			...
		val_unseen/
	mp3d/
		start_episodes.json
		start_episodes.json.gz
	hm3d/
		start_episodes.json
		start_episodes.json.gz
```

---

## 4. Installation

### Setup Conda Environment

Create and activate the conda environment, then install the necessary dependencies:

```bash
# Create environment with Python 3.10
conda create -n abot_explorer python=3.10
conda activate abot_explorer

# Install main requirements
pip install -r requirements.txt

# Install the explore_agent package in editable mode
cd explore_agent
pip install -e .
```

---

## 5. Evaluation Environment Setup

Evaluation requires two separate components: Habitat and 3DGS (PGSR). Please set up these environments individually as they have distinct dependency requirements.

### Habitat Evaluation (v0.1.7)

Follow the official installation guides for Habitat Lab and Habitat Sim version 0.1.7:

- [Habitat Lab v0.1.7](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7)
- [Habitat Sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7)

### 3DGS Evaluation (PGSR)

We use PGSR for rendering in 3DGS evaluations. Please refer to the official repository for installation instructions:

- [PGSR Repository](https://github.com/zju3dv/PGSR)

---

## 6. Inference on 3DGS Environment

This section details how to run inference using 3D Gaussian Splatting scenes rendered via PGSR.

### Configuration

Edit `config/infer_interiorgs.yaml` to set your paths. Example configuration:

```yaml
model_path: /home/User/ckpt/qwen2_5_vl_32000
map_dir: /home/User/map_data/interiorgs
result_dir: /home/User/results/interiorgs
```

### Start Rendering Service

Before running inference, you must start the rendering service. This service exposes a fixed IP and port to render images for the specific scene:

```bash
cd gaussian_splatting_sim
python render_sim.py --load_ply_root /{3DGS_ASSETS_DIR}/{SCENE_ID}/gs --gpus 0
```

Example:

```bash
python render_sim.py --load_ply_root /home/Users/assets_data/interiorgs/0726_841575/gs --gpus 0
```

**Note**: Keep this terminal running while performing inference. The program will start a service that listens for image rendering requests.

### Run Inference

Once the rendering service is active, run the inference script from the explore_agent directory in a new terminal:

```bash
cd explore_agent
CUDA_VISIBLE_DEVICES=0 python scripts/infer_close_3dgs_singal.py -c config/infer_interiorgs.yaml -s {SCENE_ID}
```

Replace `{SCENE_ID}` with your specific scene identifier.

---

## 7. Inference on Habitat Environment

This section details how to run inference within the Habitat simulation environment.

### Configuration

You need to configure two YAML files located in the config directory:

#### config/infer_habitat.yaml

Set the model paths and result directories:

```yaml
model_path: /home/User/ckpt/qwen2_5_vl_32000
result_dir: /home/User/results/interiorgs
graph_path: /home/User/abot_explorer/explore_agent/habitat_utils/data/connectivity_graphs.pkl
```

#### config/habitat_ext.yaml

Set the data paths for Habitat maps and scenes:

```yaml
DATA_PATH: /home/User/start_pose_data/mp3d/start_episodes.json.gz
SCENES_DIR: /home/User/assets_data/mp3d/v1/tasks/
```

### Run Inference

Execute the Habitat inference script:

```bash
cd explore_agent
CUDA_VISIBLE_DEVICES=0 python scripts/infer_close_habitat.py -c config/infer_habitat.yaml
```

---
