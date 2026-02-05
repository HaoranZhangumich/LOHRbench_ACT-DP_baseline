# LoHRbench: Diffusion Policy & ACT Baselines

Training code for **Diffusion Policy (DP)** and **Action Chunking with Transformers (ACT)** on LoHRbench.
Built on top of [ManiSkill3](https://github.com/haosulab/ManiSkill).

GitHub: [https://github.com/HaoranZhangumich/LOHRbench_ACT-DP_baseline](https://github.com/HaoranZhangumich/LOHRbench_ACT-DP_baseline)

## Repository Structure

```
LOHRbench_ACT-DP_baseline/
├── DP/                             # Diffusion Policy
│   └── examples/baselines/
│       └── diffusion_policy/
│           ├── train_lohrbench.py  # LoHRbench training script
│           ├── train.py            # Original ManiSkill training script
│           └── diffusion_policy/
│               ├── conditional_unet1d.py   # U-Net noise prediction network
│               ├── plain_conv.py           # CNN visual encoder
│               └── evaluate.py
├── ACT/                            # Action Chunking with Transformers
│   └── examples/baselines/
│       └── act/
│           ├── train_lohrbench.py  # LoHRbench training script
│           ├── train.py            # Original ManiSkill training script
│           └── act/
│               └── detr/
│                   ├── detr_vae.py         # DETR-VAE model
│                   ├── transformer.py      # Transformer encoder/decoder
│                   ├── backbone.py         # ResNet-18 backbone
│                   └── position_encoding.py
└── README.md
```

## Setup

```bash
conda create -n <environment-name> python=3.12.12 -y
conda activate <environment-name>

# Install torch
python -m pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121

# Install DP
cd DP/examples/baselines/diffusion_policy && pip install -e . && cd ../../../..

# Install ACT
cd ACT/examples/baselines/act && pip install -e . && cd ../../../..

# Both require CLIP for language conditioning
pip install transformers
```

## Dataset

Download the demonstration dataset from HuggingFace: **[oldTOM/LoHRbench](https://huggingface.co/datasets/oldTOM/LoHRbench)**

The HDF5 files can be **directly used** for DP and ACT training without any conversion.

## Data Format

Both methods read HDF5 trajectory files from the LoHRbench data directory. The expected structure:

```
/data/LoHRbench/
├── active_exploration/
│   ├── pick_active_exploration/
│   │   └── *merged_success_filtered.h5
│   └── stack_active_exploration/
│       └── *merged_success_filtered.h5
├── clutter/
│   ├── stack_cube_clutter/
│   └── cluttered_packing/
├── super_long_horizon/
│   ├── stack_10_cube/
│   └── reverse_stack/
└── tool_using/
    ├── fruit_placement/
    └── repackage/
```

Each HDF5 file contains trajectories with:
- `traj_<i>/obs/agent/qpos`: robot joint positions (T, 9)
- `traj_<i>/obs/sensor_data/base_camera/rgb`: base camera images (T, 224, 224, 3)
- `traj_<i>/obs/sensor_data/hand_camera/rgb`: wrist camera images (T, 224, 224, 3)
- `traj_<i>/actions`: action commands (T-1, 8)

## Training

### Diffusion Policy

```bash
python DP/examples/baselines/diffusion_policy/train_lohrbench.py \
    --data-root /data1/LoHRbench \
    --run-root /path/to/checkpoints \
    --total-iters 100000 \
    --batch-size 256 \
    --obs-horizon 2 \
    --act-horizon 8 \
    --pred-horizon 16 \
    --save-freq 2000 \
    --log-freq 200 \
    --track \
    --wandb_project_name LoHRbench
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Architecture | Conditional 1D U-Net ([64, 128, 256]) |
| Visual encoder | PlainConv (6-ch input, 256-dim output) |
| Language conditioning | CLIP ViT-B/32 (512-dim) |
| Diffusion steps | 100 (DDPM) |
| Observation horizon | 2 |
| Action horizon | 8 |
| Prediction horizon | 16 |
| Batch size | 256 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (beta1=0.95, beta2=0.999, wd=1e-6) |
| LR schedule | Cosine with 500 warmup steps |
| EMA | power=0.75 |
| Action normalization | Percentile-based (q0.5-q99.5) to [-1, 1] |
| State normalization | Mean/std (min std=0.01) |
| GPU | 1x NVIDIA A100 40GB |

### ACT

```bash
python ACT/examples/baselines/act/train_lohrbench.py \
    --data-root /data1/LoHRbench \
    --out-dir /path/to/checkpoints \
    --total-iters 100000 \
    --batch-size 128 \
    --save-freq 5000 \
    --log-freq 1000 \
    --track \
    --wandb_project_name LoHRbench
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Architecture | DETR-VAE (ResNet-18 backbone) |
| Encoder layers | 4 |
| Decoder layers | 8 |
| Hidden dim | 512 |
| Attention heads | 16 |
| Feedforward dim | 1024 |
| CVAE latent dim | 32 |
| Action chunk (num_queries) | 30 |
| Language conditioning | CLIP ViT-B/32 (512-dim) |
| Batch size | 128 |
| Learning rate | 1e-4 (backbone: 1e-5) |
| Optimizer | AdamW (wd=1e-4) |
| LR schedule | StepLR (decay at 2/3 total iters, gamma=0.1) |
| KL weight | 10 |
| Loss | L1 + 10 * KL |
| EMA | power=0.75 |
| GPU | 1x NVIDIA A100 40GB |

## Evaluation

Evaluation is done via the unified evaluation framework in [`TAMPBench/baseline/`](../TAMPBench/baseline/). See the [evaluation README](../TAMPBench/baseline/README.md) for details.

```bash
# Diffusion Policy
python TAMPBench/baseline/eval.py \
    --policy dp \
    --checkpoint /path/to/iter_100000.pt \
    --benchmark-root /path/to/TAMPBench/benchmark/table-top \
    --use-action-chunking --chunk-size 8 \
    --results-dir ./results --save-video

# ACT
python TAMPBench/baseline/eval.py \
    --policy act \
    --checkpoint /path/to/iter_100000.pt \
    --benchmark-root /path/to/TAMPBench/benchmark/table-top \
    --use-action-chunking --chunk-size 8 \
    --results-dir ./results --save-video
```

## Acknowledgements

The DP and ACT implementations are adapted from the [ManiSkill3](https://github.com/haosulab/ManiSkill) baselines with added CLIP language conditioning for multi-task LoHRbench training.