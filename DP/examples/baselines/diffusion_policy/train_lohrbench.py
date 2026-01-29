#!/usr/bin/env python3
"""
LoHRbench Diffusion Policy (UNet) Training - Multi-task + CLIP Language Conditioning
===================================================================================

Fixes:
- STRICT dataset discovery: ONLY '*merged_success_filtered.h5'
- Pads action windows to pred_horizon (fixes collate stack shape mismatch)
- tqdm progress bar always visible; training prints use tqdm.write()
- wandb logging uses step=it to align curves
- caches CLIP embeddings + norm stats

Run:
python train_lohrbench.py \
  --data-root /data1/LoHRbench \
  --run-root /data/haoran/projects \
  --total-iters 100000 \
  --batch-size 256 \
  --obs-horizon 2 \
  --act-horizon 8 \
  --pred-horizon 16 \
  --save-freq 2000 \
  --log-freq 200 \
  --track \
  --wandb_project_name LoHRbench \
  --wandb_entity haoranwh
"""

from __future__ import annotations

import os
import glob
import json
import sys
import time
import random
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn


# -----------------------------------------------------------------------------
# Task instructions (single source of truth)
# -----------------------------------------------------------------------------

TASK_INSTRUCTIONS = {
    "reverse_stack": "reverse 10 stacked cube in reverse order",
    "stack_10_cube": "stack 10 cube together, start with red cube",
    "stack_cube_clutter": "stack 3 cube together , start with red cube",
    "cluttered_packing": "put three cube in to the bowl",
    "pick_active_exploration": "pick up the can, screwdriver and cup out of the drawer",
    "stack_active_exploration": "pick up the cube and stack them together, start with red cube",
    "fruit_placement": "place four starberries into the target position",
    "repackage": "put cube into the bowl and stack the bowl on the plate",
}

TASK_TYPES = ["active_exploration", "clutter", "super_long_horizon", "tool_using"]

# STRICT requirement
REQUIRED_H5_SUFFIX = "merged_success_filtered.h5"

STATE_DIM = 9
ACTION_DIM = 8
RGB_SIZE = (224, 224)


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # Logging
    track: bool = False
    wandb_project_name: str = "LoHRbench"
    wandb_entity: Optional[str] = None
    log_freq: int = 200
    save_freq: int = 50_000

    # Where to write runs/checkpoints
    run_root: str = "./runs"
    run_group: str = "DiffusionPolicy_LoHRbench"
    run_tags: List[str] = field(default_factory=lambda: ["diffusion_policy", "lohrbench", "language"])

    # Data
    data_root: str = "/data1/LoHRbench"
    num_traj: Optional[int] = None  # max per task
    cache_dir: str = "./cache_lohrbench_dp"

    # CLIP
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_local_files_only: bool = False

    # Training
    total_iters: int = 500_000
    batch_size: int = 256
    lr: float = 1e-4

    # Horizons
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16

    # Diffusion
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    num_diffusion_iters: int = 100

    # Visual encoder
    visual_feature_dim: int = 256

    # Normalization
    q_low: float = 0.5
    q_high: float = 99.5


# -----------------------------------------------------------------------------
# Helpers: file discovery (STRICT)
# -----------------------------------------------------------------------------

def discover_lohrbench_h5_files(data_root: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for ttype in TASK_TYPES:
        ttype_dir = os.path.join(data_root, ttype)
        if not os.path.isdir(ttype_dir):
            continue
        for task_name in sorted(os.listdir(ttype_dir)):
            task_dir = os.path.join(ttype_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            files = sorted(glob.glob(os.path.join(task_dir, "**", f"*{REQUIRED_H5_SUFFIX}"), recursive=True))
            for fp in files:
                out.append((ttype, task_name, fp))
    return out


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# CLIP embeddings (cached) - safetensors-first
# -----------------------------------------------------------------------------

def compute_clip_text_embeddings(
    task_instructions: Dict[str, str],
    clip_model_name: str,
    cache_path: str,
    local_files_only: bool,
) -> Tuple[Dict[str, torch.Tensor], int]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.isfile(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        return payload["task_embeddings"], int(payload["clip_dim"])

    from transformers import CLIPModel, CLIPTokenizer

    tqdm.write(f"[CLIP] Loading model: {clip_model_name} (local_files_only={local_files_only})")

    last_err: Optional[Exception] = None
    clip_model = None
    clip_tokenizer = None

    for use_safetensors in [True, False]:
        try:
            clip_model = CLIPModel.from_pretrained(
                clip_model_name,
                use_safetensors=use_safetensors,
                local_files_only=local_files_only,
            )
            clip_tokenizer = CLIPTokenizer.from_pretrained(
                clip_model_name,
                local_files_only=local_files_only,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e

    if last_err is not None or clip_model is None or clip_tokenizer is None:
        raise OSError(
            f"Can't load CLIP model '{clip_model_name}'. Last error: {repr(last_err)}\n"
            "Fix: ensure safetensors exist in cache OR upgrade torch>=2.6 OR use local model dir."
        )

    clip_model.eval()
    task_embeddings: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for task_name, instruction in task_instructions.items():
            inputs = clip_tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
            text_outputs = clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_features = clip_model.text_projection(text_outputs.pooler_output)
            text_features = F.normalize(text_features, dim=-1)
            task_embeddings[task_name] = text_features.squeeze(0).cpu()

    clip_dim = int(next(iter(task_embeddings.values())).shape[0])
    torch.save({"task_embeddings": task_embeddings, "clip_dim": clip_dim}, cache_path)
    tqdm.write(f"[CLIP] Cached embeddings -> {cache_path} (dim={clip_dim}, tasks={len(task_embeddings)})")

    del clip_model, clip_tokenizer
    return task_embeddings, clip_dim


# -----------------------------------------------------------------------------
# Stats (cached) - correct + stable
# -----------------------------------------------------------------------------

def compute_or_load_stats(
    all_states: torch.Tensor,
    all_actions: torch.Tensor,
    cache_path: str,
    q_low: float,
    q_high: float,
) -> Dict[str, torch.Tensor]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        return torch.load(cache_path, map_location="cpu")

    tqdm.write("[STATS] Computing normalization stats (first run)...")

    state_mean = all_states.mean(dim=0, keepdim=True)
    state_std = all_states.std(dim=0, keepdim=True).clamp(min=1e-2)

    a_np = all_actions.cpu().numpy()
    a_low = np.percentile(a_np, q_low, axis=0).astype(np.float32)
    a_high = np.percentile(a_np, q_high, axis=0).astype(np.float32)

    eps = 1e-6
    a_mid = (a_low + a_high) / 2.0
    a_half = (a_high - a_low) / 2.0
    a_half[a_half < eps] = 1.0

    stats = {
        "state_mean": state_mean.cpu(),
        "state_std": state_std.cpu(),
        "action_low": torch.from_numpy(a_low),
        "action_high": torch.from_numpy(a_high),
        "action_mid": torch.from_numpy(a_mid),
        "action_half": torch.from_numpy(a_half),
        "q_low": torch.tensor([q_low], dtype=torch.float32),
        "q_high": torch.tensor([q_high], dtype=torch.float32),
    }
    torch.save(stats, cache_path)
    tqdm.write(f"[STATS] Cached stats -> {cache_path}")
    return stats


def action_to_unit_range(action: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mid = stats["action_mid"].to(action.device)
    half = stats["action_half"].to(action.device)
    return ((action - mid) / half).clamp(-1.0, 1.0)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class LoHRbenchDiffusionDataset(Dataset):
    """
    Returns fixed shapes:
      rgb:   (obs_horizon, 6, 224, 224)
      state: (obs_horizon, 9)
      lang:  (clip_dim,)
      actions: (pred_horizon, 8)  ALWAYS padded to pred_horizon
    """

    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        os.makedirs(args.cache_dir, exist_ok=True)

        file_list = discover_lohrbench_h5_files(args.data_root)
        if not file_list:
            raise FileNotFoundError(
                f"No LoHRbench files found under {args.data_root} ending with '{REQUIRED_H5_SUFFIX}'"
            )

        tqdm.write(f"[DATA] Found {len(file_list)} HDF5 files (suffix={REQUIRED_H5_SUFFIX})")
        for ttype in TASK_TYPES:
            c = sum(1 for t, _, _ in file_list if t == ttype)
            if c:
                tqdm.write(f"  {ttype}: {c} files")

        # CLIP cache
        clip_cache = os.path.join(
            args.cache_dir,
            f"clip_{_hash_str(args.clip_model_name + json.dumps(TASK_INSTRUCTIONS, sort_keys=True))}.pt"
        )
        self.task_embeddings, self.clip_dim = compute_clip_text_embeddings(
            TASK_INSTRUCTIONS,
            args.clip_model_name,
            clip_cache,
            local_files_only=args.clip_local_files_only,
        )

        self.resize = T.Resize(RGB_SIZE, antialias=True)

        self.trajs: List[Tuple[str, str, int, str]] = []
        self.states: List[torch.Tensor] = []   # (T,9)
        self.actions: List[torch.Tensor] = []  # (T-1,8)
        self.slices: List[Tuple[int, int]] = []  # (traj_idx, t)

        traj_count_per_task = defaultdict(int)

        tqdm.write("[DATA] Indexing trajectories (states/actions only)...")
        pbar = tqdm(total=len(file_list), desc="index_h5", dynamic_ncols=True, file=sys.stdout)
        for _, task_name, h5_path in file_list:
            with h5py.File(h5_path, "r") as f:
                traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
                for traj_key in traj_keys:
                    if args.num_traj is not None and traj_count_per_task[task_name] >= args.num_traj:
                        break

                    g = f[traj_key]
                    try:
                        qpos = g["obs"]["agent"]["qpos"][:]   # (T,9)
                        acts = g["actions"][:]                # (T-1,8?)
                    except KeyError:
                        continue

                    if qpos.ndim != 2 or qpos.shape[1] != STATE_DIM:
                        continue
                    if acts.ndim != 2:
                        continue

                    if acts.shape[1] == ACTION_DIM:
                        pass
                    elif acts.shape[1] > ACTION_DIM:
                        acts = acts[:, :ACTION_DIM]
                    else:
                        continue

                    Tlen = qpos.shape[0]
                    if acts.shape[0] != Tlen - 1:
                        continue
                    L = Tlen - 1

                    # We allow shorter than pred_horizon because we will PAD in __getitem__
                    # But must be at least obs_horizon
                    if L < args.obs_horizon:
                        continue

                    traj_idx = len(self.trajs)
                    self.trajs.append((h5_path, traj_key, L, task_name))
                    self.states.append(torch.from_numpy(qpos.astype(np.float32)))
                    self.actions.append(torch.from_numpy(acts.astype(np.float32)))
                    traj_count_per_task[task_name] += 1

                    # IMPORTANT: t range only needs to ensure obs window exists.
                    # Actions will be padded if reaching end.
                    t_min = args.obs_horizon - 1
                    t_max = L - 1  # last valid action index is L-1; we may pad beyond end
                    for t in range(t_min, t_max + 1):
                        self.slices.append((traj_idx, t))
            pbar.update(1)
        pbar.close()

        tqdm.write(f"[DATA] Loaded trajectories: {len(self.trajs)}")
        for task_name, c in sorted(traj_count_per_task.items()):
            tqdm.write(f"  {task_name}: {c} trajectories")
        tqdm.write(f"[DATA] Total training samples: {len(self.slices)}")

        # Stats from all available states/actions (correct)
        all_states = torch.cat([s[:-1] for s in self.states], dim=0)  # (sum(T-1), 9)
        all_actions = torch.cat(self.actions, dim=0)                  # (sum(T-1), 8)

        stats_cache = os.path.join(
            args.cache_dir,
            f"stats_{_hash_str(args.data_root + str(args.num_traj) + REQUIRED_H5_SUFFIX + f'{args.q_low}_{args.q_high}')}.pt"
        )
        self.stats = compute_or_load_stats(all_states, all_actions, stats_cache, args.q_low, args.q_high)
        del all_states, all_actions

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int):
        traj_idx, t = self.slices[idx]
        h5_path, traj_key, L, task_name = self.trajs[traj_idx]

        obs_start = t - (self.args.obs_horizon - 1)
        obs_end = t + 1
        assert obs_start >= 0

        state_seq = self.states[traj_idx][obs_start:obs_end]  # (obs_horizon, 9)

        # Action window may run past end -> pad
        act_start = t
        act_end = t + self.args.pred_horizon
        act_seq = self.actions[traj_idx][act_start:min(act_end, L)]  # (<=pred_horizon, 8)
        need = self.args.pred_horizon - act_seq.shape[0]
        if need > 0:
            # repeat last available action
            if act_seq.shape[0] > 0:
                last = act_seq[-1]
            else:
                last = self.actions[traj_idx][-1]
            act_seq = torch.cat([act_seq, last.unsqueeze(0).repeat(need, 1)], dim=0)
        # now exactly pred_horizon
        assert act_seq.shape[0] == self.args.pred_horizon and act_seq.shape[1] == ACTION_DIM

        # Load images on-the-fly (obs_horizon frames)
        with h5py.File(h5_path, "r") as f:
            g = f[traj_key]
            base_rgb = g["obs"]["sensor_data"]["base_camera"]["rgb"][obs_start:obs_end]
            hand_rgb = g["obs"]["sensor_data"]["hand_camera"]["rgb"][obs_start:obs_end]

        base_t, hand_t = [], []
        for i in range(base_rgb.shape[0]):
            b = self.resize(torch.from_numpy(base_rgb[i]).permute(2, 0, 1))
            h = self.resize(torch.from_numpy(hand_rgb[i]).permute(2, 0, 1))
            base_t.append(b)
            hand_t.append(h)

        base_t = torch.stack(base_t, dim=0)
        hand_t = torch.stack(hand_t, dim=0)
        rgb = torch.cat([base_t, hand_t], dim=1)  # (obs_horizon, 6, 224, 224)

        # Normalize
        sm = self.stats["state_mean"][0]
        ss = self.stats["state_std"][0]
        state_seq = (state_seq - sm) / ss

        act_seq = action_to_unit_range(act_seq, self.stats)

        lang = self.task_embeddings[task_name]

        return {"observations": {"rgb": rgb, "state": state_seq, "lang": lang}, "actions": act_seq}


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------

class Agent(nn.Module):
    def __init__(self, args: Args, obs_state_dim: int, lang_dim: int):
        super().__init__()
        self.args = args
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.act_dim = ACTION_DIM

        self.visual_encoder = PlainConv(
            in_channels=6,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        )

        global_cond_dim = args.obs_horizon * (args.visual_feature_dim + obs_state_dim) + lang_dim

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.img_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
        )

    def encode_obs(self, obs_seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        rgb = obs_seq["rgb"].float() / 255.0
        rgb = self.img_norm(rgb)

        B = rgb.shape[0]
        img_seq = rgb.flatten(end_dim=1)                 # (B*H, 6, 224, 224)
        visual = self.visual_encoder(img_seq)            # (B*H, D)
        visual = visual.reshape(B, self.obs_horizon, -1) # (B, H, D)

        feat = torch.cat([visual, obs_seq["state"]], dim=-1)
        feat = feat.flatten(start_dim=1)

        lang = obs_seq["lang"]
        if lang.ndim == 1:
            lang = lang.unsqueeze(0)
        return torch.cat([feat, lang], dim=-1)

    def compute_loss(self, obs_seq: Dict[str, torch.Tensor], action_seq: torch.Tensor, device: torch.device) -> torch.Tensor:
        B = action_seq.shape[0]
        obs_cond = self.encode_obs(obs_seq)

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        return F.mse_loss(noise_pred, noise)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def main():
    args = tyro.cli(Args)
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    run_name = args.exp_name or f"lohrbench_dp_lang__seed{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.run_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_run = None
    if args.track:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            group=args.run_group,
            tags=args.run_tags,
            sync_tensorboard=True,
            save_code=True,
        )

    writer = SummaryWriter(run_dir)

    tqdm.write("=" * 80)
    tqdm.write("[MAIN] Loading LoHRbench Diffusion Dataset (with language)...")
    tqdm.write("=" * 80)
    dataset = LoHRbenchDiffusionDataset(args=args)

    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=args.seed),
        pin_memory=(device.type == "cuda"),
    )

    agent = Agent(args=args, obs_state_dim=STATE_DIM, lang_dim=dataset.clip_dim).to(device)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(args=args, obs_state_dim=STATE_DIM, lang_dim=dataset.clip_dim).to(device)

    optimizer = optim.AdamW(agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    def save_ckpt(tag: str):
        ema.copy_to(ema_agent.parameters())
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(
            {
                "agent": agent.state_dict(),
                "ema_agent": ema_agent.state_dict(),
                "args": vars(args),
                "task_instructions": TASK_INSTRUCTIONS,
                "clip_model_name": args.clip_model_name,
                "norm_stats": dataset.stats,
            },
            path,
        )
        tqdm.write(f"✅ Saved checkpoint: {path}")

    tqdm.write("\n" + "=" * 80)
    tqdm.write("[TRAIN] Starting training")
    tqdm.write(f"  device={device}")
    tqdm.write(f"  run_dir={run_dir}")
    tqdm.write(f"  ckpt_dir={ckpt_dir}")
    tqdm.write(f"  obs_horizon={args.obs_horizon}, act_horizon={args.act_horizon}, pred_horizon={args.pred_horizon}")
    tqdm.write("=" * 80 + "\n")

    agent.train()
    timings = defaultdict(float)

    pbar = tqdm(
        total=args.total_iters,
        desc="train",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
        leave=True,
    )

    last_tick = time.time()
    for it, batch in enumerate(train_loader):
        if it >= args.total_iters:
            break

        timings["data_loading"] += time.time() - last_tick

        tick = time.time()
        obs = batch["observations"]
        obs = {
            "rgb": obs["rgb"].to(device, non_blocking=True),
            "state": obs["state"].to(device, non_blocking=True),
            "lang": obs["lang"].to(device, non_blocking=True),
        }
        actions = batch["actions"].to(device, non_blocking=True)
        timings["to_device"] += time.time() - tick

        tick = time.time()
        loss = agent.compute_loss(obs_seq=obs, action_seq=actions, device=device)
        timings["forward"] += time.time() - tick

        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - tick

        tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - tick

        lr_val = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_val:.2e}")
        pbar.update(1)

        if it % args.log_freq == 0:
            tqdm.write(f"Iter {it}/{args.total_iters} | loss: {loss.item():.4f} | lr: {lr_val:.2e}")

            writer.add_scalar("losses/total_loss", loss.item(), it)
            writer.add_scalar("charts/learning_rate", lr_val, it)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, it)

            if wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "loss/total": loss.item(),
                        "lr": lr_val,
                        "time/data_loading": timings["data_loading"],
                        "time/to_device": timings["to_device"],
                        "time/forward": timings["forward"],
                        "time/backward": timings["backward"],
                        "time/ema": timings["ema"],
                    },
                    step=it,
                )

        if it > 0 and (it % args.save_freq == 0):
            save_ckpt(f"iter_{it}")

        last_tick = time.time()

    save_ckpt("final")
    pbar.close()
    writer.close()
    tqdm.write("\n[TRAIN] Done.")


if __name__ == "__main__":
    main()
