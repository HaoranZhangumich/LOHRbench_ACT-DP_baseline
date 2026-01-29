#!/usr/bin/env python3
"""
ACT Training for LoHRbench - Multi-task with CLIP Language Conditioning
=======================================================================

This version fixes:
✅ Progress bar always visible (tqdm -> stdout, dynamic_ncols, no print stomping)
✅ Logging doesn't hide the bar (uses tqdm.write)
✅ W&B logs correctly (wandb.log with step=cur_iter)
✅ Output dir configurable (--out-dir); checkpoints saved to:
   {out_dir}/runs/{run_name}/checkpoints/
✅ Dataset discovery prefers '*merged_success_filtered.h5'
✅ Enforces action_dim == 8 (strict)

Example:
python train_lohrbench.py \
  --data-root /data1/LoHRbench \
  --out-dir /data/haoran/projects \
  --total-iters 100000 \
  --batch-size 128 \
  --save-freq 5000 \
  --log-freq 1000 \
  --track \
  --wandb_project_name LoHRbench \
  --wandb_entity haoranwh
"""

ALGO_NAME = "BC_ACT_lohrbench"

import os
import glob
import sys
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from diffusers.training_utils import EMAModel
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from act.utils import IterationBasedBatchSampler, worker_init_fn

import tyro


# ── Task instructions (single source of truth) ──────────────────────────────

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

PD_TOKEN = "pd_joint_pos"
FALLBACK_ALLOW_DIMS = {7, 8}

# Your requirement:
OUTPUT_ACTION_DIM = 8


# ── Args ─────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # W&B
    track: bool = False
    wandb_project_name: str = "LoHRbench"
    wandb_entity: Optional[str] = None

    # Output
    out_dir: str = "/data/haoran/projects"

    # Data
    data_root: str = "/data1/LoHRbench"
    use_filtered_success: bool = True
    num_traj: Optional[int] = None
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_local_files_only: bool = False
    """If True: don't try to download from HuggingFace; only use local cache"""

    # Training
    total_iters: int = 500_000
    batch_size: int = 128
    lr: float = 1e-4
    kl_weight: float = 10

    # Horizon
    num_queries: int = 30

    # Backbone
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False

    # Transformer
    enc_layers: int = 4
    dec_layers: int = 8
    dim_feedforward: int = 1024
    hidden_dim: int = 512
    dropout: float = 0.1
    nheads: int = 16
    pre_norm: bool = False

    # Logging & checkpointing
    log_freq: int = 1000
    save_freq: int = 50_000
    num_dataload_workers: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_action_dim(h5_path: str) -> Optional[int]:
    try:
        with h5py.File(h5_path, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            if not traj_keys:
                return None
            return int(f[traj_keys[0]]["actions"].shape[1])
    except Exception:
        return None


def discover_h5_files(root: str, use_filtered_success: bool = True) -> List[Tuple[str, str, str]]:
    """
    Discover HDF5 files.

    Rule:
    - Prefer files ending with '*merged_success_filtered.h5' (your requirement).
    - If none exist, fallback to old discovery.
    """
    results: List[Tuple[str, str, str]] = []
    suffix = "merged_success_filtered.h5"

    for ttype in TASK_TYPES:
        ttype_dir = os.path.join(root, ttype)
        if not os.path.isdir(ttype_dir):
            continue

        for task_name in sorted(os.listdir(ttype_dir)):
            task_dir = os.path.join(ttype_dir, task_name)
            if not os.path.isdir(task_dir):
                continue

            search_dir = os.path.join(task_dir, "filtered_success") if use_filtered_success else task_dir
            if not os.path.isdir(search_dir):
                continue

            # Preferred files
            preferred = sorted(glob.glob(os.path.join(search_dir, "**", f"*{suffix}"), recursive=True))
            if preferred:
                for fp in preferred:
                    results.append((ttype, task_name, fp))
                continue

            # Fallback discovery
            files = sorted(
                set(
                    glob.glob(os.path.join(search_dir, "**", "*.[hH][dD][fF]5"), recursive=True)
                    + glob.glob(os.path.join(search_dir, "**", "*.[hH]5"), recursive=True)
                )
            )
            if not files:
                continue

            with_token = [fp for fp in files if PD_TOKEN in os.path.basename(fp)]
            if with_token:
                for fp in with_token:
                    results.append((ttype, task_name, fp))
                continue

            for fp in files:
                dim = _get_action_dim(fp)
                if dim in FALLBACK_ALLOW_DIMS:
                    results.append((ttype, task_name, fp))

    return results


# ── Dataset ──────────────────────────────────────────────────────────────────

class LoHRbenchDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        num_queries: int,
        clip_model_name: str,
        clip_local_files_only: bool,
        use_filtered_success: bool = True,
        num_traj_per_task: Optional[int] = None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])

        file_list = discover_h5_files(data_root, use_filtered_success)
        if not file_list:
            raise FileNotFoundError(f"No HDF5 files found under {data_root}")

        tqdm.write(f"Discovered {len(file_list)} HDF5 files across tasks:")
        for ttype in TASK_TYPES:
            count = sum(1 for t, _, _ in file_list if t == ttype)
            if count > 0:
                tqdm.write(f"  {ttype}: {count} files")

        self.clip_dim = self._precompute_clip_embeddings(
            clip_model_name=clip_model_name,
            local_files_only=clip_local_files_only,
        )

        self.traj_index = []
        self.traj_states = []
        self.traj_actions = []
        self.slices = []

        traj_count_per_task = defaultdict(int)
        for _, task_name, h5_path in file_list:
            if num_traj_per_task is not None and traj_count_per_task[task_name] >= num_traj_per_task:
                continue
            loaded = self._index_h5(
                h5_path=h5_path,
                task_name=task_name,
                num_traj_limit=num_traj_per_task,
                traj_count=traj_count_per_task,
            )
            traj_count_per_task[task_name] += loaded

        tqdm.write(f"\nIndexed {len(self.traj_index)} trajectories, {len(self.slices)} samples")
        for task_name, count in sorted(traj_count_per_task.items()):
            tqdm.write(f"  {task_name}: {count} trajectories")

        # Note: this is in-memory since we already loaded state/actions to compute it
        self.norm_stats = self._compute_norm_stats()

    def _precompute_clip_embeddings(self, clip_model_name: str, local_files_only: bool) -> int:
        """
        Safetensors-first load to avoid torch.load (transformers blocks torch.load unless torch>=2.6).
        """
        from transformers import CLIPModel, CLIPTokenizer

        tqdm.write(f"Loading CLIP model: {clip_model_name} (local_files_only={local_files_only})")
        last_err = None

        # Attempt 1: safetensors-only
        try:
            clip_model = CLIPModel.from_pretrained(
                clip_model_name,
                use_safetensors=True,
                local_files_only=local_files_only,
            )
            clip_tokenizer = CLIPTokenizer.from_pretrained(
                clip_model_name,
                local_files_only=local_files_only,
            )
            return self._compute_clip_embeds(clip_model, clip_tokenizer)
        except Exception as e:
            last_err = e
            tqdm.write(f"⚠️ CLIP safetensors load failed: {repr(e)}")

        # Attempt 2: allow non-safetensors (requires torch>=2.6)
        try:
            clip_model = CLIPModel.from_pretrained(
                clip_model_name,
                use_safetensors=False,
                local_files_only=local_files_only,
            )
            clip_tokenizer = CLIPTokenizer.from_pretrained(
                clip_model_name,
                local_files_only=local_files_only,
            )
            return self._compute_clip_embeds(clip_model, clip_tokenizer)
        except Exception as e:
            raise RuntimeError(
                f"Can't load CLIP '{clip_model_name}'.\n\n"
                f"Most likely causes:\n"
                f"1) transformers blocks torch.load unless torch>=2.6. Fix:\n"
                f"   pip install -U --index-url https://download.pytorch.org/whl/cu121 "
                f"torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0\n"
                f"2) Offline + no safetensors cached. Fix: run once online OR download snapshot locally.\n"
                f"3) Local folder named '{clip_model_name}' shadows HF.\n\n"
                f"First error (safetensors attempt): {repr(last_err)}\n"
                f"Second error (fallback attempt): {repr(e)}\n"
            )

    def _compute_clip_embeds(self, clip_model, clip_tokenizer) -> int:
        clip_model.eval()
        self.task_embeddings = {}

        with torch.no_grad():
            for task_name, instruction in TASK_INSTRUCTIONS.items():
                inputs = clip_tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
                text_outputs = clip_model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                text_features = clip_model.text_projection(text_outputs.pooler_output)
                text_features = F.normalize(text_features, dim=-1)
                self.task_embeddings[task_name] = text_features.squeeze(0).cpu()

        clip_dim = next(iter(self.task_embeddings.values())).shape[0]
        tqdm.write(f"CLIP embeddings computed: {len(self.task_embeddings)} tasks, dim={clip_dim}")
        del clip_model, clip_tokenizer
        return clip_dim

    def _index_h5(self, h5_path: str, task_name: str, num_traj_limit: Optional[int], traj_count: dict) -> int:
        loaded = 0
        try:
            with h5py.File(h5_path, "r") as f:
                traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
                for traj_key in traj_keys:
                    if num_traj_limit is not None and traj_count[task_name] + loaded >= num_traj_limit:
                        break

                    g = f[traj_key]
                    try:
                        qpos = g["obs"]["agent"]["qpos"][:]  # (T, 9)
                        actions = g["actions"][:]           # (T-1, act_dim)
                    except KeyError:
                        continue

                    Tlen = qpos.shape[0]
                    if actions.shape[0] != Tlen - 1:
                        continue

                    act_dim = actions.shape[1]
                    if act_dim != OUTPUT_ACTION_DIM:
                        # strict: only accept dim 8
                        continue

                    episode_len = Tlen - 1
                    state = torch.from_numpy(qpos[:-1].astype(np.float32))     # (T-1, 9)
                    actions_t = torch.from_numpy(actions.astype(np.float32))   # (T-1, 8)

                    traj_idx = len(self.traj_index)
                    self.traj_index.append((h5_path, traj_key, episode_len, task_name))
                    self.traj_states.append(state)
                    self.traj_actions.append(actions_t)
                    self.slices += [(traj_idx, ts) for ts in range(episode_len)]
                    loaded += 1

        except Exception as e:
            tqdm.write(f"Error indexing {h5_path}: {e}")

        return loaded

    def _compute_norm_stats(self):
        all_states = torch.cat(self.traj_states, dim=0)
        all_actions = torch.cat(self.traj_actions, dim=0)

        state_mean = all_states.mean(dim=0, keepdim=True)
        state_std = all_states.std(dim=0, keepdim=True).clamp(min=1e-2)
        action_mean = all_actions.mean(dim=0, keepdim=True)
        action_std = all_actions.std(dim=0, keepdim=True).clamp(min=1e-2)

        return {
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
        }

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]
        h5_path, traj_key, _, task_name = self.traj_index[traj_idx]

        state = self.traj_states[traj_idx][ts]

        with h5py.File(h5_path, "r") as f:
            g = f[traj_key]
            base_rgb = g["obs"]["sensor_data"]["base_camera"]["rgb"][ts]
            hand_rgb = g["obs"]["sensor_data"]["hand_camera"]["rgb"][ts]

        base_rgb_t = self.transforms(torch.from_numpy(base_rgb).permute(2, 0, 1))
        hand_rgb_t = self.transforms(torch.from_numpy(hand_rgb).permute(2, 0, 1))
        rgb = torch.stack([base_rgb_t, hand_rgb_t], dim=0)  # (2, 3, 224, 224)

        actions = self.traj_actions[traj_idx]  # (T-1, 8)
        act_seq = actions[ts : ts + self.num_queries]
        if act_seq.shape[0] < self.num_queries:
            target = act_seq[-1]
            act_seq = torch.cat(
                [act_seq, target.unsqueeze(0).repeat(self.num_queries - act_seq.shape[0], 1)],
                dim=0,
            )

        state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
        act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        lang_embed = self.task_embeddings[task_name]
        return {"observations": {"state": state, "rgb": rgb}, "actions": act_seq, "lang_embed": lang_embed}


# ── Agent ────────────────────────────────────────────────────────────────────

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


class Agent(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, args, lang_embed_dim: int):
        super().__init__()
        self.kl_weight = args.kl_weight
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        backbone = build_backbone(args)
        transformer = build_transformer(args)
        encoder = build_encoder(args)

        self.model = DETRVAE(
            [backbone],
            transformer,
            encoder,
            state_dim=state_dim,
            action_dim=act_dim,
            num_queries=args.num_queries,
            lang_embed_dim=lang_embed_dim,
        )

    def compute_loss(self, obs, action_seq, lang_embed=None):
        obs["rgb"] = obs["rgb"].float() / 255.0
        obs["rgb"] = self.normalize(obs["rgb"])
        a_hat, (mu, logvar) = self.model(obs, action_seq, lang_embed=lang_embed)
        total_kld, _, _ = kl_divergence(mu, logvar)
        l1 = F.l1_loss(action_seq, a_hat, reduction="mean")
        loss = l1 + total_kld[0] * self.kl_weight
        return {"l1": l1, "kl": total_kld[0], "loss": loss}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[:-len(".py")]
        run_name = f"lohrbench__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_dir = os.path.join(args.out_dir, "runs", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    tqdm.write("=" * 80)
    tqdm.write("Loading LoHRbench dataset...")
    tqdm.write("=" * 80)

    dataset = LoHRbenchDataset(
        data_root=args.data_root,
        num_queries=args.num_queries,
        clip_model_name=args.clip_model_name,
        clip_local_files_only=args.clip_local_files_only,
        use_filtered_success=args.use_filtered_success,
        num_traj_per_task=args.num_traj,
    )

    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)

    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )

    wandb_run = None
    if args.track:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            group="ACT_LoHRbench",
            tags=["act", "lohrbench", "language"],
        )

    writer = SummaryWriter(run_dir)

    state_dim = 9
    act_dim = OUTPUT_ACTION_DIM
    agent = Agent(state_dim, act_dim, args, lang_embed_dim=dataset.clip_dim).to(device)

    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(state_dim, act_dim, args, lang_embed_dim=dataset.clip_dim).to(device)

    param_dicts = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    lr_drop = int((2 / 3) * args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    def save_ckpt(tag: str):
        ema.copy_to(ema_agent.parameters())
        ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(
            {
                "norm_stats": dataset.norm_stats,
                "agent": agent.state_dict(),
                "ema_agent": ema_agent.state_dict(),
                "clip_model_name": args.clip_model_name,
                "task_instructions": TASK_INSTRUCTIONS,
                "args": vars(args),
            },
            ckpt_path,
        )
        tqdm.write(f"✅ Saved checkpoint: {ckpt_path}")

    tqdm.write("\n" + "=" * 80)
    tqdm.write(f"Training for {args.total_iters} iterations")
    tqdm.write(f"  output dir: {run_dir}")
    tqdm.write(f"  checkpoints: {ckpt_dir}")
    tqdm.write("=" * 80 + "\n")

    # ✅ Always-visible progress bar:
    pbar = tqdm(
        total=args.total_iters,
        desc="train",
        dynamic_ncols=True,
        file=sys.stdout,     # <- force stdout
        mininterval=0.5,
        leave=True,
    )

    agent.train()
    timings = defaultdict(float)

    for cur_iter, data_batch in enumerate(train_dataloader):
        # IterationBasedBatchSampler should stop at total_iters, but be extra safe:
        if cur_iter >= args.total_iters:
            break

        tick = time.time()

        obs_batch = {k: v.to(device, non_blocking=True) for k, v in data_batch["observations"].items()}
        act_batch = data_batch["actions"].to(device, non_blocking=True)
        lang_batch = data_batch["lang_embed"].to(device, non_blocking=True)

        loss_dict = agent.compute_loss(obs=obs_batch, action_seq=act_batch, lang_embed=lang_batch)
        total_loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())

        timings["update"] += time.time() - tick

        # ✅ Update progress bar EVERY iter
        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            l1=f"{loss_dict['l1'].item():.4f}",
            kl=f"{loss_dict['kl'].item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        pbar.update(1)

        # Logging
        if cur_iter % args.log_freq == 0:
            tqdm.write(
                f"Iter {cur_iter}/{args.total_iters} | loss: {total_loss.item():.4f} | "
                f"l1: {loss_dict['l1'].item():.4f} | kl: {loss_dict['kl'].item():.4f}"
            )
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("charts/backbone_learning_rate", optimizer.param_groups[1]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            writer.add_scalar("losses/l1", loss_dict["l1"].item(), cur_iter)
            writer.add_scalar("losses/kl", loss_dict["kl"].item(), cur_iter)
            writer.add_scalar("time/update", timings["update"], cur_iter)

            if wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "loss/total": total_loss.item(),
                        "loss/l1": loss_dict["l1"].item(),
                        "loss/kl": loss_dict["kl"].item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "lr_backbone": optimizer.param_groups[1]["lr"],
                    },
                    step=cur_iter,  # ✅ ensures plots are correct
                )

        # Checkpointing
        if cur_iter > 0 and cur_iter % args.save_freq == 0:
            save_ckpt(f"iter_{cur_iter}")

    # Final checkpoint
    save_ckpt("final")
    pbar.close()
    writer.close()
    tqdm.write("\nTraining complete!")
