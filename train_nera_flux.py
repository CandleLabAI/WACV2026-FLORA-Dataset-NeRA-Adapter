from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)
from PIL import Image
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
import yaml

from kan import NERAKANConfig, NERAKANAdapter, add_nera_kan_to_model, save_nera_kan_state

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# Prompt encoding helpers
# -------------------------------------------------------------------------
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer=None,
    prompt=None,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = text_input_ids.shape[0]

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer=None,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = text_input_ids.shape[0]

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class FashionDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path, tokenizer1, tokenizer2, vae_scale_factor: int):
        self.image_root = image_root
        self.df = pd.read_csv(csv_path)
        self.input_image_preprocessor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["img_paths"]
        image = Image.open(image_path).convert("RGB").resize((1024, 1024))
        image = self.input_image_preprocessor.preprocess(image)

        prompt = row["Input Prompt"]

        text_ids1 = self.tokenizer1(
            prompt,
            padding="max_length",
            max_length=self.tokenizer1.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_ids2 = self.tokenizer2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "x": image.squeeze(0),
            "input_ids1": text_ids1.squeeze(0),
            "input_ids2": text_ids2.squeeze(0),
        }


# -------------------------------------------------------------------------
# Config / setup
# -------------------------------------------------------------------------
def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_models(cfg: Dict[str, Any]):
    pretrained = cfg["pretrained_model_name_or_path"]
    cache_dir = cfg.get("cache_dir")

    text_encoder1 = CLIPTextModel.from_pretrained(
        pretrained, subfolder="text_encoder", cache_dir=cache_dir
    )
    text_encoder2 = T5EncoderModel.from_pretrained(
        pretrained, subfolder="text_encoder_2", cache_dir=cache_dir
    )
    tokenizer1 = CLIPTokenizer.from_pretrained(
        pretrained, subfolder="tokenizer", cache_dir=cache_dir
    )
    tokenizer2 = T5Tokenizer.from_pretrained(
        pretrained, subfolder="tokenizer_2", cache_dir=cache_dir
    )
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", cache_dir=cache_dir)

    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained, subfolder="transformer", cache_dir=cache_dir
    )

    nera_cfg_dict = cfg.get("nera", {})
    nera_config = NERAKANConfig(
        r=nera_cfg_dict.get("r", 16),
        num_knots=nera_cfg_dict.get("num_knots", 8),
        alpha=nera_cfg_dict.get("alpha", 1e-3),
        target_modules=tuple(nera_cfg_dict.get("target_modules", ("to_k", "to_out.0", "to_v", "to_q"))),
    )
    add_nera_kan_to_model(transformer, nera_config)

    # Freeze everything then unfreeze only NeRA-KAN adapter parameters.
    for p in transformer.parameters():
        p.requires_grad = False
    for _, module in transformer.named_modules():
        if isinstance(module, NERAKANAdapter):
            module.enable_training()

    return transformer, vae, text_encoder1, text_encoder2, tokenizer1, tokenizer2, nera_config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_weight_dtype(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
def train(cfg_path: Path):
    cfg = load_config(cfg_path)

    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    save_every = cfg.get("save_every", 5)
    adapter_dir = Path(cfg.get("nera", {}).get("adapter_dir", "./nera_adapter"))
    adapter_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("grad_accum", 8),
        mixed_precision=train_cfg.get("mixed_precision", "fp16"),
        log_with=train_cfg.get("log_with", "wandb"),
    )
    set_seed(train_cfg.get("seed", 42))

    (
        transformer,
        vae,
        text_encoder1,
        text_encoder2,
        tokenizer1,
        tokenizer2,
        nera_config,
    ) = build_models(cfg)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Build dataset/dataloader
    dataset = FashionDataset(
        csv_path=Path(data_cfg["csv_path"]),
        image_root=Path(data_cfg["image_root"]),
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        vae_scale_factor=vae_scale_factor,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )

    transformer_trainable = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    print(f"Trainable parameters (NeRA-KAN): {count_parameters(transformer):,}")

    vae.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        transformer_trainable,
        lr=train_cfg.get("lr", 1e-5),
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        subfolder="scheduler",
        cache_dir=cfg.get("cache_dir"),
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    epochs = train_cfg.get("epochs", 20)
    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=train_cfg.get("warmup_steps", 100),
        num_training_steps=num_training_steps,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = get_weight_dtype(accelerator)

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder1.to(accelerator.device, dtype=weight_dtype)
    text_encoder2.to(accelerator.device, dtype=weight_dtype)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                latent_input = batch["x"].to(device=accelerator.device, dtype=weight_dtype)
                bsz, _, lat_h, lat_w = latent_input.shape

                latents = vae.encode(latent_input).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    latents.shape[0],
                    latents.shape[2],
                    latents.shape[3],
                    accelerator.device,
                    weight_dtype,
                )

                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )

                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_latents,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )

                pooled_prompt_embeds = _encode_prompt_with_clip(
                    text_encoder=text_encoder1,
                    device=accelerator.device,
                    text_input_ids=batch["input_ids1"],
                )

                prompt_embeds = _encode_prompt_with_t5(
                    text_encoder=text_encoder2,
                    device=accelerator.device,
                    text_input_ids=batch["input_ids2"],
                )

                text_ids = torch.zeros(
                    bsz, prompt_embeds.shape[1], 3, device=accelerator.device, dtype=weight_dtype
                )

                guidance = torch.tensor([3.5], device=accelerator.device)
                guidance = guidance.expand(bsz)

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(lat_h * vae_scale_factor),
                    width=int(lat_w * vae_scale_factor),
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)

                target = noise - latents

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), max_norm=5.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {
                "epoch": epoch,
                "step": step,
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            print(logs)

        if (epoch + 1) % save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model_to_save = accelerator.unwrap_model(transformer)
                save_nera_kan_state(model_to_save, adapter_dir / "adapter.pt", cfg=nera_config)

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Train FLUX with NERA adapters.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
