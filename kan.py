# NeRA (Nonlinear low-rank Expressive Representation Adapter) adapters: apply / disable / save / load 
from __future__ import annotations
import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Callable, Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
from contextlib import contextmanager


@dataclass
class NERAKANConfig:
    r: int = 16                    # adapter feature rank
    num_knots: int = 8             # KAN spline knots
    alpha: float = 1e-3            # scaling factor (trainable inside module)
    target_modules: Tuple[str, ...] = (
        "to_q","to_v", "to_k"
    )
    
    ridge_lambda: float = 1e-5
    include_bias: bool = True      # fold a constant offset if base has bias
    device: Optional[str] = None   # None => infer from module


class NERAKANFeatureMap(nn.Module):

    def __init__(self, d_in: int, r: int, num_knots: int = 8):
        super().__init__()
        self.d_in, self.r, self.num_knots = d_in, r, num_knots
        # Learnable knot positions per input dim
        self.knots = nn.Parameter(torch.linspace(-3, 3, num_knots).repeat(d_in, 1))   # [d_in, K]
        # Per-(dim,knot) to r features
        self.coeff = nn.Parameter(torch.zeros(d_in, num_knots, r))                    # [d_in, K, r]
        # Optional pre-nonlinearity scale per input dim
        self.in_scale = nn.Parameter(torch.ones(d_in))

        nn.init.normal_(self.coeff, std=1e-3)  # near-zero init => near-null adapter

    def _tri_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Supports arbitrary leading dims, expects last dim == d_in.
        x: [..., d_in] -> basis: [..., d_in, K]
        """
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, self.d_in)
        x_flat = x_flat * self.in_scale   # [d_in] broadcast
        diffs = x_flat.unsqueeze(-1) - self.knots      # [B*, d_in, K]
        basis_flat = torch.relu(1 - diffs.abs())       # triangular basis in [-1, 1]
        return basis_flat.reshape(*prefix, self.d_in, self.num_knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., d_in] -> [..., r]
        """
        prefix = x.shape[:-1]
        basis_flat = self._tri_basis(x).reshape(-1, self.d_in, self.num_knots)
        z_flat = torch.einsum("bdk,dkr->br", basis_flat, self.coeff)  # [B*, r]
        return z_flat.reshape(*prefix, self.r)


class NERAKANAdapter(nn.Module):

    def __init__(self, base_linear: nn.Linear, cfg: NERAKANConfig):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.cfg = cfg
        d_in = base_linear.in_features
        d_out = base_linear.out_features

        self.kan = NERAKANFeatureMap(d_in, cfg.r, cfg.num_knots)
        self.readout = nn.Linear(cfg.r, d_out, bias=False)
        # scale parameter (trainable by default)
        self.alpha = nn.Parameter(torch.tensor(cfg.alpha, dtype=torch.float32))
        self.alpha.requires_grad_(False)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        dy = self.readout(self.kan(x))
        return y + self.alpha * dy

    def enable_adapter(self, enabled: bool = True):
        self.enabled = enabled

    def disable_adapter(self): 
        self.enable_adapter(False)

    def enable_training(self):
        # Only train KAN + readout; keep base and alpha frozen
        for p in self.base.parameters(): p.requires_grad = False
        for p in self.kan.parameters(): p.requires_grad = True
        for p in self.readout.parameters(): p.requires_grad = True
        self.alpha.requires_grad_(True)

# =========================
# Utilities to find/patch modules by name
# =========================
def _iter_named_linears(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            yield name, m

def add_nera_kan_to_model(model: nn.Module, cfg: NERAKANConfig) -> Dict[str, NERAKANAdapter]:
    """
    Hugging Face-friendly injector: replace target Linear modules (matching
    cfg.target_modules such as q_proj, k_proj, etc.) with NERAKANAdapter
    wrappers. Returns a dict of {full_name: adapter}.
    """
    injected: Dict[str, NERAKANAdapter] = {}
    for full_name, lin in list(_iter_named_linears(model)):
        if full_name.split(".")[-1] in cfg.target_modules:
            parent, child_name = _parent_and_attr(model, full_name)
            adapter = NERAKANAdapter(lin, cfg)
            # move adapter to same dtype/device as base
            adapter.to(lin.weight.device, dtype=lin.weight.dtype)
            setattr(parent, child_name, adapter)
            injected[full_name] = adapter
    return injected

def _parent_and_attr(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def enable_nera_kan(model: nn.Module, enabled: bool = True):
    for _, m in model.named_modules():
        if isinstance(m, NERAKANAdapter):
            m.enable_adapter(enabled)

def remove_nera_kan_from_model(model: nn.Module):
    """
    Strip adapters without merging: replace NERAKANAdapter with its base Linear.
    """
    replacements = []
    for name, m in model.named_modules():
        if isinstance(m, NERAKANAdapter):
            replacements.append(name)
    for name in replacements:
        parent, attr = _parent_and_attr(model, name)
        base = getattr(parent, attr).base
        setattr(parent, attr, base)

def save_nera_kan_state(model: nn.Module, path: str, cfg: Optional[NERAKANConfig] = None):
    """
    Save only adapter weights + config (not the base model). Also writes a
    JSON config file next to the weight file for convenience. If `cfg` is
    provided, it is used as the canonical NERAKANConfig in config.json.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    meta = {}
    for name, m in model.named_modules():
        if isinstance(m, NERAKANAdapter):
            state[f"{name}.kan"] = m.kan.state_dict()
            state[f"{name}.readout"] = m.readout.state_dict()
            state[f"{name}.alpha"] = m.alpha.detach().cpu()
            meta[name] = asdict(m.cfg)
    payload = {"state": state, "meta": meta}
    torch.save(payload, out_path)

    # Also emit JSON (tuple fields become lists for JSON compatibility).
    meta_json = {
        name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        for name, cfg in meta.items()
    }
    # Store a canonical NERAKANConfig:
    # - prefer user-supplied cfg
    # - otherwise take the first adapter's cfg if present
    # - otherwise fall back to defaults
    if cfg is not None:
        first_cfg = asdict(cfg)
    else:
        first_cfg = next(iter(meta.values()), asdict(NERAKANConfig()))
    config_json = {
        "nerakan_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in first_cfg.items()},
    }
    config_path = out_path.with_name("config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)


def load_nera_kan_config(config_path: str | Path) -> NERAKANConfig:
    """
    Load a saved NERAKANConfig from config.json (written by save_nera_kan_state).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg_dict = data.get("nerakan_config") or {}
    return _dict_to_nerakan_config(cfg_dict)


def _dict_to_nerakan_config(cfg: Dict[str, Any]) -> NERAKANConfig:
    """Convert JSON-loaded dict (lists for tuples) into NERAKANConfig."""
    fixed = dict(cfg)
    if "target_modules" in fixed and isinstance(fixed["target_modules"], list):
        fixed["target_modules"] = tuple(fixed["target_modules"])
    return NERAKANConfig(**fixed)

def load_nera_kan_state(model: nn.Module, path: str, strict: bool = True, map_location=None):
    blob = torch.load(path, map_location=map_location)
    state: Dict[str, Dict[str, torch.Tensor]] = blob["state"]
    # meta = blob.get("meta", {})  # available if needed
    loaded = 0
    for name, m in model.named_modules():
        if isinstance(m, NERAKANAdapter):
            m.kan.load_state_dict(state[f"{name}.kan"])
            m.readout.load_state_dict(state[f"{name}.readout"])
            with torch.no_grad():
                m.alpha.copy_(state[f"{name}.alpha"].to(m.alpha.device, m.alpha.dtype))
            loaded += 1
    if strict and loaded != len(state) // 3:
        raise RuntimeError(f"Loaded {loaded} adapters, file contains {len(state)//3}")
