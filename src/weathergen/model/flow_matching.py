# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Flow-matching (rectified flow) decoders in physical space, conditioned on the latent state.

For a flow-matching stream, the decoder of that stream becomes a conditional denoiser. With
normalised values at the target points, x_k (target at step k) and x_{k-1} (previous state at the
same points), the flow variable is

    y = x_k - b,   b = x_{k-1} (target: increment)  or  b = 0 (target: state)

and, at flow time t in [0, 1] (t = 0 noise, t = 1 data):

    x_t = (1 - t) * eps + t * y,      eps ~ N(0, I)

The decoder keeps predicting a STATE (x-prediction): its coordinate queries are conditioned on
(x_t, x_{k-1}, t) through a zero-initialised FlowMatchingConditioner, and

    x_hat_k = Decoder(z_k | x_t, x_{k-1}, t),    y_hat = x_hat_k - b

Training: the decoder output x_hat_k is returned as the prediction, so the existing LossPhysical
MSE against x_k is ||y_hat - y||^2 = (1 - t)^2 ||v_hat - v||^2 with v = y - eps: a valid
flow-matching loss (x-prediction). Optional `loss_weighting: v` rescales it to the velocity loss.

Sampling: Euler/Heun ODE from t = 0 to 1 with v_hat = (y_hat - x_t) / (1 - t); the last step returns
the decoder output directly (no singularity at t = 1). Members differ by their noise draws; each
member is conditioned on its own previous sample (temporal coherence).

Zero-initialised conditioning: at start the decoder ignores (x_t, x_{k-1}, t), so the model is
exactly the deterministic model it is fine-tuned from (all members = deterministic prediction).

Config (main config):

    flow_matching:
      enabled: True
      streams: [CICE]
      target: increment            # increment | state
      time_sampling: logit_normal  # logit_normal | uniform
      logit_normal: {mean: 0.0, std: 1.0}
      prev_noise_std: 0.0          # noise on x_{k-1} during training (exposure bias)
      loss_weighting: x            # x (plain MSE) | v (velocity loss, weight 1/(1-t)^2)
      max_loss_weight: 100.0       # clip of the v weighting
      inference:
        num_members: 8
        num_steps: 16
        solver: heun               # heun | euler
        noise_correlation: 0.0     # AR(1) correlation of the noise across lead times
"""

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "enabled": False,
    "streams": [],
    "target": "increment",
    "time_sampling": "logit_normal",
    "logit_normal": {"mean": 0.0, "std": 1.0},
    "prev_noise_std": 0.0,
    "loss_weighting": "x",
    "max_loss_weight": 100.0,
    "inference": {"num_members": 8, "num_steps": 16, "solver": "heun", "noise_correlation": 0.0},
}


def get_flow_matching_config(cf) -> dict:
    """Normalised view of the `flow_matching` config entry (disabled if absent)."""
    fm = cf.get("flow_matching", None) if cf is not None else None
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}
    out["logit_normal"] = dict(_DEFAULTS["logit_normal"])
    out["inference"] = dict(_DEFAULTS["inference"])
    if not fm or not fm.get("enabled", False):
        return out
    for k in ("target", "time_sampling", "prev_noise_std", "loss_weighting", "max_loss_weight"):
        if fm.get(k, None) is not None:
            out[k] = fm.get(k)
    if fm.get("logit_normal", None):
        out["logit_normal"].update(dict(fm.get("logit_normal")))
    if fm.get("inference", None):
        out["inference"].update(dict(fm.get("inference")))
    out["streams"] = list(fm.get("streams", []) or [])
    out["enabled"] = len(out["streams"]) > 0

    assert out["target"] in ("increment", "state"), f"flow_matching.target: {out['target']}"
    assert out["time_sampling"] in ("logit_normal", "uniform"), out["time_sampling"]
    assert out["loss_weighting"] in ("x", "v"), out["loss_weighting"]
    assert out["inference"]["solver"] in ("heun", "euler"), out["inference"]["solver"]
    assert int(out["inference"]["num_steps"]) >= 1 and int(out["inference"]["num_members"]) >= 1
    assert 0.0 <= float(out["inference"]["noise_correlation"]) < 1.0
    return out


def time_embedding(t: torch.Tensor, num_freqs: int = 8) -> torch.Tensor:
    """(N,) flow time in [0, 1] -> (N, 2 * num_freqs + 1) sinusoidal features."""
    t = t.float().unsqueeze(-1)
    freqs = math.pi * (2.0 ** torch.arange(num_freqs, device=t.device, dtype=torch.float32))
    return torch.cat([t, torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)


class FlowMatchingConditioner(torch.nn.Module):
    """
    Maps (x_t, x_{k-1}, t) at each target point to an additive term for the decoder's coordinate
    queries. Two-layer MLP whose OUTPUT layer is zero-initialised: at start the conditioning is
    exactly zero and the decoder is the deterministic decoder.

    Parameters live directly on this module (no nn.Linear sub-modules) so that WeatherGenerator's
    re-initialisation of modules missing from a checkpoint (`reset_parameters()`) keeps the zero
    initialisation of the output layer.
    """

    NUM_FREQS = 8

    def __init__(self, num_channels: int, dim_embed: int, stream_name: str, hidden: int = 0):
        super().__init__()
        self.name = f"FlowMatchingConditioner_{stream_name}"
        self.num_channels = num_channels
        self.dim_embed = dim_embed
        self.dim_in = 2 * num_channels + 2 * self.NUM_FREQS + 1
        self.hidden = hidden or dim_embed
        self.w1 = torch.nn.Parameter(torch.empty(self.hidden, self.dim_in))
        self.b1 = torch.nn.Parameter(torch.empty(self.hidden))
        self.w2 = torch.nn.Parameter(torch.empty(dim_embed, self.hidden))
        self.b2 = torch.nn.Parameter(torch.empty(dim_embed))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b1)
        torch.nn.init.zeros_(self.w2)  # zero output: exact identity of the decoder at start
        torch.nn.init.zeros_(self.b2)

    def forward(self, x_t: torch.Tensor, x_prev: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t, x_prev: (N, C) normalised; t: (N,) -> (N, dim_embed)."""
        feats = torch.cat(
            [torch.nan_to_num(x_t.float()), torch.nan_to_num(x_prev.float()),
             time_embedding(t, self.NUM_FREQS)],
            dim=-1,
        )
        h = F.gelu(F.linear(feats, self.w1, self.b1))
        return F.linear(h, self.w2, self.b2)


def sample_flow_time(n: int, fm_cfg: dict, device, generator=None) -> torch.Tensor:
    """(n,) flow times in (0, 1)."""
    if fm_cfg["time_sampling"] == "uniform":
        t = torch.rand(n, device=device, generator=generator)
    else:
        ln = fm_cfg["logit_normal"]
        z = torch.randn(n, device=device, generator=generator)
        t = torch.sigmoid(float(ln["mean"]) + float(ln["std"]) * z)
    return t.clamp(1e-4, 1.0 - 1e-4)


def flow_base(x_prev: torch.Tensor, fm_cfg: dict) -> torch.Tensor:
    """b in y = x_k - b."""
    return x_prev if fm_cfg["target"] == "increment" else torch.zeros_like(x_prev)


def training_inputs(x_target, x_prev, t_per_point, fm_cfg, generator=None):
    """
    Build the noisy flow variable for one training pass.
    x_target, x_prev: (N, C) normalised (NaN allowed); t_per_point: (N,).
    Returns (x_t, x_prev_cond, base) with NaN replaced by 0 in the network inputs.
    """
    x_prev = torch.nan_to_num(x_prev.float())
    if fm_cfg["prev_noise_std"] > 0.0:
        x_prev = x_prev + fm_cfg["prev_noise_std"] * torch.randn(
            x_prev.shape, device=x_prev.device, generator=generator
        )
    base = flow_base(x_prev, fm_cfg)
    y = torch.nan_to_num(x_target.float()) - base
    eps = torch.randn(y.shape, device=y.device, generator=generator)
    t = t_per_point.float().unsqueeze(-1)
    x_t = (1.0 - t) * eps + t * y
    return x_t, x_prev, base


def apply_loss_weighting(pred, x_target, t_per_point, fm_cfg):
    """
    loss_weighting 'x': unchanged. 'v': rescale the residual so that LossPhysical's MSE becomes
    the velocity loss, weight 1 / (1 - t)^2 (clipped). Training only.
    pred: (E, N, C); x_target: (N, C); t_per_point: (N,).
    """
    if fm_cfg["loss_weighting"] == "x":
        return pred
    w = (1.0 / (1.0 - t_per_point.float()) ** 2).clamp(max=float(fm_cfg["max_loss_weight"]))
    tgt = torch.nan_to_num(x_target.float()).to(pred.dtype)
    return tgt + torch.sqrt(w).to(pred.dtype)[None, :, None] * (pred - tgt)


def ode_time_grid(num_steps: int, device) -> torch.Tensor:
    """num_steps + 1 times from 0 to 1."""
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device)


def sample_member(denoise, x_prev, eps, fm_cfg):
    """
    Integrate the flow for one member.
    denoise(x_t, x_prev, t_scalar) -> x_hat (N, C) state prediction of the decoder.
    x_prev: (N, C) conditioning state; eps: (N, C) initial noise (t = 0).
    Returns the state x_hat_k (N, C): the decoder output of the last ODE step.
    """
    inf = fm_cfg["inference"]
    num_steps, solver = int(inf["num_steps"]), inf["solver"]
    base = flow_base(x_prev, fm_cfg)
    ts = ode_time_grid(num_steps, eps.device)
    x = eps
    x_hat = None
    for i in range(num_steps):
        t0, t1 = ts[i], ts[i + 1]
        x_hat = denoise(x, x_prev, t0)
        if i == num_steps - 1:
            break  # last step: x(1) = y_hat, i.e. the state is the decoder output itself
        v0 = (x_hat - base - x) / (1.0 - t0)
        x_euler = x + (t1 - t0) * v0
        if solver == "heun":
            x_hat1 = denoise(x_euler, x_prev, t1)
            v1 = (x_hat1 - base - x_euler) / (1.0 - t1)
            x = x + (t1 - t0) * 0.5 * (v0 + v1)
        else:
            x = x_euler
    return x_hat


def next_noise(prev_eps, shape, device, rho: float, generator=None):
    """AR(1) noise across lead times: eps_k = rho eps_{k-1} + sqrt(1 - rho^2) eta."""
    eta = torch.randn(shape, device=device, generator=generator)
    if prev_eps is None or rho <= 0.0 or prev_eps.shape != eta.shape:
        return eta
    return rho * prev_eps + math.sqrt(1.0 - rho * rho) * eta
