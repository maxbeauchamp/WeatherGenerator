# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Re-injection of forcing streams (e.g. ERA5) into the latent state at every forecast step.

Without it, forcing streams are only seen in the input window: the forecast engine then rolls
out purely in latent space and has to "guess" the forcing at later lead times. With it, at each
forecast step the forcing data valid at that step is encoded with the regular encoder and added
to the latent state before the forecast engine:

    tokens_k = FE( tokens_{k-1} + W . LN( Enc(forcing_k) ) + b )

W and b are initialised to ZERO, so a model with re-injection behaves exactly like the model it
is fine-tuned from until W is trained.

Config (main config, not the stream files):

    forcing_reinjection:
      enabled: True
      streams: [ERA5]        # streams re-encoded at every forecast step (source channels used)
      train_encoder: False   # False: encode the forcing without gradient (cheaper)
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_forcing_reinjection_config(cf) -> dict:
    """Normalised view of the `forcing_reinjection` config entry (disabled if absent)."""
    fr = cf.get("forcing_reinjection", None) if cf is not None else None
    if not fr or not fr.get("enabled", False):
        return {"enabled": False, "streams": [], "train_encoder": False}
    streams = list(fr.get("streams", []) or [])
    return {
        "enabled": len(streams) > 0,
        "streams": streams,
        "train_encoder": bool(fr.get("train_encoder", False)),
    }


class ForcingInjection(torch.nn.Module):
    """
    Zero-initialised affine projection of the encoded forcing, added to the latent tokens.

    The parameters live directly on this module (no nn.Linear sub-module) on purpose: when a
    checkpoint without these weights is loaded, WeatherGenerator re-initialises new modules by
    calling their `reset_parameters()`. Here that keeps the zero initialisation, whereas an
    nn.Linear would be re-initialised randomly and change the behaviour of the loaded model.
    """

    def __init__(self, dim_embed: int, name: str = "ForcingInjection") -> None:
        super().__init__()
        self.name = name
        self.dim_embed = dim_embed
        self.weight = torch.nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.bias = torch.nn.Parameter(torch.empty(dim_embed))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, tokens: torch.Tensor, forcing_tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens         : latent state       (batch, num_tokens, dim_embed)
        forcing_tokens : encoded forcing    (batch, num_tokens, dim_embed)
        """
        # nan_to_num: with zero weights the injection must be an exact identity (0 * NaN = NaN)
        f = torch.nan_to_num(F.layer_norm(forcing_tokens.float(), (self.dim_embed,)))
        w, b = self.weight.to(tokens.dtype), self.bias.to(tokens.dtype)
        return tokens + F.linear(f.to(tokens.dtype), w, b)
