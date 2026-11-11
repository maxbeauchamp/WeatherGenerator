# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Input channels added to a stream after pre-training (e.g. ssrd / strd in the ERA5 forcing of the
ice + ERA5 model).

The stream embedder of a pre-trained model has a fixed number of input channels, and its
StreamEmbedTransformer mixes the channels (self-attention over channels, LayerNorm and unembedding
over all of them): adding channels to it would change the output for the old channels even with
zero weights. Instead, the stream embedder keeps exactly its pre-trained input (the old channels,
in the same order) and a small side network embeds the new channels; its output is ADDED to the
stream's tokens:

    tokens = E_stream(x_old) + E_extra(x_new),    E_extra zero-initialised at its output

so the model is exactly the pre-trained one at start. Switch (stream config, off by default):

    source : [... old channels ..., 'ssrd', 'strd']
    embed_extra_channels : ['ssrd', 'strd']
"""

import math

import torch
import torch.nn.functional as F


def source_channel_names(stream_info) -> list[str]:
    """
    Source channels of a stream in the order of the data columns, as written in the stream
    config by the data sampler before the model is built. 'val' first: in inference only the
    'val' list is written, a 'train' list may come from the json of the loaded run.
    """
    for stage in ("val", "train", "test"):
        names = stream_info.get(f"{stage}_source_channels")
        if names is not None and len(names) > 0:
            return list(names)
    raise ValueError(
        f"embed_extra_channels: no source channel list in the config of stream "
        f"{stream_info.get('name', '?')} (the data sampler must be created before the model)."
    )


def extra_channel_columns(
    source_names: list[str], num_columns: int, extra: list[str]
) -> tuple[list[int], list[int]]:
    """
    Split the token columns into the columns of the pre-trained embedder and the new ones.
    The data channels are the last len(source_names) columns of a token (after stream id, time,
    local coordinates and geoinfo).
    Returns (base_columns, extra_columns), each in increasing order.
    """
    missing = [c for c in extra if c not in source_names]
    if missing:
        raise ValueError(f"embed_extra_channels {missing} are not source channels {source_names}")
    if len(set(extra)) != len(extra):
        raise ValueError(f"embed_extra_channels has duplicates: {extra}")
    offset = num_columns - len(source_names)
    if offset < 0:
        raise ValueError(f"{len(source_names)} source channels but {num_columns} token columns")
    extra_cols = sorted(offset + source_names.index(c) for c in extra)
    base_cols = [i for i in range(num_columns) if i not in set(extra_cols)]
    return base_cols, extra_cols


class ExtraChannelEmbed(torch.nn.Module):
    """
    Side embedder of the new channels: (N, token_size, num_extra) -> (N, num_tokens, dim_out),
    MLP with a zero-initialised output layer.

    Parameters live directly on this module (no nn.Linear sub-modules) so that WeatherGenerator's
    re-initialisation of modules missing from a checkpoint (`reset_parameters()`) keeps the zero
    initialisation of the output layer.
    """

    def __init__(
        self,
        num_extra: int,
        token_size: int,
        num_tokens: int,
        dim_out: int,
        stream_name: str,
        hidden: int = 256,
    ):
        super().__init__()
        self.name = f"ExtraChannelEmbed_{stream_name}"
        self.num_extra = num_extra
        self.token_size = token_size
        self.num_tokens = num_tokens
        self.dim_out = dim_out
        self.dim_in = num_extra * token_size
        self.hidden = hidden
        self.w1 = torch.nn.Parameter(torch.empty(hidden, self.dim_in))
        self.b1 = torch.nn.Parameter(torch.empty(hidden))
        self.w2 = torch.nn.Parameter(torch.empty(num_tokens * dim_out, hidden))
        self.b2 = torch.nn.Parameter(torch.empty(num_tokens * dim_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b1)
        torch.nn.init.zeros_(self.w2)  # zero output: exact identity of the model at start
        torch.nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, token_size, num_extra) normalised (padding = 0, NaN -> 0)."""
        h = F.gelu(F.linear(torch.nan_to_num(x.float()).flatten(1), self.w1, self.b1))
        out = F.linear(h, self.w2, self.b2)
        return out.reshape(-1, self.num_tokens, self.dim_out)
