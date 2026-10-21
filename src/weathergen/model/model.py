# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
import typing
import warnings

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.encoder import EncoderModule
from weathergen.model.engines import (
    BilinearDecoder,
    EnsPredictionHead,
    ForecastingEngine,
    IdentityEngine,
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    LatentState,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
)
from weathergen.model import flow_matching as fmatch
from weathergen.model.flow_matching import FlowMatchingConditioner, get_flow_matching_config
from weathergen.model.forcing_injection import ForcingInjection, get_forcing_reinjection_config
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.norms import AdaLayerNorm
from weathergen.model.utils import get_num_parameters
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import get_dtype, is_stream_forcing

logger = logging.getLogger(__name__)

type StreamName = str


class ModelOutput:
    """
    Representation of model output
    """

    physical: list[dict[StreamName, torch.Tensor]]
    latent: list[dict[str, torch.Tensor | LatentState]]

    def __init__(self, len_output: int) -> None:
        self.physical = [{} for _ in range(len_output)]
        self.latent = [{} for _ in range(len_output)]

    def add_physical_prediction(
        self, fstep: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[fstep][stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

    def get_physical_prediction(
        self, fstep: int, stream_name: StreamName | None = None, sample_idx: int | None = None
    ):
        pred = self.physical[fstep]
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_prediction(self, fstep: int):
        return self.latent[fstep]


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self, cf) -> None:
        super(ModelParams, self).__init__()

        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**cf.healpix_level
        self.dtype = get_dtype(cf.attention_dtype)

        # Positional embeddings
        self.max_tokens_local_per_cell = cf.get("ae_local_max_tokens_per_cell", 64)
        self.pe_embed = torch.nn.Parameter(
            torch.zeros(self.max_tokens_local_per_cell, cf.ae_local_dim_embed, dtype=self.dtype),
            requires_grad=False,
        )

        pe = torch.zeros(
            self.num_healpix_cells,
            cf.ae_local_num_queries,
            cf.ae_global_dim_embed,
            dtype=self.dtype,
        )
        self.pe_global = torch.nn.Parameter(pe, requires_grad=False)

        # RoPE coordinates
        self.rope_2D = cf.get("rope_2D", False)
        if self.rope_2D:
            self.num_extra_tokens = cf.num_register_tokens + cf.num_class_tokens
            total_tokens = (
                self.num_healpix_cells + self.num_extra_tokens
            ) * cf.ae_local_num_queries
            self.register_buffer(
                "rope_coords",
                torch.zeros(
                    1,
                    total_tokens,
                    2,
                    dtype=self.dtype,
                ),
            )
            self.register_buffer(
                "rope_cell_coords",
                torch.zeros(
                    self.num_healpix_cells,
                    2,
                    dtype=self.dtype,
                ),
            )
        else:
            self.rope_coords = None
            self.rope_cell_coords = None

        # HEALPix neighbours
        hlc = self.healpix_level
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(self.num_healpix_cells), 2**hlc, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hp_nbours = torch.nn.Parameter(
            torch.empty((temp.shape[0], (temp.shape[1] + 1)), dtype=torch.int32),
            requires_grad=False,
        )

        self.q_cells_lens = torch.nn.Parameter(
            torch.ones(self.num_healpix_cells + 1, dtype=torch.int32), requires_grad=False
        )
        self.q_cells_lens.data[0] = 0

    def create(self, cf: Config) -> "ModelParams":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> "ModelParams":
        """Creates positional embedding for each grid point for each stream used after stream
        embedding, positional embedding for all stream assimilated cell-level local embedding,
        initializing queries for local-to-global adapters, HEALPix neighbourhood based parameter
        initializing for target prediction.

        Sinusoidal positional encoding: Harmonic positional encoding based upon sine and cosine for
            both per stream after stream embedding and per cell level for local assimilation.

        HEALPix neighbourhood structure: Determine the neighbors for each cell and initialize each
            with its own cell number as well as the cell numbers of its neighbors. If a cell has
            fewer than eight neighbors, use its own cell number to fill the remaining slots.

        Query len based parameter creation: Calculate parameters for the calculated token length at
            each cell after local assimilation.

        Args:
            cf : Configuration
        """

        # positional encodings

        dim_embed = cf.ae_local_dim_embed
        token_idx_bias = 16
        freq_bias = 8
        self.pe_embed.data.fill_(0.0)
        position = torch.arange(
            token_idx_bias,
            token_idx_bias + self.max_tokens_local_per_cell,
            device=self.pe_embed.device,
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(freq_bias, freq_bias + dim_embed, 2, device=self.pe_embed.device)
            * -(math.log(self.max_tokens_local_per_cell) / dim_embed),
        )
        self.pe_embed.data[:, 0::2] = torch.sin(position * div[: self.pe_embed[:, 0::2].shape[1]])
        self.pe_embed.data[:, 1::2] = torch.cos(position * div[: self.pe_embed[:, 1::2].shape[1]])

        dim_embed = cf.ae_global_dim_embed

        if self.rope_2D:
            # Precompute per-cell center coordinates (lat, lon in radians) for 2D RoPE.
            # Shape: (num_healpix_cells, ae_local_num_queries, 2)
            verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            # Per-cell coords for QueryAggregationEngine (no query expansion)
            self.rope_cell_coords.data.copy_(coords)
            coords = coords.unsqueeze(1).repeat(1, cf.ae_local_num_queries, 1)
            coords_flat = coords.flatten(0, 1).unsqueeze(0)
            offset = self.num_extra_tokens * cf.ae_local_num_queries
            self.rope_coords.data.fill_(0.0)
            self.rope_coords.data[:, offset : offset + coords_flat.shape[1], :].copy_(coords_flat)

        # pe_global: always initialized. RoPE handles relative position in Q/K, but pe_global
        # provides per-cell token identity which is critical for masked cells that have no
        # content from local assimilation. Without it, masked cells are identical and the
        # teacher representation (evaluated without dropout) collapses to low rank.
        self.pe_global.data.fill_(0.0)
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=self.pe_global.device) / dim_embed
        self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 0::2] += (
            torch.sin(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 1::2] += (
            torch.cos(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )

        # healpix neighborhood structure

        hlc = self.healpix_level
        num_healpix_cells = self.num_healpix_cells
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(np.arange(num_healpix_cells), 2**hlc, order="nested").transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        # nbors *and* self
        self.hp_nbours.data[:, 0] = torch.arange(temp.shape[0], device=self.hp_nbours.device)
        self.hp_nbours.data[:, 1:] = torch.from_numpy(temp).to(self.hp_nbours.device)

        # precompute for varlen attention
        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

        # ensure all params have grad set to False

        return


class Model(torch.nn.Module):
    """WeatherGenerator model architecture

    WeatherGenerator consists of the following components:

    embeds: embedding networks: Stream specific embedding networks.

    ae_local_blocks: Local assimilation engine: transformer based network to combine different input
        streams per healpix cell.

    ae_adapter: Assimilation engine adapter: Adapter to transform local assimilation engine
        information to the global assimilation engine.

    ae_aggregation_blocks: Query aggregation engine: after the learnable queries are created per
        non-masked healpix cell, this engine combines information from all non-masked cells by
        using dense attention layers.

    ae_global_blocks: Global assimilation engine: Transformer network alternating between local and
        global attention based upon global attention density rate.

    fe_blocks: Forecasting engine: Transformer network using the output of global attention to
        advance the latent representation in time.

    embed_target_coords: Embedding networks for coordinates: Initializes embedding networks tailored
        for metadata embedded target coordinates. The architecture is either a linear layer or a
        multi-layer perceptron, determined by the configuration of the embedding target coordinate
        networks.

    pred_adapter_kv: Prediction adapter: Adapter to transform the global assimilation/forecasting
        engine output to the prediction engine. Uses an MLP if `cf.pred_adapter_kv` is True,
        otherwise it uses an identity function.

    target_token_engines: Prediction engine: Transformer based prediction network that generates
        output corresponding to target coordinates.

    pred_heads: Prediction head: Final layers using target token engines output for mapping target
        coordinates to its physical space.
    """

    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size):
        """
        Args:
            cf : Configuration with model parameters
            sources_size : List of number of channels for models
            targets_num_channels : List with size of each output sample for coordinates target
                embedding
            targets_coords_size : List with size of each input sample for coordinates target
                embedding
        """
        super(Model, self).__init__()

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.embed_target_coords = None
        self.encoder: EncoderModule | None = None
        self.forecast_engine: ForecastingEngine | IdentityEngine | None = None
        # forcing re-injection at every forecast step (None = disabled, default behaviour)
        self.forcing_injection: ForcingInjection | None = None
        self.forcing_reinjection = get_forcing_reinjection_config(cf)
        # flow-matching decoders (None = disabled, default behaviour); sampling only at inference
        self.fm_cfg = get_flow_matching_config(cf)
        self.fm_conditioners: nn.ModuleDict | None = None
        self.fm_sampling = False
        self.pred_heads = None
        self.q_cells: torch.Tensor | None = None
        self.streams: dict[str, typing.Any] = cf.streams
        self.target_token_engines = None

        assert cf.get("forecast", {}).get("att_dense_rate", 1.0) == 1.0, (
            "Local attention not adapted for register tokens"
        )
        self.num_register_tokens = cf.num_register_tokens
        self.latent_heads = None
        self.latent_pre_norm = None
        # auxiliary tokens
        self.class_token_idxs = list(
            range(cf.num_register_tokens, cf.num_register_tokens + cf.num_class_tokens)
        )
        self.register_token_idxs = list(range(cf.num_register_tokens))
        self.aux_token_idxs = list(range(cf.num_register_tokens + cf.num_class_tokens))
        self.num_aux_tokens = cf.num_register_tokens + cf.num_class_tokens

    def _create_latent_pred_head(
        self, global_cfg, name, loss_cfg, use_class_token, use_patch_token
    ):
        if loss_cfg["head"].lower() == "mlp":
            return LatentPredictionHeadMLP(
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "transformer":
            return LatentPredictionHeadTransformer(
                global_cfg,
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "identity":
            return LatentPredictionHeadIdentity()
        else:
            assert False, f"Unknown latent prediction head type {loss_cfg['head']}"

    def create(self) -> "Model":
        """Create each individual module of the model"""
        cf = self.cf

        self.encoder = EncoderModule(
            cf, self.sources_size, self.targets_num_channels, self.targets_coords_size
        )

        mode_cfg = cf.training_config
        if cf.fe_num_blocks > 0:
            self.forecast_engine = ForecastingEngine(cf, mode_cfg, self.num_healpix_cells)
        else:
            self.forecast_engine = IdentityEngine()

        if self.forcing_reinjection["enabled"]:
            self.forcing_injection = ForcingInjection(cf.ae_global_dim_embed)
            if is_root():
                logger.info(
                    "Forcing re-injection enabled for streams "
                    f"{self.forcing_reinjection['streams']} (zero-initialised projection, "
                    f"train_encoder={self.forcing_reinjection['train_encoder']})"
                )

        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()

        # determine stream names once so downstream components use consistent keys
        loss_terms = [
            v.type for _, v in cf.training_config.losses.items() if v.get("enabled", True)
        ]
        if cf.validation_config.get("losses"):
            loss_terms += [
                v.type for _, v in cf.validation_config.losses.items() if v.get("enabled", True)
            ]

        if "LossPhysical" in loss_terms:
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                # skip for the moment to ensure target embedding and tte exist (ordering of
                # cf.streams is random)
                if si.get("pred_spatial_shared") is None:
                    # extract and setup relevant parameters
                    etc = si["embed_target_coords"]
                    tr = si["target_readout"]
                    num_layers = tr["num_layers"]
                    tr_mlp_hidden_factor = (
                        tr["mlp_hidden_factor"] if "mlp_hidden_factor" in tr else 2
                    )
                    tr_dim_head_proj = tr["dim_head_proj"] if "dim_head_proj" in tr else None
                    softcap = tr["softcap"] if "softcap" in tr else 0.0

                    dims_embed = [
                        si["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    if is_root():
                        logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

                    dim_coord_in = self.targets_coords_size[i_stream]

                    # embedding network for coordinates
                    if etc["net"] == "linear":
                        self.embed_target_coords[stream_name] = NamedLinear(
                            f"embed_target_coords_{stream_name}",
                            in_features=dim_coord_in,
                            out_features=dims_embed[0],
                            bias=False,
                        )
                    elif etc["net"] == "mlp":
                        self.embed_target_coords[stream_name] = MLP(
                            dim_coord_in,
                            dims_embed[0],
                            hidden_factor=8,
                            with_residual=False,
                            dropout_rate=dropout_rate,
                            norm_eps=self.cf.mlp_norm_eps,
                            name=f"embed_target_coords_{stream_name}",
                        )
                    else:
                        assert False

                    if cf.decoder_type == "Linear":
                        tte = BilinearDecoder(
                            stream_name,
                            dims_embed[0],
                            cf.ae_global_dim_embed,
                            self.targets_num_channels[i_stream],
                        )
                    else:
                        # target prediction engines
                        tte_version = (
                            TargetPredictionEngine
                            if cf.decoder_type != "PerceiverIOCoordConditioning"
                            else TargetPredictionEngineClassic
                        )
                        tte = tte_version(
                            cf,
                            dims_embed,
                            dim_coord_in,
                            tr_dim_head_proj,
                            tr_mlp_hidden_factor,
                            softcap,
                            stream_config=si,
                        )

                    self.target_token_engines[stream_name] = tte

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

            # iterate again to setup shared spatial pred heads if specified in config
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                pred_spatial_shared = si.get("pred_spatial_shared")
                if pred_spatial_shared is not None:
                    if pred_spatial_shared not in self.streams.keys():
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += " but no stream with that name found."
                        raise ValueError(msg)
                    if pred_spatial_shared == stream_name:
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += "but cannot share with itself."
                        raise ValueError(msg)
                    logger.debug(
                        f"{stream_name} shares spatial prediction head with {pred_spatial_shared}."
                    )

                    self.embed_target_coords[stream_name] = self.embed_target_coords[
                        pred_spatial_shared
                    ]
                    self.target_token_engines[stream_name] = self.target_token_engines[
                        pred_spatial_shared
                    ]

                    assert pred_spatial_shared in self.streams.keys()
                    si_other = self.streams[pred_spatial_shared]
                    dims_embed = [
                        si_other["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

        # flow-matching conditioning of the decoders of the selected streams
        if self.fm_cfg["enabled"]:
            assert cf.decoder_type != "Linear", "flow matching needs a target prediction engine"
            self.fm_conditioners = nn.ModuleDict()
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                if stream_name not in self.fm_cfg["streams"]:
                    continue
                assert stream_name in self.pred_heads, f"no decoder for FM stream {stream_name}"
                assert si["pred_head"]["ens_size"] == 1, (
                    f"flow matching stream {stream_name}: use pred_head.ens_size 1 (members come "
                    "from flow_matching.inference.num_members)"
                )
                self.fm_conditioners[stream_name] = FlowMatchingConditioner(
                    self.targets_num_channels[i_stream],
                    si["embed_target_coords"]["dim_embed"],
                    stream_name,
                )
            if is_root():
                logger.info(
                    f"Flow matching decoders for {list(self.fm_conditioners.keys())}: "
                    f"target={self.fm_cfg['target']}, "
                    f"loss_weighting={self.fm_cfg['loss_weighting']}, "
                    f"inference={self.fm_cfg['inference']}"
                )

        # Latent heads for losses
        self.latent_heads = nn.ModuleDict()
        self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)

        ssl_losses_cfgs = [
            v
            for _, v in cf.training_config.losses.items()
            if v.type == "LossLatentSSLStudentTeacher" and v.get("enabled", True)
        ]

        # TODO: support multiple LossLatentSSLStudentTeacher terms
        assert len(ssl_losses_cfgs) <= 1, "To be implemented."
        for ssl_target_losses in ssl_losses_cfgs:
            self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)
            for loss, loss_conf in ssl_target_losses.loss_fcts.items():
                if loss == "iBOT":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=True,
                    )
                elif loss == "JEPA":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=False,
                        use_patch_token=True,
                    )
                elif loss == "DINO":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=False,
                    )

        return self

    def reset_parameters(self):
        def _reset_params(module):
            reset_types = AdaLayerNorm | nn.Linear | nn.LayerNorm
            if isinstance(module, reset_types | ForcingInjection | FlowMatchingConditioner):
                module.reset_parameters()
            else:
                pass

        self.apply(_reset_params)

    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        num_params_embed = [
            get_num_parameters(self.encoder.embed_engine.embeds[name])
            for name in self.streams.keys()
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.encoder.ae_local_engine.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.encoder.ae_global_engine.ae_global_blocks)

        num_params_q_cells = (
            np.prod(self.encoder.q_cells.shape) if self.encoder.q_cells.requires_grad else 0
        )
        num_params_ae_adapter = get_num_parameters(self.encoder.ae_local_global_engine)

        num_params_ae_aggregation = get_num_parameters(
            self.encoder.ae_aggregation_engine.ae_aggregation_blocks
        )

        num_params_latent_heads = get_num_parameters(self.latent_heads)
        num_params_latent_heads += get_num_parameters(self.latent_pre_norm)

        num_params_fe = get_num_parameters(self.forecast_engine.fe_blocks)
        num_params_fi = (
            get_num_parameters(self.forcing_injection) if self.forcing_injection is not None else 0
        )
        num_params_fm = (
            get_num_parameters(self.fm_conditioners) if self.fm_conditioners is not None else 0
        )

        mdict = self.embed_target_coords
        num_params_embed_tcs = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.target_token_engines
        num_params_tte = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.pred_heads
        num_params_preds = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]

        print("-----------------")
        print(f"Total number of trainable parameters: {num_params_total:,}")
        print("Number of parameters:")
        print("  Embedding networks:")
        [
            print("    {} : {:,}".format(si["name"], np))
            for si, np in zip(self.streams.values(), num_params_embed, strict=False)
        ]
        print(f" Local assimilation engine: {num_params_ae_local:,}")
        print(f" Local-global adapter: {num_params_ae_adapter:,}")
        print(f" Learnable queries: {num_params_q_cells:,}")
        print(f" Query Aggregation engine: {num_params_ae_aggregation:,}")
        print(f" Global assimilation engine: {num_params_ae_global:,}")
        print(f" Latent prediction heads and pre-norm: {num_params_latent_heads:,}")
        print(f" Forecast engine: {num_params_fe:,}")
        if self.forcing_injection is not None:
            print(f" Forcing re-injection: {num_params_fi:,}")
        if self.fm_conditioners is not None:
            print(f" Flow-matching conditioning: {num_params_fm:,}")
        print(" coordinate embedding, prediction networks and prediction heads:")
        zps = zip(
            self.streams.keys(),
            num_params_embed_tcs,
            num_params_tte,
            num_params_preds,
            strict=False,
        )
        for stream_name, np0, np1, np2 in zps:
            print(f"   {stream_name} : {np0:,} / {np1:,} / {np2:,}")
        print("-----------------")

    def tokens_to_latent_state(self, tokens_post_norm, tokens) -> LatentState:
        """
        Extract separate parts from global latent space representation and store in LatentState
        """
        toks_pn = tokens_post_norm
        return LatentState(
            register_tokens=toks_pn[:, self.register_token_idxs] if toks_pn is not None else None,
            class_token=toks_pn[:, self.class_token_idxs] if tokens_post_norm is not None else None,
            patch_tokens=toks_pn[:, self.num_aux_tokens :] if toks_pn is not None else None,
            z_pre_norm=tokens,
        )

    def forward(self, model_params: ModelParams, batch: ModelBatch) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch
        Returns:
            A list containing all prediction results
        """

        output = ModelOutput(batch.get_output_len())

        tokens, posteriors = self.encoder(model_params, batch)
        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        shape = (len(batch), batch.get_num_source_steps(), *tokens.shape[1:])
        # collapse along input step dimension
        tokens = tokens.reshape(shape).sum(axis=1)

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        get_forcing = getattr(batch, "get_forcing_samples", None)
        forcing = get_forcing() if (self.forcing_injection is not None and get_forcing) else None
        # flow matching: per-stream sampling state (previous member states, noise) across steps
        fm_state = {}
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in batch.get_output_idxs():
            # forcing re-injection: add the encoded forcing valid at this step
            if forcing is not None and step in forcing:
                tokens = self.inject_forcing(model_params, tokens, forcing[step], len(batch))

            without_grad = p_fwd and self.training and step != max(batch.get_output_idxs())
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding with torch.no_grad():
                tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
                continue

            tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
            # decoder predictions
            output = self.predict_decoders(model_params, step, tokens, batch, output, fm_state)
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(model_params, step, tokens, batch, output)

        return output

    def inject_forcing(self, model_params, tokens, forcing_batch, batch_size: int) -> torch.Tensor:
        """
        Encode the forcing streams valid at the current forecast step with the regular encoder
        and add them to the latent tokens through the (zero-initialised) ForcingInjection.
        """
        if self.forcing_reinjection["train_encoder"]:
            f_tokens, _ = self.encoder(model_params, forcing_batch)
        else:
            with torch.no_grad():
                f_tokens, _ = self.encoder(model_params, forcing_batch)
        shape = (batch_size, forcing_batch.get_num_source_steps(), *f_tokens.shape[1:])
        f_tokens = f_tokens.reshape(shape).sum(axis=1)
        return self.forcing_injection(tokens, f_tokens)

    def set_flow_matching_sampling(self, enabled: bool) -> None:
        """Generate members by ODE sampling (inference) instead of the one-pass training loss."""
        self.fm_sampling = bool(enabled) and self.fm_conditioners is not None
        if self.fm_sampling and is_root():
            logger.info(f"Flow matching: sampling enabled ({self.fm_cfg['inference']})")

    def _fm_gather(self, batch, stream_name, step, attr, lens, num_channels, device):
        """Concatenate fm_target / fm_prev of all samples (NaN where missing): (N, C)."""
        parts = []
        for sample, n in zip(batch.samples, lens, strict=True):
            values = getattr(sample.streams_data[stream_name], attr, None)
            v = values[step] if values is not None and step < len(values) else None
            if v is None or v.shape[0] != n:
                v = torch.full((n, num_channels), float("nan"), device=device)
            parts.append(v.to(device=device, dtype=torch.float32))
        return torch.cat(parts) if parts else torch.zeros((0, num_channels), device=device)

    def predict_flow_matching(
        self, stream_name, step, batch, tc_tokens, decode, t_coords_lens, fm_state
    ) -> torch.Tensor:
        """
        Flow-matching decoder of one stream at one forecast step (see model/flow_matching.py).
        Returns the state prediction(s), shape (members, N, C): 1 member in the training /
        validation pass, num_members when sampling.
        """
        fm = self.fm_cfg
        cond = self.fm_conditioners[stream_name]
        dev, num_ch, n_pts = tc_tokens.device, cond.num_channels, tc_tokens.shape[0]
        x_prev = self._fm_gather(batch, stream_name, step, "fm_prev", t_coords_lens, num_ch, dev)

        out_dtype = {}

        def denoise(x_t, x_prev_c, t_pts):
            queries = tc_tokens + cond(x_t, x_prev_c, t_pts).to(tc_tokens.dtype)
            out = decode(queries)  # (1, N, C) state prediction
            out_dtype.setdefault("dtype", out.dtype)
            return out

        if not self.fm_sampling:
            # training / validation: one noisy pass, LossPhysical MSE vs x_k is the FM loss
            x_tgt = self._fm_gather(
                batch, stream_name, step, "fm_target", t_coords_lens, num_ch, dev
            )
            t_s = fmatch.sample_flow_time(len(t_coords_lens), fm, dev)
            t_pts = torch.repeat_interleave(t_s, torch.tensor(t_coords_lens, device=dev))
            x_t, x_prev_c, _ = fmatch.training_inputs(x_tgt, x_prev, t_pts, fm)
            pred = denoise(x_t, x_prev_c, t_pts)
            if self.training:
                pred = fmatch.apply_loss_weighting(pred, x_tgt, t_pts, fm)
            return pred

        # sampling: num_members ODE integrations, each conditioned on its own previous state
        inf = fm["inference"]
        n_mem, rho = int(inf["num_members"]), float(inf["noise_correlation"])
        state = fm_state.setdefault(stream_name, {"prev": None, "eps": None})
        prev = state["prev"]
        if prev is None or prev.shape[1] != n_pts:
            # first step: analysis x_{k-1} (truth at the initial time) for every member
            prev = torch.nan_to_num(x_prev).unsqueeze(0).expand(n_mem, -1, -1)
        members, eps_all = [], []
        for m in range(n_mem):
            prev_eps = state["eps"][m] if state["eps"] is not None else None
            eps = fmatch.next_noise(prev_eps, (n_pts, num_ch), dev, rho)

            def denoise_m(x_t, x_prev_c, t):
                return denoise(x_t, x_prev_c, t.expand(n_pts))[0].float()

            members.append(fmatch.sample_member(denoise_m, prev[m], eps, fm))
            eps_all.append(eps)
        pred = torch.stack(members, 0)
        state["prev"], state["eps"] = pred.detach(), torch.stack(eps_all, 0)
        # same dtype as the decoder output (exact identity with the deterministic decoder at init)
        return pred.to(out_dtype.get("dtype", pred.dtype))

    def predict_latent(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute latent predictions
        """

        # safe latent prediction
        tokens_post_norm = self.latent_pre_norm(tokens) if step == 0 else None
        latent_state = self.tokens_to_latent_state(tokens_post_norm, tokens)
        output.add_latent_prediction(step, "latent_state", latent_state)

        # latent predictions for SSL training
        for name, head in self.latent_heads.items():
            output.add_latent_prediction(step, name, head(latent_state))

        return output

    def predict_decoders(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
        fm_state: dict | None = None,
    ) -> ModelOutput:
        """
        Compute decoder-based predictions

        Predict outputs at the specific target coordinates based on the input weather state and
        pre-training task and projects the latent space representation back to physical space.

        Args:
            model_params : Query and embedding parameters
            fstep : Number of forecast steps
            tokens : Tokens from global assimilation engine
            streams_data : Used to initialize target coordinates tokens and index information
                List of StreamData len(streams_data) == batch_size_per_gpu
            target_coords_idxs : Indices of target coordinates
        Returns:
            Prediction output tokens in physical representation for each target_coords.
        """
        # Empty dicts evaluate to False in python
        if not self.pred_heads:
            return output

        # remove register  and class tokens
        tokens = tokens[:, self.num_aux_tokens :]

        # get 1-ring neighborhood for prediction
        batch_size = len(batch)
        s = [batch_size, self.num_healpix_cells, self.cf.ae_local_num_queries, tokens.shape[-1]]
        idxs = model_params.hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)
        tokens_nbors = tokens.reshape(s).flatten(0, 1)[idxs.flatten()].flatten(0, 1)
        # TODO: precompute in model_params?
        tokens_nbors_lens = torch.full(
            (s[0] * s[1] + 1,), fill_value=9, dtype=torch.int32, device=tokens_nbors.device
        )
        tokens_nbors_lens[0] = 0

        # pair with tokens from assimilation engine to obtain target tokens
        for stream_name in self.streams.keys():
            # extract target coords for current stream and fstep and convert to one tensor
            t_coords = [
                batch.samples[i_b].streams_data[stream_name].target_coords[step]
                for i_b in range(batch_size)
            ]
            t_coords_lens = [len(t) for t in t_coords]
            t_coords = torch.cat(t_coords)

            if len(t_coords) == 0:
                continue

            # embed token coords
            tc_embed = self.embed_target_coords[stream_name]
            tc_tokens = checkpoint(tc_embed, t_coords, use_reentrant=False)

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                logger.warning(
                    (
                        f"Skipping prediction for {stream_name} because",
                        f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens.",
                    )
                )
                pred = torch.tensor([], device=tc_tokens.device)

            # skip empty lengths
            elif tc_tokens.shape[0] == 0:
                pred = torch.tensor([], device=tc_tokens.device)

            else:
                # lens for varlen attention
                tcls = torch.cat(
                    [
                        sample.streams_data[stream_name].target_coords_lens[step]
                        for sample in batch.samples
                    ]
                )
                tcs_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device=tcls.device), tcls])

                if self.cf.decoder_type == "Linear":
                    pred = self.target_token_engines[stream_name](
                        tc_tokens,
                        tokens.reshape(-1, s[-1]),  # collapse the batch and token dimensions
                        tcs_lens,
                    ).unsqueeze(0)  # add ensemble dim: shape is then [1, preds_per_coord, channels]
                else:

                    def decode(queries, stream_name=stream_name, tcs_lens=tcs_lens):
                        toks = self.target_token_engines[stream_name](
                            latent=tokens_nbors,
                            output=queries,
                            latent_lens=tokens_nbors_lens,
                            output_lens=tcs_lens,
                            coordinates=t_coords,
                        )
                        # final prediction head to map back to physical space
                        return self.pred_heads[stream_name](toks)

                    if self.fm_conditioners is not None and stream_name in self.fm_conditioners:
                        pred = self.predict_flow_matching(
                            stream_name, step, batch, tc_tokens, decode, t_coords_lens, fm_state
                        )
                    else:
                        pred = decode(tc_tokens)

            # recover batch dimension (ragged, so as list)
            pred = torch.split(pred, t_coords_lens, dim=1)
            output.add_physical_prediction(step, stream_name, pred)

        return output
