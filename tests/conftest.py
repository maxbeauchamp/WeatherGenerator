# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
CPU fallback for flash_attn, for local tests (e.g. macOS, no GPU).

flash_attn is GPU-only and imported unconditionally by weathergen.model.attention. A
pure-torch reference implementation of the two functions used by the code is provided. Slow,
exact up to float rounding, no dropout.
- flash_attn missing (macOS): installed as the `flash_attn` module.
- flash_attn installed (JUPITER): CUDA tensors go to the real flash_attn, CPU tensors (tests
  built on the CPU) to the fallback, which the real kernels do not support.
"""

import importlib.util
import math
import sys
import types

import torch


def _attention(q, k, v, softmax_scale, softcap):
    """q (Lq, H, D), k/v (Lk, H, D) -> (Lq, H, D)."""
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    w = torch.softmax(logits, dim=-1)
    return torch.einsum("hqk,khd->qhd", w, v.float()).to(q.dtype)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    **kwargs,
):
    assert not causal, "CPU fallback: causal attention not implemented"
    out = torch.zeros((q.shape[0], q.shape[1], v.shape[-1]), dtype=q.dtype, device=q.device)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        if cq[i + 1] > cq[i] and ck[i + 1] > ck[i]:
            out[cq[i] : cq[i + 1]] = _attention(
                q[cq[i] : cq[i + 1]], k[ck[i] : ck[i + 1]], v[ck[i] : ck[i + 1]],
                softmax_scale, softcap,
            )  # fmt: skip
    return out


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, softcap=0.0, **kw):
    """q (B, Lq, H, D), k/v (B, Lk, H, D)."""
    assert not causal, "CPU fallback: causal attention not implemented"
    return torch.stack(
        [_attention(q[b], k[b], v[b], softmax_scale, softcap) for b in range(q.shape[0])]
    )


if importlib.util.find_spec("flash_attn") is None:
    _mod = types.ModuleType("flash_attn")
    _mod.flash_attn_varlen_func = flash_attn_varlen_func
    _mod.flash_attn_func = flash_attn_func
    _mod.__cpu_fallback__ = True
    sys.modules["flash_attn"] = _mod
else:
    import flash_attn as _fa

    _real = {
        "flash_attn_varlen_func": _fa.flash_attn_varlen_func,
        "flash_attn_func": _fa.flash_attn_func,
    }
    _cpu = {"flash_attn_varlen_func": flash_attn_varlen_func, "flash_attn_func": flash_attn_func}

    def _dispatch(name):
        def fn(q, *args, **kwargs):
            return (_real if q.is_cuda else _cpu)[name](q, *args, **kwargs)

        return fn

    for _name in _real:
        setattr(_fa, _name, _dispatch(_name))
        if "weathergen.model.attention" in sys.modules:  # already imported: patch its names too
            setattr(sys.modules["weathergen.model.attention"], _name, getattr(_fa, _name))
