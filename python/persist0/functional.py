import numpy as np
import torch

from ._core import h0_batched


def h0_persistence(D: torch.Tensor):
    """H0 persistence of a batch of distance matrices.

    D: [B, N, N] float tensor, symmetric, zero diagonal. May require grad.
    Returns (idx, deaths): idx is [B, N-1, 2] long (death edge i<j),
    deaths is [B, N-1] gathered from D so gradients flow.
    """
    if D.dim() != 3 or D.shape[-1] != D.shape[-2]:
        raise ValueError("D must be [B, N, N]")
    D_np = np.ascontiguousarray(D.detach().to("cpu", torch.float32).numpy())
    idx = torch.from_numpy(h0_batched(D_np).astype(np.int64)).to(D.device)
    b = torch.arange(D.shape[0], device=D.device)[:, None]
    deaths = D[b, idx[..., 0], idx[..., 1]]
    return idx, deaths
