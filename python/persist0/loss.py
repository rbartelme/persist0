import torch
import torch.nn as nn

from .functional import h0_persistence


class TopoH0Loss(nn.Module):
    """Match the sorted H0 death vector of a live distance matrix to a reference.

    top_k: if set, compare only the k largest deaths (most persistent bars).
    """

    def __init__(self, top_k: int | None = None):
        super().__init__()
        self.top_k = top_k

    def forward(self, D_live: torch.Tensor, D_ref: torch.Tensor) -> torch.Tensor:
        _, live = h0_persistence(D_live)
        with torch.no_grad():
            _, ref = h0_persistence(D_ref)
        live = live.sort(dim=1, descending=True).values
        ref = ref.sort(dim=1, descending=True).values
        if self.top_k is not None:
            live, ref = live[:, : self.top_k], ref[:, : self.top_k]
        return torch.nn.functional.mse_loss(live, ref)
