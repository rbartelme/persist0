from ._core import h0_batched

__all__ = ["h0_batched", "h0_persistence", "TopoH0Loss"]


def __getattr__(name):
    if name == "h0_persistence":
        from .functional import h0_persistence
        return h0_persistence
    if name == "TopoH0Loss":
        from .loss import TopoH0Loss
        return TopoH0Loss
    raise AttributeError(f"module 'persist0' has no attribute {name!r}")
