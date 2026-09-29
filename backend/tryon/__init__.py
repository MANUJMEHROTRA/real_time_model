from .base import TryOnBackend, TryOnError, TryOnRequest, TryOnResult


def get_backend(name: str) -> TryOnBackend:
    if name == "mock":
        from .mock import MockBackend

        return MockBackend()
    if name == "fashn":
        from .fashn import FashnBackend

        return FashnBackend()
    if name == "replicate":
        from .replicate import ReplicateBackend

        return ReplicateBackend()
    if name == "hf-idm":
        from .hf_space import IDMVTONSpace

        return IDMVTONSpace()
    if name == "hf-ootd":
        from .hf_space import OOTDiffusionSpace

        return OOTDiffusionSpace()
    raise TryOnError(f"Unknown TRYON_BACKEND '{name}' (expected mock, hf-idm, hf-ootd, fashn or replicate)")


__all__ = ["TryOnBackend", "TryOnError", "TryOnRequest", "TryOnResult", "get_backend"]
