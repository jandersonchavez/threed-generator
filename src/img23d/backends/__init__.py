from .base import Backend as Backend


def create_backend(name, cache, **kwargs):
    if name == "hunyuan":
        from .hunyuan import HunyuanBackend

        return HunyuanBackend(cache, **kwargs)
    if name == "triposr":
        from .triposr import TripoSRBackend

        return TripoSRBackend(cache, **kwargs)
    raise ValueError(f"Backend desconhecido: {name}")
