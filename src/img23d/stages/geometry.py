from ..backends import create_backend
from ..runtime import clear_gpu, stage


def generate_geometry(
    image,
    cache,
    backend="hunyuan",
    *,
    resolution=192,
    steps=30,
    seed=12345,
    records=None,
    backend_factory=None,
):
    """Gera geometria e descarta o backend antes de devolver a malha CPU."""
    records = records if records is not None else []
    factory = backend_factory or create_backend
    model = None
    with stage("geometria", records):
        try:
            model = factory(
                backend, cache, resolution=resolution, steps=steps, seed=seed
            )
            return model.generate(image)
        finally:
            try:
                close = getattr(model, "close", None)
                if callable(close):
                    close()
            finally:
                # Um método bound também retém o objeto: descarte ambos antes do GC.
                close = None
                del model
                clear_gpu()
