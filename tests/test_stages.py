import gc
import weakref

import pytest
import trimesh
from PIL import Image

from img23d.stages.geometry import generate_geometry
from img23d.stages.postprocess import prepare_print_mesh
from img23d.stages.texture import TextureProcessor, apply_texture


def test_geometry_releases_backend_before_texture():
    events, refs, records = [], [], []

    class Geometry:
        def __init__(self):
            refs.append(weakref.ref(self))

        def generate(self, image):
            events.append("geometry")
            return trimesh.creation.box()

        def close(self):
            events.append("geometry_closed")

    class Texture(TextureProcessor):
        def __init__(self):
            gc.collect()
            assert refs[0]() is None
            events.append("texture_loaded")
            refs.append(weakref.ref(self))

        def apply(self, mesh, image):
            return mesh

        def close(self):
            events.append("texture_closed")

    image = Image.new("RGB", (32, 32))
    mesh = generate_geometry(
        image, ".", backend_factory=lambda *a, **k: Geometry(), records=records
    )
    result = apply_texture(mesh, image, processor_factory=Texture, records=records)
    gc.collect()
    assert result is mesh and all(ref() is None for ref in refs)
    assert events == ["geometry", "geometry_closed", "texture_loaded", "texture_closed"]
    assert [r["estado"] for r in records] == ["concluida", "concluida"]
    assert all("vram_apos_mib" in row for row in records)


@pytest.mark.parametrize("which", ["geometry", "texture"])
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_resource_closed_on_stage_failure(which, failure):
    events, records = [], []

    class Broken(TextureProcessor):
        def generate(self, image):
            raise failure("erro de teste")

        def apply(self, mesh, image):
            raise failure("erro de teste")

        def close(self):
            events.append("closed")

    image = Image.new("RGB", (32, 32))
    with pytest.raises(failure, match="erro de teste"):
        if which == "geometry":
            generate_geometry(
                image, ".", records=records, backend_factory=lambda *a, **k: Broken()
            )
        else:
            apply_texture(
                trimesh.creation.box(), image, records=records, processor_factory=Broken
            )
    assert events == ["closed"]
    assert records[0]["estado"] == "falhou"


def test_cpu_postprocess_and_explicitly_disabled_texture():
    records = []
    image = Image.new("RGB", (32, 32))
    mesh, _ = prepare_print_mesh(trimesh.creation.box(), size=80, records=records)
    assert mesh.is_volume and mesh.extents[2] == pytest.approx(80)
    assert mesh.bounds[0, 2] == pytest.approx(0)
    assert apply_texture(mesh, image, records=records) is mesh
    assert records[1]["etapa"] == "textura"
    assert records[1]["estado"] == "ignorada"


def test_model_load_failure_is_recorded_and_cleanup_runs(monkeypatch):
    from img23d.stages import geometry

    events, records = [], []

    def bad_factory(*a, **k):
        raise RuntimeError("falha na carga")

    monkeypatch.setattr(geometry, "clear_gpu", lambda: events.append("clear"))
    with pytest.raises(RuntimeError, match="carga"):
        generate_geometry(
            Image.new("RGB", (32, 32)),
            ".",
            backend_factory=bad_factory,
            records=records,
        )
    assert events == ["clear"]
    assert records[0]["estado"] == "falhou"


def test_oom_records_peak_and_memory_after_cleanup(monkeypatch):
    from types import SimpleNamespace

    from img23d import runtime
    from img23d.errors import Img23DError

    class OOM(RuntimeError):
        pass

    allocated = [0]
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 4 * 2**20,
        max_memory_reserved=lambda: 8 * 2**20,
        memory_allocated=lambda: allocated[0],
        memory_reserved=lambda: allocated[0],
        OutOfMemoryError=OOM,
    )
    monkeypatch.setattr(runtime, "torch_module", lambda: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(runtime, "clear_gpu", lambda: allocated.__setitem__(0, 0))
    records = []
    with pytest.raises(Img23DError, match="VRAM insuficiente"):
        with runtime.stage("geometria", records):
            allocated[0] = 2 * 2**20
            raise OOM()
    assert records[0]["estado"] == "falhou"
    assert records[0]["pico_vram_mib"] == 4
    assert records[0]["pico_reservado_mib"] == 8
    assert records[0]["vram_apos_mib"] == 0
