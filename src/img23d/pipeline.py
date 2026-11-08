import json
from pathlib import Path

from .backends import create_backend
from .config import configure
from .export import export_mesh
from .postprocess.report import report
from .preprocess import prepare
from .runtime import stage
from .stages import apply_texture, generate_geometry, prepare_print_mesh


def run(
    image,
    out,
    backend="hunyuan",
    size=80,
    axis="height",
    cache=None,
    resolution=192,
    steps=30,
    seed=12345,
    options=None,
    remove_background=True,
):
    from .offline import enforce_offline

    enforce_offline()
    root = configure(cache)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with stage("preprocessamento", records):
        image = prepare(image, root, remove_background=remove_background)
        image.save(out / "preprocessada.png")
    mesh = generate_geometry(
        image,
        root,
        backend,
        resolution=resolution,
        steps=steps,
        seed=seed,
        records=records,
        backend_factory=create_backend,
    )
    mesh, notes = prepare_print_mesh(
        mesh, size=size, axis=axis, options=options, records=records
    )
    mesh = apply_texture(mesh, image, records=records)
    with stage("relatorio", records):
        result = report(mesh, notes)
    with stage("exportacao", records):
        stl, mf = export_mesh(mesh, out)
    result.update(backend=backend, seed=seed, textura="nenhuma", etapas=records)
    (out / "relatorio.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stl, mf, result
