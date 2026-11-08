import json
import socket
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest
import trimesh
from PIL import Image

from img23d.export import export_mesh
from img23d.postprocess.report import report
from img23d.preprocess import prepare


def test_3mf_coordinates_and_stl_roundtrip(tmp_path):
    mesh = trimesh.creation.box([10, 20, 30])
    stl, mf = export_mesh(mesh, tmp_path)
    loaded = trimesh.load_mesh(stl)
    assert loaded.is_watertight
    assert np.allclose(loaded.extents, mesh.extents)
    with ZipFile(mf) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
        assert root.attrib["unit"] == "millimeter"
        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        vertices = root.findall(".//m:vertex", ns)
        triangles = root.findall(".//m:triangle", ns)
        assert len(vertices) == len(mesh.vertices)
        rebuilt = trimesh.Trimesh(
            [[float(v.attrib[c]) for c in "xyz"] for v in vertices],
            [[int(t.attrib[c]) for c in ("v1", "v2", "v3")] for t in triangles],
        )
        assert rebuilt.is_volume
        assert rebuilt.volume == pytest.approx(mesh.volume)


def test_transparent_preprocessing(tmp_path):
    image = Image.new("RGBA", (100, 60))
    image.paste((255, 0, 0, 255), (10, 10, 30, 50))
    result = prepare(image, tmp_path)
    assert result.size == (512, 512)
    bbox = result.getchannel("A").getbbox()
    assert abs((bbox[0] + bbox[2]) / 2 - 256) < 2


def test_report_volume_and_thin_wall():
    result = report(trimesh.creation.box([10, 10, 0.4]))
    assert result["volume_cm3"] == pytest.approx(0.04)
    assert result["espessura_minima_amostrada_mm"] < 0.8
    assert any("finas" in w for w in result["avisos"])


def test_pipeline_without_network_with_synthetic_backend(tmp_path, monkeypatch):
    from img23d import pipeline

    class Synthetic:
        def generate(self, image):
            return trimesh.creation.icosphere(subdivisions=1)

    monkeypatch.setattr(pipeline, "create_backend", lambda *a, **k: Synthetic())

    def no_network(*a, **k):
        raise AssertionError("Acesso de rede inesperado")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    image = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
    stl, mf, result = pipeline.run(image, tmp_path / "out", cache=tmp_path / "cache")
    assert stl.is_file() and mf.is_file()
    assert result["dimensoes_mm"]["altura_z"] == pytest.approx(80)
    assert (
        len(
            json.loads((stl.parent / "relatorio.json").read_text(encoding="utf-8"))[
                "etapas"
            ]
        )
        == 6
    )
