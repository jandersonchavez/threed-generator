import numpy as np
import pytest
import trimesh

from img23d.errors import Img23DError
from img23d.postprocess import Options, process, scale_mm, voxel_repair


def test_sphere_single_face_hole():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    assert not mesh.is_watertight
    result, _ = process(mesh)
    assert result.is_watertight and result.is_volume


def test_large_hole_uses_voxels():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.update_faces(mesh.triangles_center[:, 2] < 0.85)
    result, notes = process(mesh, Options(voxel_resolution=48))
    assert result.is_watertight and result.is_volume
    assert any("voxels" in note for note in notes)


def test_components_and_duplicate_faces():
    main = trimesh.creation.box([2, 2, 2])
    small = trimesh.creation.box([0.2, 0.2, 0.2])
    small.apply_translation([10, 0, 0])
    mesh = trimesh.util.concatenate([main, small])
    result, notes = process(mesh)
    assert result.is_watertight
    assert np.allclose(result.extents, [2, 2, 2])
    assert notes


def test_cut_scale_decimate_smooth():
    mesh, _ = process(
        trimesh.creation.icosphere(subdivisions=3),
        Options(smooth=2, target_faces=800, cut_base=0.05),
    )
    result = scale_mm(mesh, 80)
    assert result.is_volume
    assert result.extents[2] == pytest.approx(80)
    assert result.bounds[0, 2] == pytest.approx(0)
    assert np.allclose(result.bounds.mean(axis=0)[:2], [0, 0])


def test_width_and_invalid_scale():
    mesh = trimesh.creation.box([2, 3, 4])
    assert scale_mm(mesh, 50, "width").extents[0] == pytest.approx(50)
    for value in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(Img23DError):
            scale_mm(mesh, value)


def test_duplicate_vertices_faces_and_degenerate_triangle():
    box = trimesh.creation.box()
    vertices = np.vstack([box.vertices, box.vertices])
    faces = np.vstack([box.faces, box.faces + len(box.vertices), [0, 0, 1]])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    result, _ = process(mesh)
    assert result.is_volume
    assert len(result.faces) == 12


def test_voxel_repair_diagonal_contact():
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([1, 1, 0])
    repaired = voxel_repair(trimesh.util.concatenate([first, second]), 48)
    assert repaired.is_watertight
    assert repaired.is_volume
