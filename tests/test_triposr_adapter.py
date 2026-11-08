import numpy as np
import pytest


def test_cpu_marching_cubes_preserves_upstream_axis_contract():
    torch = pytest.importorskip("torch")
    from img23d.backends.marching_cubes import marching_cubes

    x, y, z = np.mgrid[:32, :40, :48]
    field = 1 - ((x - 15) / 7) ** 2 - ((y - 18) / 10) ** 2 - ((z - 22) / 13) ** 2
    vertices, faces = marching_cubes(torch.from_numpy(field), 0)
    # Mesmo passo que MarchingCubeHelper.forward oficial executa.
    xyz = vertices.numpy()[:, [2, 1, 0]]
    assert np.allclose(xyz.mean(axis=0), [15, 18, 22], atol=0.3)
    assert np.allclose(np.ptp(xyz, axis=0), [14, 20, 26], atol=0.3)
    import trimesh

    assert trimesh.Trimesh(xyz, faces.numpy()).is_watertight
