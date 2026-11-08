"""Adaptador CPU explícito para o contrato torchmcubes usado por TripoSR.

Não substitui a rede TSR; apenas extrai isosuperfícies sem extensão C++/CUDA.
O helper oficial inverte XYZ ao receber os vértices, por isso retornamos ZYX.
"""


def marching_cubes(level, threshold):
    import numpy as np
    import torch
    from skimage.measure import marching_cubes as extract

    vertices, faces, _, _ = extract(
        level.detach().float().cpu().numpy(), level=threshold
    )
    return (
        torch.from_numpy(np.ascontiguousarray(vertices[:, [2, 1, 0]])),
        torch.from_numpy(np.ascontiguousarray(faces.astype(np.int64))),
    )
