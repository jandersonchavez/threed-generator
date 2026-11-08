import math
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from ..errors import Img23DError


@dataclass
class Options:
    smooth: int = 0
    target_faces: int | None = None
    cut_base: float = 0.0  # fração da altura original
    voxel_resolution: int = 128


def clean(mesh):
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise Img23DError("Malha vazia ou com coordenadas inválidas.")
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def largest(mesh):
    parts = mesh.split(only_watertight=False)
    if not len(parts):
        raise Img23DError("Não há componentes válidos na malha.")
    return max(parts, key=lambda part: part.area), len(parts) - 1


def voxel_repair(mesh, resolution):
    # Grade limitada no maior eixo; evita crescimento cúbico não controlado.
    if not 32 <= resolution <= 256:
        raise Img23DError("Resolução de reparo deve estar entre 32 e 256.")
    pitch = float(max(mesh.extents)) / resolution
    if pitch <= 0:
        raise Img23DError("A malha não possui dimensões válidas.")
    voxels = mesh.voxelized(pitch)
    # Fecha pequenas lacunas antes de preencher o interior; perde detalhes < pitch.
    matrix = np.pad(voxels.matrix, 2)
    matrix = ndimage.binary_closing(matrix, iterations=1)
    matrix = ndimage.binary_fill_holes(matrix)
    # Um campo binário pode produzir contatos diagonais não manifold. Uma
    # transição subvoxel evita vértices coincidentes na isosuperfície.
    field = ndimage.gaussian_filter(matrix.astype(np.float32), sigma=0.65)
    vertices, faces, _, _ = measure.marching_cubes(
        field, level=0.501, spacing=(pitch, pitch, pitch), allow_degenerate=False
    )
    repaired = trimesh.Trimesh(vertices, faces)
    repaired.apply_translation(voxels.transform[:3, 3] - 2 * pitch)
    return clean(repaired)


def cut_flat(mesh, level):
    """Interseção com semiespaço limitado; Manifold cria tampa robusta."""
    padding = float(max(mesh.extents))
    lower = mesh.bounds[0].copy() - padding
    upper = mesh.bounds[1].copy() + padding
    lower[2] = level
    box = trimesh.creation.box(bounds=np.array([lower, upper]))
    result = trimesh.boolean.intersection([mesh, box], engine="manifold")
    if not len(result.faces):
        raise Img23DError(
            "O corte da base não produziu um sólido. Reduza a fração de corte."
        )
    return result


def process(mesh, options=None):
    opt = options or Options()
    if not 0 <= opt.smooth <= 20:
        raise Img23DError("Suavização deve ter entre 0 e 20 iterações.")
    if not 0 <= opt.cut_base < 0.5:
        raise Img23DError("Corte da base deve ser uma fração entre 0 e 0,5.")
    if opt.target_faces is not None and opt.target_faces < 100:
        raise Img23DError("A decimação requer pelo menos 100 triângulos.")
    mesh = clean(mesh)
    mesh, removed = largest(mesh)
    notes = []
    if removed:
        notes.append(f"{removed} componente(s) flutuante(s) removido(s).")
    trimesh.repair.fill_holes(mesh)
    if not mesh.is_volume:
        mesh = voxel_repair(mesh, opt.voxel_resolution)
        notes.append(
            "Reconstrução por voxels aplicada: detalhes e cavidades podem ter sido alterados."
        )
    if opt.smooth:
        trimesh.smoothing.filter_taubin(mesh, iterations=opt.smooth)
    if opt.target_faces and len(mesh.faces) > opt.target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=opt.target_faces)
    mesh = clean(mesh)
    trimesh.repair.fill_holes(mesh)
    if not mesh.is_volume:
        mesh = voxel_repair(mesh, opt.voxel_resolution)
        notes.append(
            "Reconstrução por voxels após suavização/decimação para recuperar o sólido."
        )
    level = None
    if opt.cut_base:
        level = mesh.bounds[0, 2] + mesh.extents[2] * opt.cut_base
        mesh = cut_flat(mesh, level)
    mesh = clean(mesh)
    mesh, more_removed = largest(mesh)
    if more_removed:
        notes.append(
            f"{more_removed} componente(s) adicional(is) removido(s) após reparo/corte."
        )
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh, multibody=True)
    if not mesh.is_volume:
        mesh = voxel_repair(mesh, opt.voxel_resolution)
        notes.append(
            "Reconstrução por voxels após corte/limpeza para recuperar o sólido."
        )
        # Voxels alteram o plano: recortar e tampar novamente preserva base plana.
        if level is not None:
            mesh = cut_flat(mesh, level)
        mesh = clean(mesh)
        mesh, final_removed = largest(mesh)
        if final_removed:
            notes.append(
                f"{final_removed} componente(s) removido(s) na recuperação final."
            )
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh, multibody=True)
    if not mesh.is_volume:
        raise Img23DError(
            "Não foi possível obter um sólido fechado com volume positivo. Tente outra resolução de reparo/imagem."
        )
    return mesh, notes


def scale_mm(mesh, size, axis="height"):
    if not math.isfinite(size) or size <= 0:
        raise Img23DError("A dimensão final em mm deve ser positiva e finita.")
    if axis not in ("height", "width"):
        raise Img23DError("Eixo deve ser height ou width.")
    index = 2 if axis == "height" else 0
    if mesh.extents[index] <= 1e-10:
        raise Img23DError("A malha não possui dimensão no eixo escolhido.")
    result = mesh.copy()
    result.apply_scale(size / result.extents[index])
    center = result.bounds.mean(axis=0)
    result.apply_translation([-center[0], -center[1], -result.bounds[0, 2]])
    result.units = "mm"
    return result
