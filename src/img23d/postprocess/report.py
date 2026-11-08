import numpy as np


def report(mesh, notes=(), min_wall=0.8):
    warnings = list(notes)
    components = len(mesh.split(only_watertight=False))
    if components > 1:
        warnings.append(f"Há {components} partes desconectadas/flutuantes.")
    # Raios de dentro da superfície até a face oposta: triagem, não metrologia.
    thickness = None
    try:
        ids = np.linspace(0, len(mesh.faces) - 1, min(512, len(mesh.faces)), dtype=int)
        normals = mesh.face_normals[ids]
        epsilon = max(mesh.extents) * 1e-6
        origins = mesh.triangles_center[ids] - normals * epsilon
        locations, ray_ids, _ = mesh.ray.intersects_location(
            origins, -normals, multiple_hits=False
        )
        if len(ray_ids):
            distances = np.linalg.norm(locations - origins[ray_ids], axis=1)
            distances = distances[distances > epsilon * 2]
            if len(distances):
                thickness = round(float(distances.min()), 3)
                if thickness < min_wall:
                    warnings.append(
                        f"Possíveis paredes/partes finas: amostra mínima {thickness} mm (< {min_wall} mm)."
                    )
    except (ImportError, ValueError):
        warnings.append("Não foi possível estimar espessura por raios.")
    warnings.append(
        "Espessura amostrada não garante paredes mínimas; revise no fatiador. Balanços e suportes não são avaliados."
    )
    return {
        "watertight": bool(mesh.is_watertight),
        "solido_valido": bool(mesh.is_volume),
        "volume_cm3": round(abs(float(mesh.volume)) / 1000, 4)
        if mesh.is_volume
        else None,
        "dimensoes_mm": dict(
            zip(("largura_x", "profundidade_y", "altura_z"), map(float, mesh.extents))
        ),
        "triangulos": len(mesh.faces),
        "componentes": components,
        "espessura_minima_amostrada_mm": thickness,
        "avisos": warnings,
    }
