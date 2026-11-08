import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from ..errors import Img23DError
from ..runtime import check_gpu, clear_gpu
from .base import Backend


class TripoSRBackend(Backend):
    def __init__(self, cache, resolution=192, steps=30, seed=12345):
        self.cache, self.resolution = Path(cache), resolution

    def generate(self, image):
        check_gpu()
        import torch

        vendor = self.cache / "code" / "TripoSR"
        if not (vendor / "tsr" / "system.py").is_file():
            raise Img23DError("Código TripoSR ausente. Execute setup.ps1 -WithTripoSR.")
        sys.path.insert(0, str(vendor))
        try:
            from tsr.system import TSR
        except ImportError as exc:
            raise Img23DError(
                "TripoSR não carregou. Execute setup.ps1 -WithTripoSR para preparar código e dependências."
            ) from exc
        folder = self.cache / "triposr"
        if not all((folder / name).is_file() for name in ("config.yaml", "model.ckpt")):
            raise Img23DError(
                "Pesos TripoSR ausentes. Execute python -m img23d.download --triposr."
            )
        model, codes = None, None
        try:
            model = TSR.from_pretrained(
                str(folder), config_name="config.yaml", weight_name="model.ckpt"
            )
            model.eval().to("cuda")
            model.renderer.set_chunk_size(4096)
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
            background.alpha_composite(rgba)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                codes = model([background.convert("RGB")], device="cuda")
            # Extração em CPU FP32 evita grade cúbica e torchmcubes na GPU sm_120.
            codes = codes.float().cpu()
            model.to("cpu")
            clear_gpu()
            with torch.inference_mode():
                mesh = model.extract_mesh(
                    codes, has_vertex_color=False, resolution=self.resolution
                )[0]
            if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                raise Img23DError("TripoSR produziu uma malha vazia ou inválida.")
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            )
            return mesh
        finally:
            if model is not None:
                model.to("cpu")
            del model, codes
            clear_gpu()
