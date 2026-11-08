from pathlib import Path

import trimesh

from ..errors import Img23DError
from ..runtime import check_gpu, clear_gpu
from .base import Backend


class HunyuanBackend(Backend):
    def __init__(self, cache, resolution=192, steps=30, seed=12345):
        self.cache = Path(cache)
        self.resolution, self.steps, self.seed = resolution, steps, seed

    def generate(self, image):
        check_gpu()
        import torch

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
        except ImportError as exc:
            raise Img23DError(
                "Hunyuan3D ausente ou dependência incompatível. Execute setup.ps1."
            ) from exc
        folder = self.cache / "hunyuan" / "hunyuan3d-dit-v2-mini"
        weights = folder / "model.fp16.safetensors"
        config = folder / "config.yaml"
        if not weights.is_file() or not config.is_file():
            raise Img23DError(
                "Pesos Hunyuan ausentes. Execute python -m img23d.download antes de ficar offline."
            )
        pipeline = None
        try:
            # from_single_file evita qualquer resolução remota implícita.
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                str(weights),
                str(config),
                device="cpu",
                dtype=torch.float16,
                use_safetensors=True,
            )
            # Nesta revisão upstream, offload consulta `components`, mas o
            # construtor não cria o registro. São os três módulos do próprio
            # model_cpu_offload_seq, sem módulos de textura.
            if not hasattr(pipeline, "components"):
                pipeline.components = {
                    name: getattr(pipeline, name)
                    for name in ("conditioner", "model", "vae")
                }
            for component in pipeline.components.values():
                component.eval()
            pipeline.enable_model_cpu_offload(device="cuda")
            # O upstream usa self.device também para latentes/timesteps.
            # Alterar só esse campo preserva os hooks e os pesos em CPU.
            pipeline.device = torch.device("cuda")
            pipeline.vae.surface_extractor = SurfaceExtractors["mc"]()
            with torch.inference_mode():
                mesh = pipeline(
                    image=image,
                    num_inference_steps=self.steps,
                    octree_resolution=self.resolution,
                    num_chunks=4000,
                    generator=torch.Generator(device="cuda").manual_seed(self.seed),
                    output_type="trimesh",
                )[0]
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                raise Img23DError(
                    "Hunyuan não produziu uma superfície. Tente outra imagem."
                )
            # Os modelos usam Y para cima; a aplicação usa Z para impressão.
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    3.141592653589793 / 2, [1, 0, 0]
                )
            )
            return mesh
        finally:
            if pipeline is not None:
                for hook in getattr(pipeline, "_all_hooks", []):
                    hook.offload()
                pipeline.to("cpu")
            del pipeline
            clear_gpu()
