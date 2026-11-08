"""SAM 2.1 Tiny com pesos locais e liberação explícita da GPU."""

from pathlib import Path

import numpy as np
from PIL import Image

from ..config import configure
from ..errors import Img23DError
from ..offline import enforce_offline
from ..runtime import check_gpu, clear_gpu, stage
from .selection import make_selection, selection_image

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "sam2.1_hiera_tiny.pt"


def detect(image, cache=None):
    root = configure(cache)
    enforce_offline()
    source = selection_image(image)
    checkpoint = Path(root) / "sam2" / CHECKPOINT
    if not checkpoint.is_file():
        raise Img23DError(
            "SAM 2 não está instalado no cache. Execute setup.ps1 -WithSAM2 antes de ficar offline."
        )
    import torch

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise Img23DError("Pacote SAM-2 ausente. Execute setup.ps1 -WithSAM2.") from exc
    check_gpu()
    model, generator = None, None
    metrics = []
    try:
        with stage("segmentacao_sam2", metrics):
            model = build_sam2(
                MODEL_CONFIG, str(checkpoint), device="cpu", apply_postprocessing=False
            )
            model.eval().to("cuda")
            generator = SAM2AutomaticMaskGenerator(
                model,
                points_per_side=32,
                points_per_batch=8,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.9,
                crop_n_layers=0,
                min_mask_region_area=0,
                output_mode="binary_mask",
            )
            rgb = Image.new("RGBA", source.size, "white")
            rgb.alpha_composite(source)
            # BF16 é o autocast recomendado pelo SAM 2 e suportado pela RTX 50.
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                records = generator.generate(np.array(rgb.convert("RGB")))
        return make_selection(image, records, metrics)
    finally:
        if generator is not None:
            generator.predictor.reset_predictor()
        if model is not None:
            model.to("cpu")
        del generator, model
        clear_gpu()
