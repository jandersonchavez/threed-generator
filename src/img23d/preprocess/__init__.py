from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from ..errors import Img23DError


def prepare(image, cache, remove_background=True, white=False, margin=0.15):
    if not 0 <= margin < 0.45:
        raise Img23DError("Margem deve estar entre 0 e 0,45.")
    try:
        if isinstance(image, Image.Image):
            im = image.copy()
        else:
            with Image.open(image) as source:
                source.load()
                im = source.copy()
        im = ImageOps.exif_transpose(im).convert("RGBA")
        im.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise Img23DError(
            "Imagem inválida ou grande demais. Use PNG, JPEG ou WebP."
        ) from exc
    if remove_background and im.getchannel("A").getextrema() == (255, 255):
        if not (Path(cache) / "rembg" / "u2netp.onnx").is_file():
            raise Img23DError(
                "Peso rembg ausente no cache. Execute python -m img23d.download."
            )
        from rembg import new_session, remove

        session = new_session("u2netp", providers=["CPUExecutionProvider"])
        im = remove(im, session=session)
        del session
    bbox = im.getchannel("A").point(lambda x: 255 if x > 16 else 0).getbbox()
    if bbox is None:
        raise Img23DError("A imagem está vazia após remover o fundo.")
    im = im.crop(bbox)
    factor = 512 * (1 - 2 * margin) / max(im.size)
    im = im.resize(
        (max(1, round(im.width * factor)), max(1, round(im.height * factor))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
    canvas.alpha_composite(im, ((512 - im.width) // 2, (512 - im.height) // 2))
    if white:
        background = Image.new("RGBA", canvas.size, "white")
        background.alpha_composite(canvas)
        return background.convert("RGB")
    return canvas
