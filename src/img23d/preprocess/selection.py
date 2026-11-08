"""Máscaras em CPU e seleção de objeto, independentes dos backends 3D."""

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

from ..errors import Img23DError


def selection_image(image):
    if image is None:
        raise Img23DError("Envie uma imagem primeiro.")
    result = ImageOps.exif_transpose(image).convert("RGBA")
    result.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return result


def fingerprint(image):
    image = selection_image(image)
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


@dataclass
class Selection:
    image: Image.Image
    masks: list[np.ndarray]
    source_hash: str
    metrics: list[dict]

    def validate_source(self, image):
        if fingerprint(image) != self.source_hash:
            raise Img23DError(
                "A imagem mudou. Detecte as máscaras novamente antes de gerar."
            )

    def extract(self, number):
        try:
            index = int(number) - 1
            valid = float(number) == int(number)
        except (TypeError, ValueError, OverflowError):
            valid, index = False, -1
        if not valid or not 0 <= index < len(self.masks):
            raise Img23DError(
                "Escolha o número de uma máscara detectada antes de gerar."
            )
        rgba = np.array(self.image)
        rgba[:, :, 3] = np.where(self.masks[index], rgba[:, :, 3], 0)
        if not rgba[:, :, 3].any():
            raise Img23DError("A máscara escolhida não contém pixels visíveis.")
        return Image.fromarray(rgba)

    def at_point(self, x, y):
        x, y = int(x), int(y)
        if not (0 <= x < self.image.width and 0 <= y < self.image.height):
            raise Img23DError("Clique dentro da imagem com máscaras.")
        matches = [i for i, mask in enumerate(self.masks) if mask[y, x]]
        if not matches:
            raise Img23DError(
                "Nenhuma máscara nesse ponto. Escolha outro ponto ou um número."
            )
        # Máscaras aninhadas: a menor região sob o cursor tem prioridade.
        return min(matches, key=lambda i: np.count_nonzero(self.masks[i])) + 1

    def overlay(self):
        background = Image.new("RGBA", self.image.size, "white")
        background.alpha_composite(self.image)
        pixels = np.array(background.convert("RGB"), dtype=np.float32)
        colors = [
            (255, 99, 71),
            (40, 180, 255),
            (90, 220, 100),
            (235, 190, 40),
            (190, 100, 240),
        ]
        for i, mask in enumerate(self.masks):
            pixels[mask] = (
                pixels[mask] * 0.55 + np.array(colors[i % len(colors)]) * 0.45
            )
        result = Image.fromarray(pixels.astype(np.uint8))
        draw = ImageDraw.Draw(result)
        font = ImageFont.load_default(size=20)
        for i, mask in enumerate(self.masks):
            y, x = np.unravel_index(
                np.argmax(ndimage.distance_transform_edt(mask)), mask.shape
            )
            # Mantém os números legíveis mesmo em regiões na borda da imagem.
            x = int(
                np.clip(
                    x,
                    min(20, result.width // 2),
                    max(result.width // 2, result.width - 20),
                )
            )
            y = int(
                np.clip(
                    y,
                    min(16, result.height // 2),
                    max(result.height // 2, result.height - 16),
                )
            )
            text = str(i + 1)
            box = draw.textbbox((x, y), text, anchor="mm", font=font)
            draw.rectangle(
                (box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill="black"
            )
            draw.text((x, y), text, fill="white", anchor="mm", font=font)
        return result


def make_selection(image, records, metrics=()):
    source = selection_image(image)
    visible = np.array(source.getchannel("A")) > 16
    masks = []
    minimum = max(32, round(source.width * source.height * 0.005))
    for record in sorted(records, key=lambda row: row["area"], reverse=True):
        mask = np.asarray(record["segmentation"], dtype=bool)
        if mask.shape != visible.shape:
            raise Img23DError("SAM 2 retornou máscara com dimensões incompatíveis.")
        mask = mask & visible
        if minimum <= mask.sum() and not any(
            np.array_equal(mask, old) for old in masks
        ):
            masks.append(mask.copy())
        if len(masks) >= 32:
            break
    if not masks:
        raise Img23DError(
            "SAM 2 não encontrou máscaras úteis. Tente outra imagem ou o modo rápido rembg."
        )
    return Selection(source, masks, fingerprint(image), list(metrics))
