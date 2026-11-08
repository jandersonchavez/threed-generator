from abc import ABC, abstractmethod

import trimesh
from PIL import Image


class Backend(ABC):
    @abstractmethod
    def generate(self, image: Image.Image) -> trimesh.Trimesh:
        """Gera uma malha sem unidades; não exporta nem altera escala."""

    def close(self) -> None:
        """Libera recursos persistentes, se houver; compatível com backends existentes.

        Hunyuan/TripoSR já descarregam seus modelos no finally de generate.
        Um backend futuro que retenha modelos deve sobrescrever este método.
        """
