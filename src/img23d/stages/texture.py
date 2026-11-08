from abc import ABC, abstractmethod

import trimesh
from PIL import Image

from ..runtime import clear_gpu, stage


class TextureProcessor(ABC):
    """Contrato de textura independente da interface Backend de geometria.

    A fábrica deve carregar recursos somente ao ser chamada; apply devolve uma
    malha com dados CPU e close deve descarregar qualquer modelo persistente.
    """

    @abstractmethod
    def apply(self, mesh: trimesh.Trimesh, image: Image.Image) -> trimesh.Trimesh:
        pass

    def close(self) -> None:
        pass


def apply_texture(mesh, image, *, records=None, processor_factory=None):
    """Sem fábrica: etapa explicitamente ignorada. Modos concretos vêm na fase 4."""
    records = records if records is not None else []
    processor = None
    with stage("textura", records, enabled=processor_factory is not None):
        if processor_factory is None:
            return mesh
        try:
            processor = processor_factory()
            return processor.apply(mesh, image)
        finally:
            try:
                if processor is not None:
                    processor.close()
            finally:
                del processor
                clear_gpu()
