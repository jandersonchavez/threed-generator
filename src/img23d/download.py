"""Única etapa que permite rede: aquisição explícita de pesos."""

import argparse
import hashlib
import json

from .config import configure


def download(cache=None, triposr=False, sam2=False):
    root = configure(cache, offline=False)
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        snapshot_download(
            "tencent/Hunyuan3D-2mini",
            local_dir=root / "hunyuan",
            allow_patterns=[
                "hunyuan3d-dit-v2-mini/config.yaml",
                "hunyuan3d-dit-v2-mini/model.fp16.safetensors",
            ],
        )
        if triposr:
            snapshot_download(
                "stabilityai/TripoSR",
                local_dir=root / "triposr",
                allow_patterns=["config.yaml", "model.ckpt"],
            )
            # Dependência indireta: tokenizer TripoSR busca este config no Hub.
            hf_hub_download("facebook/dino-vitb16", "config.json")
        if sam2:
            hf_hub_download(
                "facebook/sam2.1-hiera-tiny",
                "sam2.1_hiera_tiny.pt",
                local_dir=root / "sam2",
            )
        from rembg import new_session

        session = new_session("u2netp", providers=["CPUExecutionProvider"])
        del session
        manifest = {}
        for path in root.rglob("*"):
            if "code" in path.relative_to(root).parts:
                continue
            if path.is_file() and path.suffix in (
                ".safetensors",
                ".ckpt",
                ".onnx",
                ".yaml",
                ".pt",
            ):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                manifest[str(path.relative_to(root))] = {
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
        (root / "pesos.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        from .errors import Img23DError

        raise Img23DError(
            "Download de pesos falhou. Confira conexão, espaço em disco e acesso aos repositórios; execute novamente para retomar."
        ) from exc
    finally:
        configure(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache")
    parser.add_argument("--triposr", action="store_true")
    parser.add_argument(
        "--sam2", action="store_true", help="Inclui o peso SAM 2.1 Tiny"
    )
    parser.add_argument(
        "--sam2-only", action="store_true", help="Baixa somente o SAM 2.1 Tiny"
    )
    args = parser.parse_args()
    try:
        if args.sam2_only:
            root = configure(args.cache, offline=False)
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    "facebook/sam2.1-hiera-tiny",
                    "sam2.1_hiera_tiny.pt",
                    local_dir=root / "sam2",
                )
            finally:
                configure(root)
        else:
            download(args.cache, args.triposr, args.sam2)
    except Exception as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print("Pesos prontos. A geração usará somente arquivos locais.")


if __name__ == "__main__":
    main()
