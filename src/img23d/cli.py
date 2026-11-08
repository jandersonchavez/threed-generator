import argparse
import json
import logging

from .errors import Img23DError


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Imagem para STL/3MF local, sem textura."
    )
    parser.add_argument("input", nargs="?")
    dimensions = parser.add_mutually_exclusive_group()
    dimensions.add_argument("--height", type=float)
    dimensions.add_argument("--width", type=float)
    parser.add_argument("--backend", choices=["hunyuan", "triposr"], default="hunyuan")
    parser.add_argument("--out", default="saida")
    parser.add_argument("--cache")
    parser.add_argument(
        "--resolution", type=int, default=192, choices=range(64, 257), metavar="64..256"
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--target-faces", type=int)
    parser.add_argument(
        "--cut-base", type=float, default=0, help="Fração da altura a cortar, ex.: 0.03"
    )
    parser.add_argument("--voxel-resolution", type=int, default=128)
    parser.add_argument("--keep-background", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--check-gpu", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from .config import configure

    configure(args.cache)
    try:
        if args.check_gpu:
            from .runtime import check_gpu

            print(json.dumps(check_gpu(), ensure_ascii=False, indent=2))
            return 0
        if args.ui:
            from .ui.app import launch

            launch(args.cache)
            return 0
        if not args.input:
            parser.error("Informe uma imagem ou use --ui/--check-gpu.")
        if args.steps < 1 or args.steps > 100:
            parser.error("--steps deve estar entre 1 e 100.")
        from .pipeline import run
        from .postprocess import Options

        stl, mf, result = run(
            args.input,
            args.out,
            args.backend,
            args.width
            if args.width is not None
            else (args.height if args.height is not None else 80),
            "width" if args.width is not None else "height",
            args.cache,
            args.resolution,
            args.steps,
            args.seed,
            Options(
                args.smooth, args.target_faces, args.cut_base, args.voxel_resolution
            ),
            not args.keep_background,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Arquivos: {stl}\n{mf}")
        return 0
    except (Img23DError, OSError, ImportError) as exc:
        logging.error("Falha: %s", exc)
        return 1
    except Exception as exc:
        logging.exception(
            "Falha inesperada (%s). Consulte o diagnóstico acima.", type(exc).__name__
        )
        return 1
