import json
import threading
from pathlib import Path
from uuid import uuid4

from ..config import configure

_generation_lock = threading.Lock()


def build_app(cache=None):
    configure(cache)
    from ..offline import enforce_offline

    enforce_offline()
    import gradio as gr

    from ..errors import Img23DError
    from ..pipeline import run
    from ..postprocess import Options
    from ..preprocess.sam2 import detect

    def detect_objects(image):
        with _generation_lock:
            try:
                selection = detect(image, cache)
                return (
                    selection,
                    selection.overlay(),
                    gr.update(
                        choices=[str(i + 1) for i in range(len(selection.masks))],
                        value=None,
                    ),
                    None,
                    f"{len(selection.masks)} máscaras. Clique numa região ou escolha o número; confira o recorte antes de gerar.",
                )
            except Img23DError as exc:
                raise gr.Error(str(exc)) from exc
            except Exception as exc:
                import logging

                logging.exception("Falha na segmentação SAM 2")
                raise gr.Error(
                    "Falha no SAM 2. Consulte o terminal para o diagnóstico."
                ) from exc

    def choose_number(selection, number):
        if selection is None:
            raise gr.Error("Detecte as máscaras primeiro.")
        try:
            return selection.extract(
                number
            ), f"Máscara {number} selecionada. Confira o objeto isolado."
        except Img23DError as exc:
            raise gr.Error(str(exc)) from exc

    def choose_click(selection, evt: gr.SelectData):
        if selection is None:
            raise gr.Error("Detecte as máscaras primeiro.")
        try:
            number = selection.at_point(*evt.index)
            preview, message = choose_number(selection, number)
            return str(number), preview, message
        except Img23DError as exc:
            raise gr.Error(str(exc)) from exc

    def reset_selection(image, mode):
        return (
            None,
            None,
            gr.update(choices=[], value=None),
            None,
            "Seleção limpa. No modo SAM 2, detecte e escolha o objeto.",
            gr.update(interactive=mode == "sam2"),
        )

    def generate(
        image, backend, height, cut, smooth, resolution, mode, selection, number
    ):
        if image is None:
            raise gr.Error("Envie uma imagem primeiro.")
        with _generation_lock:
            try:
                folder = Path("outputs") / uuid4().hex
                metadata = {"modo": mode}
                if mode == "sam2":
                    if selection is None:
                        raise Img23DError(
                            "Detecte as máscaras e selecione o objeto antes de gerar."
                        )
                    selection.validate_source(image)
                    image = selection.extract(number)
                    metadata.update(mascara=int(number), etapas=selection.metrics)
                elif mode != "rembg":
                    raise Img23DError("Modo de seleção desconhecido.")
                stl, mf, result = run(
                    image,
                    folder,
                    backend=backend,
                    size=height,
                    cache=cache,
                    resolution=int(resolution),
                    options=Options(smooth=2 if smooth else 0, cut_base=cut),
                    remove_background=mode == "rembg",
                )
                result["segmentacao"] = metadata
                if mode == "sam2":
                    image.save(folder / "objeto_selecionado.png")
                (folder / "relatorio.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return str(stl.resolve()), str(stl.resolve()), str(mf.resolve()), result
            except Img23DError as exc:
                raise gr.Error(str(exc)) from exc
            except Exception as exc:
                import logging

                logging.exception("Falha na geração")
                raise gr.Error(
                    "Falha na geração. Consulte o terminal para o diagnóstico."
                ) from exc

    with gr.Blocks(
        title="Imagem para impressão 3D",
        analytics_enabled=False,
        theme=gr.themes.Base(
            font=["Arial", "sans-serif"], font_mono=["Consolas", "monospace"]
        ),
    ) as app:
        gr.Markdown(
            "# Imagem → impressão 3D\nProcessamento local, sem textura. Revise o modelo no fatiador antes de imprimir."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Imagem", image_mode="RGBA")
                selection = gr.State(value=None, time_to_live=1200)
                mode = gr.Radio(
                    choices=[
                        ("Rápido (rembg)", "rembg"),
                        ("Escolher objeto (SAM 2)", "sam2"),
                    ],
                    value="rembg",
                    label="Seleção do objeto",
                )
                detect_button = gr.Button(
                    "Detectar máscaras (SAM 2)", interactive=False
                )
                masks = gr.Image(
                    type="pil",
                    label="Máscaras numeradas — clique no objeto",
                    interactive=False,
                    sources=[],
                )
                number = gr.Dropdown(choices=[], value=None, label="Número da máscara")
                preview = gr.Image(
                    type="pil",
                    label="Objeto selecionado",
                    interactive=False,
                    sources=[],
                    image_mode="RGBA",
                )
                status = gr.Markdown(
                    "Modo rápido: o rembg remove o fundo ao gerar. No SAM 2, detecte e escolha o objeto primeiro."
                )
                backend = gr.Dropdown(
                    ["hunyuan", "triposr"], value="hunyuan", label="Modelo"
                )
                height = gr.Number(value=80, minimum=1, label="Altura final (mm)")
                cut = gr.Slider(
                    0, 0.3, value=0, step=0.01, label="Fração da base a cortar"
                )
                smooth = gr.Checkbox(label="Suavização leve")
                resolution = gr.Dropdown(
                    [128, 192, 256], value=192, label="Resolução da geometria"
                )
                button = gr.Button("Gerar", variant="primary")
            with gr.Column():
                viewer = gr.Model3D(label="Geometria")
                stl = gr.File(label="Download STL")
                mf = gr.File(label="Download 3MF")
                result = gr.JSON(label="Relatório de imprimibilidade")
        button.click(
            generate,
            [image, backend, height, cut, smooth, resolution, mode, selection, number],
            [viewer, stl, mf, result],
            concurrency_limit=1,
            concurrency_id="gpu",
        )
        detect_button.click(
            detect_objects,
            [image],
            [selection, masks, number, preview, status],
            concurrency_limit=1,
            concurrency_id="gpu",
        )
        masks.select(choose_click, [selection], [number, preview, status])
        number.input(choose_number, [selection, number], [preview, status])
        reset_outputs = [selection, masks, number, preview, status, detect_button]
        image.change(reset_selection, [image, mode], reset_outputs)
        mode.change(reset_selection, [image, mode], reset_outputs)
    return app.queue(default_concurrency_limit=1)


def launch(cache=None):
    build_app(cache).launch(server_name="127.0.0.1", share=False, inbrowser=True)
