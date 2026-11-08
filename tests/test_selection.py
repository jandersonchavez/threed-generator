import numpy as np
import pytest
from PIL import Image

from img23d.errors import Img23DError
from img23d.preprocess.selection import make_selection


def sample():
    image = Image.new("RGBA", (100, 80), "white")
    big = np.zeros((80, 100), dtype=bool)
    big[5:75, 5:95] = True
    small = np.zeros_like(big)
    small[20:50, 40:70] = True
    return image, make_selection(
        image,
        [
            {"segmentation": big, "area": big.sum()},
            {"segmentation": small, "area": small.sum()},
        ],
    )


def test_number_click_overlap_and_overlay():
    _, selection = sample()
    assert selection.at_point(50, 30) == 2
    assert selection.at_point(10, 10) == 1
    assert selection.overlay().size == (100, 80)
    rgba = np.array(selection.extract("2"))
    assert rgba[30, 50, 3] == 255
    assert rgba[10, 10, 3] == 0
    with pytest.raises(Img23DError, match="Nenhuma máscara"):
        selection.at_point(0, 0)
    for bad in (None, 0, 3, -1, 1.5, "x"):
        with pytest.raises(Img23DError):
            selection.extract(bad)


def test_changed_image_rejected_and_existing_alpha_preserved():
    image, selection = sample()
    selection.validate_source(image.copy())
    image.putpixel((0, 0), (0, 0, 0, 0))
    with pytest.raises(Img23DError, match="imagem mudou"):
        selection.validate_source(image)
    mask = np.ones((80, 100), dtype=bool)
    selected = make_selection(image, [{"segmentation": mask, "area": mask.sum()}])
    assert np.array(selected.extract(1))[0, 0, 3] == 0


def test_no_useful_masks_and_deduplication():
    image, selection = sample()
    with pytest.raises(Img23DError, match="não encontrou"):
        make_selection(image, [])
    record = {"segmentation": selection.masks[0], "area": selection.masks[0].sum()}
    assert len(make_selection(image, [record, record]).masks) == 1


def test_ui_requires_selection_and_bypasses_rembg(tmp_path, monkeypatch):
    import gradio as gr

    from img23d import pipeline
    from img23d.ui.app import build_app

    image, selection = sample()
    observed = {}

    def fake_run(image, folder, **kwargs):
        observed.update(kwargs)
        observed["image"] = image
        folder.mkdir(parents=True)
        return folder / "model.stl", folder / "model.3mf", {}

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    app = build_app(tmp_path / "models")
    generate = next(f.fn for f in app.fns.values() if f.fn.__name__ == "generate")
    choose_click = next(
        f.fn for f in app.fns.values() if f.fn.__name__ == "choose_click"
    )
    event = gr.SelectData(None, {"index": (50, 30), "value": None})
    number, _, _ = choose_click(selection, event)
    assert number == "2"
    with pytest.raises(gr.Error):
        generate(image, "hunyuan", 80, 0, False, 128, "sam2", None, None)
    generate(image, "hunyuan", 80, 0, False, 128, "sam2", selection, "2")
    assert observed["remove_background"] is False
    assert np.array(observed["image"])[10, 10, 3] == 0
    changed = image.copy()
    changed.putpixel((0, 0), (0, 0, 0, 0))
    with pytest.raises(gr.Error):
        generate(changed, "hunyuan", 80, 0, False, 128, "sam2", selection, "2")
    generate(image, "hunyuan", 80, 0, False, 128, "rembg", None, None)
    assert observed["remove_background"] is True


@pytest.mark.parametrize("fail", [False, True])
def test_sam_releases_predictor_and_model_even_on_failure(tmp_path, monkeypatch, fail):
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    pytest.importorskip("sam2")
    from img23d.preprocess import sam2 as integration

    builder = importlib.import_module("sam2.build_sam")
    automatic = importlib.import_module("sam2.automatic_mask_generator")
    events = []

    class Model:
        def eval(self):
            return self

        def to(self, device):
            events.append(device)
            return self

    class Generator:
        def __init__(self, model, **kwargs):
            self.predictor = SimpleNamespace(
                reset_predictor=lambda: events.append("reset")
            )

        def generate(self, image):
            if fail:
                raise RuntimeError("falha simulada")
            mask = np.ones(image.shape[:2], dtype=bool)
            return [{"segmentation": mask, "area": mask.sum()}]

    (tmp_path / "sam2").mkdir()
    (tmp_path / "sam2" / integration.CHECKPOINT).touch()
    monkeypatch.setattr(builder, "build_sam2", lambda *a, **k: Model())
    monkeypatch.setattr(automatic, "SAM2AutomaticMaskGenerator", Generator)
    monkeypatch.setattr(integration, "check_gpu", lambda: None)
    monkeypatch.setattr(integration, "stage", lambda *a: nullcontext())
    monkeypatch.setattr(integration, "clear_gpu", lambda: events.append("clear"))
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    if fail:
        with pytest.raises(RuntimeError, match="simulada"):
            integration.detect(Image.new("RGB", (64, 64), "white"), tmp_path)
    else:
        integration.detect(Image.new("RGB", (64, 64), "white"), tmp_path)
    assert events == ["cuda", "reset", "cpu", "clear"]
