import os
from pathlib import Path


def configure(cache=None, offline=True):
    root = Path(cache or os.environ.get("IMG23D_CACHE", "models")).resolve()
    os.environ["IMG23D_CACHE"] = str(root)
    os.environ["HF_HOME"] = str(root / "hf")
    os.environ["HF_HUB_CACHE"] = str(root / "hf" / "hub")
    os.environ["U2NET_HOME"] = str(root / "rembg")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    return root
