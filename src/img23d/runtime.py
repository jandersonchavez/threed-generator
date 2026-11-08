import gc
import logging
import time
from contextlib import contextmanager

from .errors import Img23DError

log = logging.getLogger("img23d")


def torch_module():
    try:
        import torch

        return torch
    except ImportError:
        return None


def clear_gpu():
    gc.collect()
    torch = torch_module()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_gpu():
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        raise Img23DError(
            "CUDA indisponível. Execute setup.ps1 e confira o driver NVIDIA."
        )
    version = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
    if version < (12, 8):
        raise Img23DError("É necessário PyTorch com CUDA 12.8 ou superior para RTX 50.")
    total = torch.cuda.get_device_properties(0).total_memory
    # Teto do allocator PyTorch: no máximo 7 GiB, deixando margem ao Windows.
    torch.cuda.set_per_process_memory_fraction(min(0.9, 7 * 2**30 / total), device=0)
    # Operação real: is_available sozinho não verifica kernels compatíveis.
    x = torch.ones((32, 32), device="cuda", dtype=torch.float16)
    (x @ x).sum().item()
    del x
    info = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": torch.cuda.get_device_capability(),
        "livre_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
    }
    clear_gpu()
    return info


@contextmanager
def stage(name, records, *, enabled=True):
    torch = torch_module()
    cuda = torch is not None and torch.cuda.is_available()
    clear_gpu()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    status = "concluida" if enabled else "ignorada"
    try:
        yield
    except BaseException as exc:
        status = "falhou"
        if cuda and isinstance(exc, torch.cuda.OutOfMemoryError):
            raise Img23DError(
                "VRAM insuficiente. Feche programas que usam GPU e reduza --resolution (ex.: 128)."
            ) from exc
        raise
    finally:
        if cuda:
            torch.cuda.synchronize()
        row = {
            "etapa": name,
            "estado": status,
            "segundos": round(time.perf_counter() - start, 3),
            "pico_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if cuda
            else 0,
            "pico_reservado_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)
            if cuda
            else 0,
        }
        clear_gpu()
        row.update(
            vram_apos_mib=round(torch.cuda.memory_allocated() / 2**20, 1)
            if cuda
            else 0,
            reservado_apos_mib=round(torch.cuda.memory_reserved() / 2**20, 1)
            if cuda
            else 0,
        )
        records.append(row)
        log.info("%s", row)
