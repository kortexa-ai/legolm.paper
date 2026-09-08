"""Portable process limits; service borrowing remains outside the paper runner."""
from __future__ import annotations

import importlib.util
import os


def configure(device="cuda"):
    if not device.startswith("cuda"):
        return
    import torch
    # Honor an operator's installed service/GPU guard when present. A clean
    # public checkout has no dependency on that optional external module.
    if importlib.util.find_spec("smarty_gpu_rails") is not None:
        import smarty_gpu_rails  # noqa: F401
    if torch.cuda.device_count()!=1:
        raise RuntimeError("Select exactly one experiment GPU with CUDA_VISIBLE_DEVICES before starting")
    expected=os.environ.get("PAPER1_EXPECTED_GPU")
    if expected and expected not in torch.cuda.get_device_name(0):
        raise RuntimeError("The visible GPU differs from PAPER1_EXPECTED_GPU")
    fraction=float(os.environ.get("PAPER1_GPU_ALLOCATOR_FRACTION",os.environ.get("SMARTY_GPU_ALLOCATOR_FRACTION","0.88")))
    floor=float(os.environ.get("PAPER1_GPU_PRELOAD_MIN_FREE_GIB",os.environ.get("SMARTY_GPU_PRELOAD_MIN_FREE_GIB","10")))
    if not 0<fraction<=.88 or not floor>=10:
        raise ValueError("Allocator fraction must be at most .88 and free-memory floor at least 10 GiB")
    torch.cuda.set_per_process_memory_fraction(fraction,0)
    free,_=torch.cuda.mem_get_info(0)
    if free/(1024**3)<floor:
        raise MemoryError(f"Preload has less than the declared {floor:g} GiB free-memory floor")
