"""
utils/device.py
---------------
Single source of truth for device selection across the project.

Priority order:
  1. CUDA (NVIDIA/AMD via ROCm)  - picks the GPU with the MOST free VRAM
  2. MPS  (Apple Silicon)
  3. CPU  (fallback - always works, just slower)

IMPORTANT: Do NOT use a module-level DEVICE singleton.
Call get_device() at the point of use (inside functions / __init__ methods).
A singleton set at import time can silently return CPU on systems where CUDA
initialises after the first import, or pick the wrong GPU on multi-GPU machines.
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Minimum free VRAM (bytes) required to attempt loading models onto a GPU.
# Flan-T5-large in fp16 needs ~3 GB; set 2 GB as a safe lower bound.
_MIN_FREE_VRAM_BYTES = 2 * 1024 ** 3  # 2 GB


def _best_cuda_device():
    """
    Among all visible CUDA devices, return the one with the most free VRAM.
    Returns None if no device has enough free VRAM to be useful.

    Uses an actual CUDA tensor allocation to confirm the device is truly
    functional - not just listed by the driver.
    """
    if not torch.cuda.is_available():
        return None

    best_device = None
    best_free = 0

    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            # Query free memory via a lightweight reset + query cycle
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)

            logger.info(
                f"[device] GPU {i}: {props.name} | "
                f"free={free_bytes/1e9:.1f} GB / total={total_bytes/1e9:.1f} GB"
            )

            if free_bytes < _MIN_FREE_VRAM_BYTES:
                logger.info(
                    f"[device] GPU {i} skipped - only {free_bytes/1e9:.2f} GB free "
                    f"(minimum required: {_MIN_FREE_VRAM_BYTES/1e9:.1f} GB)"
                )
                continue

            # Confirm the device actually works with a real allocation
            probe = torch.zeros(1, device=torch.device("cuda", i))
            del probe

            if free_bytes > best_free:
                best_free = free_bytes
                best_device = torch.device("cuda", i)

        except Exception as e:
            logger.warning(f"[device] GPU {i} probe failed - skipping: {e}")
            continue

    return best_device


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Call this inside __init__ or at the start of a function - NOT at module
    import time - so the selection reflects the actual runtime state.

    Returns
    -------
    torch.device
        cuda:N  - best NVIDIA/AMD GPU by free VRAM (N = device index)
        mps     - Apple Silicon GPU
        cpu     - fallback
    """
    # --- Try CUDA first (NVIDIA, or AMD via ROCm) ---
    cuda_device = _best_cuda_device()
    if cuda_device is not None:
        props = torch.cuda.get_device_properties(cuda_device.index)
        free, _ = torch.cuda.mem_get_info(cuda_device.index)
        logger.info(
            f"[device] Selected: {cuda_device} - {props.name} "
            f"({free/1e9:.1f} GB free VRAM)"
        )
        return cuda_device

    # --- Try MPS (Apple Silicon M-series) ---
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            probe = torch.zeros(1, device="mps")
            del probe
            logger.info("[device] Selected: mps (Apple Silicon)")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"[device] MPS probe failed - falling back to CPU: {e}")

    # --- CPU fallback ---
    logger.info(
        "[device] Selected: cpu - no usable GPU found. "
        "Inference works but will be slower. "
        "For training, consider Google Colab (free T4 GPU)."
    )
    return torch.device("cpu")


def device_info() -> str:
    """Return a one-line human-readable summary of the selected device."""
    d = get_device()
    if d.type == "cuda":
        props = torch.cuda.get_device_properties(d.index)
        free, total = torch.cuda.mem_get_info(d.index)
        return (
            f"CUDA GPU {d.index}: {props.name} | "
            f"{free/1e9:.1f} GB free / {total/1e9:.1f} GB total"
        )
    if d.type == "mps":
        return "Apple Silicon MPS"
    return "CPU (no GPU)"
