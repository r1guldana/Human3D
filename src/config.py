import os

import torch


def get_device(preferred=None):
    """Return the active torch device, preferring CUDA when available."""
    requested = (preferred or os.getenv("SMPLX_DEVICE", "auto")).lower()

    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU selected: {torch.cuda.get_device_name(0)}")
        return device

    if requested == "cpu":
        device = torch.device("cpu")
        print("[Device] CPU selected")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU detected: {torch.cuda.get_device_name(0)}")
        return device

    device = torch.device("cpu")
    print("[Device] No CUDA found, using CPU")
    return device


DEVICE = get_device()