"""Verify the training environment and report which device PyTorch will use."""
import platform

import datasets
import matplotlib
import numpy as np
import tokenizers
import torch
import tqdm
import transformers


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def main() -> None:
    print(f"Python        {platform.python_version()}")
    print(f"torch         {torch.__version__}")
    print(f"transformers  {transformers.__version__}")
    print(f"datasets      {datasets.__version__}")
    print(f"tokenizers    {tokenizers.__version__}")
    print(f"numpy         {np.__version__}")
    print(f"matplotlib    {matplotlib.__version__}")
    print(f"tqdm          {tqdm.__version__}")
    print()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB, CUDA {torch.version.cuda}")

    device = get_device()
    x = torch.randn(1024, 1024, device=device)
    y = x @ x.T
    print(f"Using device: {device} (matmul test OK, sum={y.sum().item():.2f})")


if __name__ == "__main__":
    main()
