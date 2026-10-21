"""PyTorch datasets/loaders over the token files written by prepare_data.py."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DATA_DIR = PROCESSED_DIR / "wikipedia"  # default dataset; others live alongside it (e.g. processed/verdict)


class TokenDataset(Dataset):
    """Splits a flat token file into non-overlapping blocks of `block_size`.

    Each item is (x, y), where y is x shifted one token to the left — the
    next-token targets for language modeling. The file is memory-mapped, so
    the dataset never has to be loaded into RAM in full.
    """

    def __init__(self, bin_path: Path, block_size: int, random_windows: bool = False):
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        # False: block i always covers the same tokens (i * block_size onward).
        # True: every time a block is requested it starts at a random token, so the model
        # sees each passage at many different positions instead of always the same one.
        self.random_windows = random_windows
        self.n_tokens = self.bin_path.stat().st_size // np.dtype(np.uint16).itemsize
        self._tokens = None  # opened lazily so DataLoader workers don't pickle (copy) the whole array

    @property
    def tokens(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        return self._tokens

    def __getstate__(self):
        return {**self.__dict__, "_tokens": None}

    def __len__(self) -> int:
        return (self.n_tokens - 1) // self.block_size

    def __getitem__(self, idx: int):
        if self.random_windows:
            # any start that still leaves room for block_size inputs + 1 target
            start = int(torch.randint(0, self.n_tokens - self.block_size, (1,)))
        else:
            start = idx * self.block_size
        chunk = torch.from_numpy(self.tokens[start:start + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def load_meta(data_dir: Path = DATA_DIR) -> dict:
    return json.loads((data_dir / "meta.json").read_text())


def create_dataloaders(batch_size: int = 8, block_size: int = 256, data_dir: Path = DATA_DIR,
                       num_workers: int = 0):
    train_ds = TokenDataset(data_dir / "train.bin", block_size)
    val_ds = TokenDataset(data_dir / "val.bin", block_size)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader
