"""Rainfall field super-resolution CNN.

Demonstrates spatial "downscaling" (coarse -> fine rainfall field) with a
convolutional super-resolution architecture inspired by SRCNN.

IMPORTANT (honesty note from the Phase 0 audit):
- The NASA POWER record in this repository is a *single location* monthly
  time series (data/raw/nasa_power/nasa_power_updated.csv). There is no real
  gridded coarse + fine pair available, so this script builds an illustrative
  synthetic rainfall field from the observed monthly record and a smooth
  spatial kernel, then trains the CNN to recover the fine field from a
  downsampled coarse version.
- This is a *demonstration harness* for the downscaling methodology, not a
  claim that a validated downscaled climate product has been produced.
- Generated outputs are written to data/processed/downscaled/.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "nasa_power_updated.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "downscaled"


class DownscaleCNN(nn.Module):
    """SRCNN-style 3-layer super-resolution network."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_field(record: pd.DataFrame, size: int = 128, seed: int = 42) -> np.ndarray:
    """Build a smooth synthetic rainfall field from the monthly record.

    The spatial structure is generated from the observed rainfall statistics
    so the magnitude of the field reflects the real record, while the spatial
    pattern is an analytical stand-in for data we do not currently possess.
    """
    rng = np.random.default_rng(seed)
    rainfall = record["Rainfall"].to_numpy(dtype=np.float32)
    amp = rainfall.std()
    base = rainfall.mean()

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    field = base + amp * (
        0.5 * np.sin(2 * np.pi * (x / size) * 3 + 0.7)
        + 0.35 * np.cos(2 * np.pi * (y / size) * 4)
        + 0.15 * rng.normal(0, 1, (size, size))
    )
    return np.clip(field, 0.0, None)


def make_pairs(field: np.ndarray, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Create (coarse, fine) training pairs by spatial downsampling.

    fine size = coarse size * factor. Returns arrays shaped H x W with
    fine as the ground truth and coarse as the CNN input.
    """
    from skimage.transform import resize

    fine_size = field.shape[0]
    coarse_size = fine_size // factor

    coarse = resize(field, (coarse_size, coarse_size), preserve_range=True)
    fine = resize(coarse, (fine_size, fine_size), order=3, preserve_range=True)

    # Bias the fine field slightly beyond the direct resize so the model
    # has something to learn beyond a pure bicubic interpolation.
    fine = 0.9 * fine + 0.1 * field
    return fine, coarse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to NASA POWER csv (default: project raw data)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field-size", type=int, default=128)
    parser.add_argument("--factor", type=int, default=4,
                        help="Coarse -> fine spatial super-resolution factor")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    record = pd.read_csv(args.data)
    field = build_field(record, size=args.field_size, seed=args.seed)
    fine, coarse = make_pairs(field, factor=args.factor)

    fine_t = torch.from_numpy(fine)[None, None].float().to(device)
    coarse_t = torch.from_numpy(coarse)[None, None].float().to(device)

    model = DownscaleCNN().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[parallel] DataParallel over {torch.cuda.device_count()} devices")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"[training] input {coarse.shape} -> output {fine.shape}, {args.epochs} epochs")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(coarse_t)
        loss = loss_fn(out, fine_t)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            rel = (loss.item() / fine_t.var().item())
            print(f"  epoch {epoch:3d} | mse={loss.item():.6f} | rel_mse={rel:.4f}")
    elapsed = time.perf_counter() - start
    print(f"[training] done in {elapsed:.1f}s ({elapsed / max(args.epochs, 1):.2f}s/epoch)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "downscale_cnn.pt")
    np.savez(
        OUTPUT_DIR / "fields.npz",
        coarse=coarse,
        fine=fine,
    )
    print(f"[output] saved model and fields to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()