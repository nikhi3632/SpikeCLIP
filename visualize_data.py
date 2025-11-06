#!/usr/bin/env python3
"""
Minimal utility: for each provided .npz file, save a single PNG named {label_name}.png
under outputs/data_viz/.

Logic:
- Read label_name from the .npz (handles bytes/scalars)
- Choose representative image: prefer any 2D array; otherwise sum projection of the first 3D array
- Save grayscale PNG to outputs/data_viz/{label_name}
"""

import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs/data_viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CALTECH = OUTPUT_DIR / "u_caltech"
OUT_CIFAR = OUTPUT_DIR / "u_cifar"
OUT_CALTECH.mkdir(parents=True, exist_ok=True)
OUT_CIFAR.mkdir(parents=True, exist_ok=True)


def _to_str(x: Any) -> str:
    try:
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        if hasattr(x, "item"):
            v = x.item()
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="ignore")
            return str(v)
        return str(x)
    except Exception:
        return str(x)


def _sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return safe[:128] if safe else "sample"


def _representative_image(d: np.lib.npyio.NpzFile) -> Tuple[str, np.ndarray | None]:
    for key in d.files:
        arr = d[key]
        if arr.ndim == 2:
            return key, arr
    for key in d.files:
        arr = d[key]
        if arr.ndim == 3:
            try:
                return key, arr.sum(axis=0)
            except Exception:
                continue
    return "image", None


def save_label_image(npz_path: Path, out_dir: Path | None = None) -> Path | None:
    with np.load(npz_path, allow_pickle=False) as d:
        label_name = None
        if "label_name" in d.files:
            try:
                label_name = _to_str(d["label_name"])
            except Exception:
                label_name = None
        if not label_name:
            print(f"Skipping {npz_path.name}: no label_name")
            return None

        rep_key, rep_img = _representative_image(d)
        if rep_img is None:
            print(f"Skipping {npz_path.name}: no suitable array to visualize")
            return None

        # Determine class subfolder from label_name: text after first '_' up to last '.'
        cls_name = None
        if "_" in label_name:
            try:
                cls_part = label_name.split("_", 1)[1]
                cls_name = cls_part.rsplit(".", 1)[0]
            except Exception:
                cls_name = None
        cls_name = cls_name or "unknown"
        safe_class = _sanitize_filename(cls_name)

        safe_label = _sanitize_filename(label_name)
        target_dir = (out_dir if out_dir is not None else OUTPUT_DIR) / safe_class
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / safe_label
        if out_path.suffix.lower() != ".jpg":
            out_path = out_path.with_suffix(".jpg")

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        im = ax.imshow(rep_img, cmap="gray")
        ax.set_title(f"{safe_label} ({rep_key})")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, format="jpg")
        plt.close(fig)
        return out_path


def main() -> None:
    args = [Path(p) for p in sys.argv[1:]]
    if not args:
        # Process every .npz in data/u-caltech and data/u-cifar (train and test)
        datasets = [
            (Path("data/u-caltech"), OUT_CALTECH),
            (Path("data/u-cifar"), OUT_CIFAR),
        ]
        total = 0
        for data_root, out_root in datasets:
            for split in ("train", "test"):
                split_dir = data_root / split
                if not split_dir.exists():
                    continue
                files = sorted([p for p in split_dir.iterdir() if p.suffix == ".npz"], key=lambda x: x.stem)
                for p in files:
                    out = save_label_image(p, out_root)
                    if out is not None:
                        print(f"Saved {out}")
                        total += 1
        if total == 0:
            print("No .npz files found under data/. Run fetch_data.py first or pass file paths.")
        return

    # If specific files/dirs provided, process them and route outputs by dataset
    for p in args:
        path = p
        if path.is_dir():
            files = sorted([q for q in path.rglob("*.npz")], key=lambda x: x.stem)
            for q in files:
                out_root = OUT_CALTECH if "u-caltech" in str(q) else OUT_CIFAR if "u-cifar" in str(q) else OUTPUT_DIR
                out = save_label_image(q, out_root)
                if out is not None:
                    print(f"Saved {out}")
        else:
            out_root = OUT_CALTECH if "u-caltech" in str(path) else OUT_CIFAR if "u-cifar" in str(path) else OUTPUT_DIR
            out = save_label_image(path, out_root)
            if out is not None:
                print(f"Saved {out}")


if __name__ == "__main__":
    main()
