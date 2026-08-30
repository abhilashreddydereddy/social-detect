"""Image-branch dataset from CSV manifests (CIFake and similar)."""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFile, ImageFilter
from torch.utils.data import Dataset

# High-res / slightly corrupt social-media dumps are common.
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1)


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _load_rgb(path: Path, decode_side: int = 256) -> Image.Image:
    """Open image, downscale early so huge files don't dominate training I/O."""
    with Image.open(path) as img:
        # Faster partial decode for large JPEGs when supported.
        try:
            img.draft("RGB", (decode_side, decode_side))
        except Exception:
            pass
        img = img.convert("RGB")
        if max(img.size) > decode_side:
            img.thumbnail((decode_side, decode_side), Image.BILINEAR)
        return img.copy()


class TrainTransform:
    """Picklable train-time transform (required for Windows DataLoader workers)."""

    def __init__(self, image_size: int, jpeg_aug: bool = True, filter_aug: bool = True) -> None:
        self.image_size = image_size
        self.jpeg_aug = jpeg_aug
        self.filter_aug = filter_aug

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Upsample small CIFAKE tiles or downscale already-loaded RGB.
        img = img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            # Mild brightness/contrast jitter without torchvision dependency.
            arr = np.asarray(img).astype(np.float32)
            factor = random.uniform(0.85, 1.15)
            arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        if self.filter_aug and random.random() < 0.55:
            effect = random.choice(("color", "warm", "cool", "soft", "sharp", "mono"))
            if effect == "color":
                img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.35))
            elif effect == "warm":
                arr = np.asarray(img).astype(np.float32)
                arr[..., 0] *= random.uniform(1.03, 1.12)
                arr[..., 2] *= random.uniform(0.88, 0.98)
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            elif effect == "cool":
                arr = np.asarray(img).astype(np.float32)
                arr[..., 0] *= random.uniform(0.88, 0.98)
                arr[..., 2] *= random.uniform(1.03, 1.12)
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            elif effect == "soft":
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
            elif effect == "sharp":
                img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 1.8))
            else:
                img = ImageEnhance.Color(img).enhance(random.uniform(0.0, 0.2))
        if self.jpeg_aug and random.random() < 0.4:
            img = _jpeg_compress(img, quality=random.randint(40, 95))
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        return _to_tensor_normalized(img)


class EvalTransform:
    """Picklable eval-time transform."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        return _to_tensor_normalized(img)


def build_train_transform(
    image_size: int,
    jpeg_aug: bool = True,
    filter_aug: bool = True,
) -> TrainTransform:
    return TrainTransform(image_size, jpeg_aug=jpeg_aug, filter_aug=filter_aug)


def build_eval_transform(image_size: int) -> EvalTransform:
    return EvalTransform(image_size)


class ImageManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        path_column: str = "path",
        label_column: str = "label",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.df = pd.read_csv(manifest)
        if self.df.empty:
            raise ValueError(f"Manifest is empty: {manifest}")
        if path_column not in self.df.columns or label_column not in self.df.columns:
            raise ValueError(f"Manifest must contain '{path_column}' and '{label_column}' columns")
        self.path_column = path_column
        self.label_column = label_column
        self.transform = transform or build_eval_transform(224)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self.df)
        last_err: Exception | None = None
        for attempt in range(8):
            j = (idx + attempt) % n
            row = self.df.iloc[j]
            path = Path(str(row[self.path_column]))
            label = int(row[self.label_column])
            try:
                img = _load_rgb(path)
                tensor = self.transform(img)
                return {
                    "image": tensor,
                    "label": torch.tensor(label, dtype=torch.long),
                    "path": str(path),
                }
            except Exception as exc:  # noqa: BLE001 — skip corrupt files during training
                last_err = exc
                continue
        raise OSError(f"Failed to load image near index {idx}: {last_err}")


def collate_images(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
    }
