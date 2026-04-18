#!/usr/bin/env python3
"""Remote hi-res sharpening helper for SimplePod outputs.

This script sharpens an already-upscaled image deterministically, with a mild
global pass and a stronger face-focused pass guided by a face mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, *, threshold: int = 32) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return arr >= threshold


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8) >= 32


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return np.asarray(dilated, dtype=np.uint8) >= 32


def blur_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(image, dtype=np.float32) / 255.0


def unsharp(
    image: Image.Image,
    *,
    radius: float,
    percent: int,
    threshold: int,
    contrast: float,
) -> Image.Image:
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    if contrast != 1.0:
        sharpened = ImageEnhance.Contrast(sharpened).enhance(contrast)
    return sharpened


def blend_with_alpha(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = np.clip(alpha[..., None], 0.0, 1.0)
    return (base * (1.0 - alpha_3) + overlay * alpha_3).clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--face-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-threshold", type=int, default=32)
    parser.add_argument("--global-radius", type=float, default=1.2)
    parser.add_argument("--global-percent", type=int, default=110)
    parser.add_argument("--global-threshold", type=int, default=2)
    parser.add_argument("--global-blend", type=float, default=0.35)
    parser.add_argument("--face-radius", type=float, default=1.8)
    parser.add_argument("--face-percent", type=int, default=190)
    parser.add_argument("--face-threshold", type=int, default=2)
    parser.add_argument("--face-contrast", type=float, default=1.04)
    parser.add_argument("--face-grow", type=int, default=28)
    parser.add_argument("--face-blur", type=float, default=14.0)
    parser.add_argument("--face-blend", type=float, default=0.92)
    args = parser.parse_args()

    image_path = Path(args.image)
    face_mask_path = Path(args.face_mask)
    output_path = Path(args.output)

    image = load_rgb(image_path)
    width, height = image.size
    face_mask = load_mask(face_mask_path, threshold=args.mask_threshold)
    if face_mask.shape != (height, width):
        face_mask = resize_mask(face_mask, (width, height))

    face_alpha = blur_mask(dilate(face_mask, args.face_grow), args.face_blur) * args.face_blend

    global_sharp = unsharp(
        image,
        radius=args.global_radius,
        percent=args.global_percent,
        threshold=args.global_threshold,
        contrast=1.0,
    )
    face_sharp = unsharp(
        image,
        radius=args.face_radius,
        percent=args.face_percent,
        threshold=args.face_threshold,
        contrast=args.face_contrast,
    )

    base = np.asarray(image, dtype=np.float32)
    global_arr = np.asarray(global_sharp, dtype=np.float32)
    face_arr = np.asarray(face_sharp, dtype=np.float32)

    out = blend_with_alpha(base, global_arr, np.full((height, width), args.global_blend, dtype=np.float32))
    out = blend_with_alpha(out.astype(np.float32), face_arr, face_alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(output_path)
    print("STATUS=ok")


if __name__ == "__main__":
    main()
