#!/usr/bin/env python3
"""Remote skin-tone harmonization helper for SimplePod outputs.

This script is intended to run on the remote pod where Pillow and numpy are
available. It refines a semantic exposed-skin mask using simple color checks,
samples the solved face tone from the current composite, and harmonizes only
the surviving non-face skin regions toward that tone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path, *, threshold: int = 32) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return arr >= threshold


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def skin_like_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    y, cb, cr = rgb_to_ycbcr(rgb)

    rgb_rule = (
        (r > 40)
        & (g > 20)
        & (b > 10)
        & ((max_c - min_c) > 10)
        & (np.abs(r - g) > 5)
        & (r > g)
        & (r > b)
    )
    ycbcr_rule = (
        (cr > 132)
        & (cr < 180)
        & (cb > 85)
        & (cb < 140)
        & (y > 40)
    )
    return rgb_rule & ycbcr_rule


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(img.filter(ImageFilter.MaxFilter(radius * 2 + 1))) >= 32


def blur_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def color_stats(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = rgb[mask]
    if len(pixels) == 0:
        raise ValueError("mask selected no pixels")
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return mean, np.maximum(std, 1.0)


def harmonize(
    image: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    src_mean, src_std = color_stats(image, source_mask)
    dst_mean, dst_std = color_stats(image, target_mask)

    out = image.astype(np.float32).copy()
    region = out[target_mask]
    normalized = (region - dst_mean) / dst_std
    remapped = normalized * src_std + src_mean
    blended = region * (1.0 - strength) + remapped * strength
    out[target_mask] = np.clip(blended, 0, 255)
    return out.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--candidate-mask", required=True)
    parser.add_argument("--face-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--refined-mask-output", required=True)
    parser.add_argument("--threshold", type=int, default=32)
    parser.add_argument("--min-region-pixels", type=int, default=600)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--blur", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.85)
    args = parser.parse_args()

    image_path = Path(args.image)
    candidate_mask_path = Path(args.candidate_mask)
    face_mask_path = Path(args.face_mask)
    output_path = Path(args.output)
    refined_mask_output_path = Path(args.refined_mask_output)

    rgb = load_rgb(image_path)
    candidate_mask = load_mask(candidate_mask_path, threshold=args.threshold)
    face_mask = load_mask(face_mask_path, threshold=args.threshold)

    # Exclude the already-solved face/neck region from body-skin harmonization.
    candidate_mask &= ~dilate(face_mask, 12)

    # Reject semantic false positives such as gloves/sleeves by requiring the
    # current pixels to look like skin in broad RGB + YCbCr ranges.
    refined_mask = candidate_mask & skin_like_mask(rgb)
    if refined_mask.sum() < args.min_region_pixels:
        refined_mask = np.zeros_like(refined_mask, dtype=bool)
        save_mask(refined_mask, refined_mask_output_path)
        Image.fromarray(rgb, mode="RGB").save(output_path)
        print("STATUS=skipped")
        print("REASON=no_non_face_skin")
        return

    alpha = blur_mask(dilate(refined_mask, args.dilate), args.blur)
    alpha_bool = alpha > 1e-3
    harmonized = harmonize(rgb, face_mask, alpha_bool, strength=args.strength).astype(np.float32)
    base = rgb.astype(np.float32)
    out = (base * (1.0 - alpha[..., None]) + harmonized * alpha[..., None]).clip(0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined_mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(output_path)
    save_mask(refined_mask, refined_mask_output_path)
    print("STATUS=ok")
    print(f"PIXELS={int(refined_mask.sum())}")


if __name__ == "__main__":
    main()
