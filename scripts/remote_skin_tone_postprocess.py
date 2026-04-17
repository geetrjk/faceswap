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


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_f = rgb.astype(np.float32) / 255.0
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]

    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    delta = max_c - min_c

    hue = np.zeros_like(max_c)
    nonzero = delta > 1e-6

    r_mask = nonzero & (max_c == r)
    g_mask = nonzero & (max_c == g)
    b_mask = nonzero & (max_c == b)

    hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    hue /= 6.0

    saturation = np.zeros_like(max_c)
    valid_value = max_c > 1e-6
    saturation[valid_value] = delta[valid_value] / max_c[valid_value]
    value = max_c
    return hue, saturation, value


def skin_like_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    y, cb, cr = rgb_to_ycbcr(rgb)
    hue, saturation, value = rgb_to_hsv(rgb)

    rgb_rule = (
        (r > 60)
        & (g > 35)
        & (b > 20)
        & ((max_c - min_c) > 15)
        & (np.abs(r - g) > 10)
        & (r > g)
        & (r > b)
    )
    ycbcr_rule = (
        (cr > 135)
        & (cr < 175)
        & (cb > 85)
        & (cb < 135)
        & (y > 40)
    )
    hsv_rule = (
        (saturation > 0.08)
        & (saturation < 0.65)
        & (value > 0.20)
        & ((hue < 0.14) | (hue > 0.94))
    )
    return rgb_rule & ycbcr_rule & hsv_rule


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


def smooth_texture_mask(rgb: np.ndarray, *, blur_radius: float = 2.0, detail_threshold: float = 14.0) -> np.ndarray:
    luma = Image.fromarray(rgb, mode="RGB").convert("L")
    blurred = luma.filter(ImageFilter.GaussianBlur(blur_radius))
    luma_arr = np.asarray(luma, dtype=np.float32)
    blurred_arr = np.asarray(blurred, dtype=np.float32)
    detail = np.abs(luma_arr - blurred_arr)
    return detail <= detail_threshold


def color_stats(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = rgb[mask]
    if len(pixels) == 0:
        raise ValueError("mask selected no pixels")
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return mean, np.maximum(std, 1.0)


def face_tone_compatible_mask(
    rgb: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    chroma_distance_limit: float = 22.0,
    luminance_margin: float = 70.0,
) -> np.ndarray:
    y, cb, cr = rgb_to_ycbcr(rgb)
    src_y = y[source_mask]
    src_cb = cb[source_mask]
    src_cr = cr[source_mask]
    if len(src_y) == 0:
        return np.zeros_like(target_mask, dtype=bool)

    mean_cb = float(src_cb.mean())
    mean_cr = float(src_cr.mean())
    min_y = float(np.percentile(src_y, 2)) - luminance_margin
    max_y = float(np.percentile(src_y, 98)) + luminance_margin

    chroma_distance = np.sqrt((cb - mean_cb) ** 2 + (cr - mean_cr) ** 2)
    return target_mask & (chroma_distance <= chroma_distance_limit) & (y >= min_y) & (y <= max_y)


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
    parser.add_argument("--min-region-pixels", type=int, default=2500)
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
    # current pixels to both look broadly like skin and stay reasonably close
    # to the solved face tone. This keeps fully covered targets like Spider-Man
    # from treating red gloves as exposed skin.
    refined_mask = candidate_mask & skin_like_mask(rgb)
    refined_mask &= face_tone_compatible_mask(rgb, face_mask, refined_mask)
    refined_mask &= smooth_texture_mask(rgb)
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
