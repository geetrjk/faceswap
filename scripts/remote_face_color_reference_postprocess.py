#!/usr/bin/env python3
"""Source-referenced face color correction helper.

This postprocess uses the original subject portrait as the color reference for
the generated face region. It preserves local luminance/shading while adjusting
only skin chroma inside a soft face mask, which is safer for identity than a
full RGB remap.
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


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    y_f = y.astype(np.float32)
    cb_f = cb.astype(np.float32) - 128.0
    cr_f = cr.astype(np.float32) - 128.0
    r = y_f + 1.402 * cr_f
    g = y_f - 0.344136 * cb_f - 0.714136 * cr_f
    b = y_f + 1.772 * cb_f
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


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
        (r > 50)
        & (g > 30)
        & (b > 15)
        & ((max_c - min_c) > 12)
        & (np.abs(r - g) > 8)
        & (r > g)
        & (r > b)
    )
    ycbcr_rule = (cr > 132) & (cr < 178) & (cb > 82) & (cb < 138) & (y > 35)
    hsv_rule = (
        (saturation > 0.07)
        & (saturation < 0.68)
        & (value > 0.18)
        & ((hue < 0.16) | (hue > 0.92))
    )
    return rgb_rule & ycbcr_rule & hsv_rule


def upper_region_mask(shape: tuple[int, int], *, frac: float = 0.82) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width * 0.5
    cy = height * 0.42
    rx = width * 0.38
    ry = height * 0.48
    ellipse = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0
    return ellipse & (yy < height * frac)


def lab_planes(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = np.asarray(Image.fromarray(rgb, mode="RGB").convert("LAB"), dtype=np.float32)
    return lab[..., 0], lab[..., 1], lab[..., 2]


def lab_to_rgb(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lab = np.stack([l, a, b], axis=-1).clip(0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(lab, mode="LAB").convert("RGB"), dtype=np.uint8)


def stats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    pixels = values[mask]
    if len(pixels) == 0:
        raise ValueError("mask selected no pixels")
    return float(pixels.mean()), max(float(pixels.std()), 1.0)


def masked_median(values: np.ndarray, mask: np.ndarray) -> float:
    pixels = values[mask]
    if len(pixels) == 0:
        raise ValueError("mask selected no pixels")
    return float(np.median(pixels))


def masked_percentile(values: np.ndarray, mask: np.ndarray, q: float) -> float:
    pixels = values[mask]
    if len(pixels) == 0:
        raise ValueError("mask selected no pixels")
    return float(np.percentile(pixels, q))


def blend_masked(base: np.ndarray, corrected: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha3 = np.clip(alpha[..., None], 0.0, 1.0)
    return (base * (1.0 - alpha3) + corrected * alpha3).clip(0, 255).astype(np.uint8)


def selective_skin_weight(l: np.ndarray, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a_mid = masked_median(a, mask)
    b_mid = masked_median(b, mask)
    chroma_dist = np.sqrt((a - a_mid) ** 2 + (b - b_mid) ** 2)
    sigma = max(masked_percentile(chroma_dist, mask, 75.0), 6.0)
    chroma_weight = np.exp(-((chroma_dist / sigma) ** 2))

    l_norm = l / 255.0
    shadow_gate = np.clip((l_norm - 0.16) / 0.18, 0.0, 1.0)
    highlight_gate = np.clip((0.92 - l_norm) / 0.18, 0.0, 1.0)
    return (chroma_weight * shadow_gate * highlight_gate * mask.astype(np.float32)).clip(0.0, 1.0)


def method_lab_mean_shift(source: np.ndarray, target: np.ndarray, mask_src: np.ndarray, mask_tgt: np.ndarray, strength: float) -> np.ndarray:
    l_t, a_t, b_t = lab_planes(target)
    _, a_s, b_s = lab_planes(source)
    a_s_mean, _ = stats(a_s, mask_src)
    b_s_mean, _ = stats(b_s, mask_src)
    a_t_mean, _ = stats(a_t, mask_tgt)
    b_t_mean, _ = stats(b_t, mask_tgt)

    a_out = a_t.copy()
    b_out = b_t.copy()
    a_out[mask_tgt] += (a_s_mean - a_t_mean) * strength
    b_out[mask_tgt] += (b_s_mean - b_t_mean) * strength
    return lab_to_rgb(l_t, a_out, b_out)


def method_lab_mean_std(source: np.ndarray, target: np.ndarray, mask_src: np.ndarray, mask_tgt: np.ndarray, strength: float) -> np.ndarray:
    l_t, a_t, b_t = lab_planes(target)
    _, a_s, b_s = lab_planes(source)
    a_s_mean, a_s_std = stats(a_s, mask_src)
    b_s_mean, b_s_std = stats(b_s, mask_src)
    a_t_mean, a_t_std = stats(a_t, mask_tgt)
    b_t_mean, b_t_std = stats(b_t, mask_tgt)

    a_out = a_t.copy()
    b_out = b_t.copy()

    a_scale = np.clip(a_s_std / a_t_std, 0.85, 1.15)
    b_scale = np.clip(b_s_std / b_t_std, 0.85, 1.15)
    a_region = ((a_t[mask_tgt] - a_t_mean) * a_scale + a_s_mean)
    b_region = ((b_t[mask_tgt] - b_t_mean) * b_scale + b_s_mean)

    a_out[mask_tgt] = a_t[mask_tgt] * (1.0 - strength) + a_region * strength
    b_out[mask_tgt] = b_t[mask_tgt] * (1.0 - strength) + b_region * strength
    return lab_to_rgb(l_t, a_out, b_out)


def method_ycbcr_mean_shift(source: np.ndarray, target: np.ndarray, mask_src: np.ndarray, mask_tgt: np.ndarray, strength: float) -> np.ndarray:
    y_t, cb_t, cr_t = rgb_to_ycbcr(target)
    _, cb_s, cr_s = rgb_to_ycbcr(source)
    cb_s_mean, _ = stats(cb_s, mask_src)
    cr_s_mean, _ = stats(cr_s, mask_src)
    cb_t_mean, _ = stats(cb_t, mask_tgt)
    cr_t_mean, _ = stats(cr_t, mask_tgt)

    cb_out = cb_t.copy()
    cr_out = cr_t.copy()
    cb_out[mask_tgt] += (cb_s_mean - cb_t_mean) * strength
    cr_out[mask_tgt] += (cr_s_mean - cr_t_mean) * strength
    rgb = np.stack([y_t, cb_out, cr_out], axis=-1).clip(0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb, mode="YCbCr").convert("RGB"), dtype=np.uint8)


def method_ycbcr_mean_std(source: np.ndarray, target: np.ndarray, mask_src: np.ndarray, mask_tgt: np.ndarray, strength: float) -> np.ndarray:
    y_t, cb_t, cr_t = rgb_to_ycbcr(target)
    _, cb_s, cr_s = rgb_to_ycbcr(source)
    cb_s_mean, cb_s_std = stats(cb_s, mask_src)
    cr_s_mean, cr_s_std = stats(cr_s, mask_src)
    cb_t_mean, cb_t_std = stats(cb_t, mask_tgt)
    cr_t_mean, cr_t_std = stats(cr_t, mask_tgt)

    cb_scale = np.clip(cb_s_std / cb_t_std, 0.85, 1.15)
    cr_scale = np.clip(cr_s_std / cr_t_std, 0.85, 1.15)
    cb_region = ((cb_t[mask_tgt] - cb_t_mean) * cb_scale + cb_s_mean)
    cr_region = ((cr_t[mask_tgt] - cr_t_mean) * cr_scale + cr_s_mean)

    cb_out = cb_t.copy()
    cr_out = cr_t.copy()
    cb_out[mask_tgt] = cb_t[mask_tgt] * (1.0 - strength) + cb_region * strength
    cr_out[mask_tgt] = cr_t[mask_tgt] * (1.0 - strength) + cr_region * strength
    rgb = np.stack([y_t, cb_out, cr_out], axis=-1).clip(0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb, mode="YCbCr").convert("RGB"), dtype=np.uint8)


def method_lab_selective_shift(source: np.ndarray, target: np.ndarray, mask_src: np.ndarray, mask_tgt: np.ndarray, strength: float) -> np.ndarray:
    l_t, a_t, b_t = lab_planes(target)
    _, a_s, b_s = lab_planes(source)
    a_delta = (masked_median(a_s, mask_src) - masked_median(a_t, mask_tgt)) * strength
    b_delta = (masked_median(b_s, mask_src) - masked_median(b_t, mask_tgt)) * strength
    weight = selective_skin_weight(l_t, a_t, b_t, mask_tgt)

    a_out = a_t.copy()
    b_out = b_t.copy()
    a_out[mask_tgt] += a_delta * weight[mask_tgt]
    b_out[mask_tgt] += b_delta * weight[mask_tgt]
    return lab_to_rgb(l_t, a_out, b_out)


def method_rgb_gain_preserve_y(source: np.ndarray, target: np.ndarray, mask_src: np.ndarray, mask_tgt: np.ndarray, strength: float) -> np.ndarray:
    source_rgb = source.astype(np.float32)
    target_rgb = target.astype(np.float32)
    src_med = np.array([masked_median(source_rgb[..., i], mask_src) for i in range(3)], dtype=np.float32)
    tgt_med = np.array([masked_median(target_rgb[..., i], mask_tgt) for i in range(3)], dtype=np.float32)

    gains = np.divide(src_med, np.maximum(tgt_med, 1.0))
    gains = 1.0 + (gains - 1.0) * strength
    gains = np.clip(gains, 0.82, 1.18)

    gained = (target_rgb * gains.reshape(1, 1, 3)).clip(0, 255)
    y_t, _, _ = rgb_to_ycbcr(target_rgb)
    _, cb_g, cr_g = rgb_to_ycbcr(gained)
    return ycbcr_to_rgb(y_t, cb_g, cr_g)


def method_rgb_gain_selective_preserve_y(source: np.ndarray, target: np.ndarray, mask_src: np.ndarray, mask_tgt: np.ndarray, strength: float) -> np.ndarray:
    target_rgb = target.astype(np.float32)
    gained_rgb = method_rgb_gain_preserve_y(source, target, mask_src, mask_tgt, strength).astype(np.float32)
    l_t, a_t, b_t = lab_planes(target)
    weight = selective_skin_weight(l_t, a_t, b_t, mask_tgt)[..., None]
    return (target_rgb * (1.0 - weight) + gained_rgb * weight).clip(0, 255).astype(np.uint8)


METHODS = {
    "lab_mean_shift": method_lab_mean_shift,
    "lab_mean_std": method_lab_mean_std,
    "lab_selective_shift": method_lab_selective_shift,
    "rgb_gain_preserve_y": method_rgb_gain_preserve_y,
    "rgb_gain_selective_preserve_y": method_rgb_gain_selective_preserve_y,
    "ycbcr_mean_shift": method_ycbcr_mean_shift,
    "ycbcr_mean_std": method_ycbcr_mean_std,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--face-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--refined-mask-output", required=True)
    parser.add_argument("--threshold", type=int, default=32)
    parser.add_argument("--face-grow", type=int, default=42)
    parser.add_argument("--alpha-dilate", type=int, default=7)
    parser.add_argument("--alpha-blur", type=float, default=12.0)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--method", choices=sorted(METHODS), default="rgb_gain_selective_preserve_y")
    args = parser.parse_args()

    source = load_rgb(Path(args.source_image))
    target = load_rgb(Path(args.image))
    face_mask = load_mask(Path(args.face_mask), threshold=args.threshold)

    height, width = target.shape[:2]
    if face_mask.shape != (height, width):
        face_mask = resize_mask(face_mask, (width, height))

    source_mask = skin_like_mask(source) & upper_region_mask(source.shape[:2])
    target_mask = skin_like_mask(target) & dilate(face_mask, args.face_grow)
    if source_mask.sum() < 500 or target_mask.sum() < 500:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.refined_mask_output).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(target, mode="RGB").save(args.output)
        save_mask(target_mask, Path(args.refined_mask_output))
        print("STATUS=skipped")
        print("REASON=insufficient_skin_pixels")
        return

    corrected = METHODS[args.method](source, target, source_mask, target_mask, args.strength)
    alpha = blur_mask(dilate(target_mask, args.alpha_dilate), args.alpha_blur)
    output = blend_masked(target.astype(np.float32), corrected.astype(np.float32), alpha)

    output_path = Path(args.output)
    mask_path = Path(args.refined_mask_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(output_path)
    save_mask(target_mask, mask_path)
    print("STATUS=ok")
    print(f"PIXELS={int(target_mask.sum())}")


if __name__ == "__main__":
    main()
