#!/usr/bin/env python3
"""Generate local source-referenced face color correction comparisons."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PY = (
    Path("/Users/blue/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")
    if Path("/Users/blue/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3").exists()
    else Path(sys.executable)
)
HELPER = ROOT / "scripts" / "remote_face_color_reference_postprocess.py"
SUBJECTS = {
    "african_child": "african child.png",
    "south_asian_teenager": "South Asian Teenager.png",
    "white_european_child": "white european child.png",
    "subject_5_year_curly": "subject_5 year curly.webp",
}
METHODS = [
    "lab_mean_shift",
    "lab_selective_shift",
    "rgb_gain_preserve_y",
    "rgb_gain_selective_preserve_y",
    "lab_mean_std",
    "ycbcr_mean_shift",
    "ycbcr_mean_std",
]
LABELS = {
    "lab_mean_shift": "LAB Shift",
    "lab_selective_shift": "Selective LAB",
    "rgb_gain_preserve_y": "RGB Gain + Y",
    "rgb_gain_selective_preserve_y": "Selective RGB + Y",
    "lab_mean_std": "LAB Mean/Std",
    "ycbcr_mean_shift": "YCbCr Shift",
    "ycbcr_mean_std": "YCbCr Mean/Std",
}
METHOD_STRENGTHS = {
    "lab_mean_shift": 0.75,
    "lab_selective_shift": 0.75,
    "rgb_gain_preserve_y": 0.45,
    "rgb_gain_selective_preserve_y": 0.45,
    "lab_mean_std": 0.75,
    "ycbcr_mean_shift": 0.75,
    "ycbcr_mean_std": 0.75,
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT, text=True)


def render_board(subject_dir: Path, labels: list[tuple[str, Path]], output_name: str, *, title_text: str, crop_fn=None) -> None:
    width = 320
    height = 320
    pad = 18
    title = 44
    label_height = 32
    bg = (248, 248, 248)
    text = (20, 20, 20)
    border = (210, 210, 210)

    cols = 3 if len(labels) <= 6 else 4
    rows = math.ceil(len(labels) / cols)
    canvas = Image.new(
        "RGB",
        (pad + cols * (width + pad), title + pad + rows * (height + label_height + pad)),
        bg,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), title_text, fill=text)

    for index, (label, path) in enumerate(labels):
        img = Image.open(path).convert("RGB")
        if crop_fn is not None:
            img = crop_fn(label, img)
        img = ImageOps.contain(img, (width, height), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (width, height + label_height), bg)
        x = (width - img.width) // 2
        y = (height - img.height) // 2
        panel.paste(img, (x, y))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((0, 0, width - 1, height - 1), outline=border, width=2)
        panel_draw.text((10, height + 7), label, fill=text)

        row = index // cols
        col = index % cols
        canvas.paste(panel, (pad + col * (width + pad), title + row * (height + label_height + pad)))

    canvas.save(subject_dir / output_name)


def build_board(subject_dir: Path, source_name: str) -> None:
    labels = [("Original Source", subject_dir / source_name), ("Baseline", subject_dir / "baseline.png")]
    labels.extend((LABELS[method], subject_dir / f"{method}.png") for method in METHODS)
    render_board(
        subject_dir,
        labels,
        "comparison_board.png",
        title_text=subject_dir.name.replace("_", " "),
    )


def _mask_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    arr = mask.convert("L")
    bbox = arr.getbbox()
    if bbox is None:
        width, height = arr.size
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        return x1, y1, x1 + side, y1 + side
    x1, y1, x2, y2 = bbox
    pad_x = max((x2 - x1) // 2, 32)
    pad_y = max((y2 - y1) // 2, 32)
    width, height = arr.size
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def build_face_board(subject_dir: Path, source_name: str) -> None:
    labels = [("Original Source", subject_dir / source_name), ("Baseline", subject_dir / "baseline.png")]
    labels.extend((LABELS[method], subject_dir / f"{method}.png") for method in METHODS)
    baseline_for_size = Image.open(subject_dir / "baseline.png").convert("RGB")
    face_mask = Image.open(subject_dir / "face_mask.png")
    if face_mask.size != baseline_for_size.size:
        face_mask = face_mask.resize(baseline_for_size.size, Image.Resampling.NEAREST)
    target_bbox = _mask_bbox(face_mask)

    def crop_fn(label: str, img: Image.Image) -> Image.Image:
        if label == "Original Source":
            w, h = img.size
            crop = (max(0, int(0.15 * w)), max(0, int(0.05 * h)), min(w, int(0.85 * w)), min(h, int(0.80 * h)))
            return img.crop(crop)
        return img.crop(target_bbox)

    render_board(
        subject_dir,
        labels,
        "comparison_face_board.png",
        title_text=f"{subject_dir.name} face crop",
        crop_fn=crop_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "test_outputs" / "visual_prompt_subject_matrix_superman_20260418_131858"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "test_outputs" / "face_color_method_matrix_superman_20260422"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for slug, source_name in SUBJECTS.items():
        subject_dir = input_dir / slug
        source_image = ROOT / "test_subjects" / source_name
        baseline = subject_dir / "final_hires_postprocess_00001_.png"
        face_mask = subject_dir / "inner_face_mask_00001_.png"
        out_subject_dir = output_dir / slug
        out_subject_dir.mkdir(parents=True, exist_ok=True)

        run(["cp", str(source_image), str(out_subject_dir / source_image.name)])
        run(["cp", str(baseline), str(out_subject_dir / "baseline.png")])
        run(["cp", str(face_mask), str(out_subject_dir / "face_mask.png")])

        for method in METHODS:
            out_image = out_subject_dir / f"{method}.png"
            out_mask = out_subject_dir / f"{method}_mask.png"
            run(
                [
                    str(RUNTIME_PY),
                    str(HELPER),
                    "--source-image",
                    str(source_image),
                    "--image",
                    str(baseline),
                    "--face-mask",
                    str(face_mask),
                    "--output",
                    str(out_image),
                    "--refined-mask-output",
                    str(out_mask),
                    "--method",
                    method,
                    "--strength",
                    str(METHOD_STRENGTHS[method]),
                ]
            )

        build_board(out_subject_dir, source_image.name)
        build_face_board(out_subject_dir, source_image.name)
        print(out_subject_dir / "comparison_board.png")
        print(out_subject_dir / "comparison_face_board.png")


if __name__ == "__main__":
    main()
