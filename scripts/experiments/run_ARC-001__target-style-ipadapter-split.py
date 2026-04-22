#!/usr/bin/env python3
"""Run ARC-001 tracked matrices and build comparison boards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNTIME_PY = Path(
    "/Users/blue/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
)
if not RUNTIME_PY.exists():
    RUNTIME_PY = Path(sys.executable)

try:
    from PIL import Image, ImageDraw, ImageOps
except ModuleNotFoundError:
    if Path(sys.executable) != RUNTIME_PY and RUNTIME_PY.exists():
        raise SystemExit(
            subprocess.run([str(RUNTIME_PY), __file__, *sys.argv[1:]], check=False).returncode
        )
    raise


ROOT = Path(__file__).resolve().parents[2]
CLI_PY = ROOT / ".venv" / "bin" / "python"
if not CLI_PY.exists():
    CLI_PY = Path(sys.executable)
BUILD_SCRIPT = ROOT / "scripts" / "build_visual_prompt_hybrid_workflow.py"
MATRIX_SCRIPT = ROOT / "scripts" / "run_visual_prompt_subject_matrix.py"
SIMPLEPOD_SCRIPT = ROOT / "scripts" / "simplepod.py"
EXP_ID = "ARC-001"
EXP_SLUG = "target-style-ipadapter-split"
WORKFLOW_DIR = ROOT / "workflows" / "experiments" / "arc"
WORKFLOW_API = WORKFLOW_DIR / f"{EXP_ID}__{EXP_SLUG}_api.json"
WORKFLOW_UI = WORKFLOW_DIR / f"{EXP_ID}__{EXP_SLUG}_ui.json"
REPORT_DIR = ROOT / "docs" / "experiments" / "reports"
REPORT_PATH = REPORT_DIR / f"{EXP_ID}__{EXP_SLUG}.md"
DEFAULT_BASELINE_DIR = ROOT / "test_outputs" / "visual_prompt_subject_matrix_superman_20260418_131858"
PREFERRED_OUTPUT_NAMES = [
    "final_hires_postprocess_00001_.png",
    "final_hires_00001_.png",
    "final_postprocess_00001_.png",
    "final_00001_.png",
]


def slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in Path(name).stem.lower()).strip("_")


def run(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        check=True,
        capture_output=capture_output,
    )


def ensure_remote_ready(port: int) -> None:
    run([str(CLI_PY), str(SIMPLEPOD_SCRIPT), "profile"])
    run([str(CLI_PY), str(BUILD_SCRIPT)])
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(CLI_PY),
            str(BUILD_SCRIPT),
            "--output",
            str(WORKFLOW_API),
            "--ui-output",
            str(WORKFLOW_UI),
        ]
    )
    run([str(CLI_PY), str(SIMPLEPOD_SCRIPT), "deploy-visual-prompt-hybrid"])
    run([str(CLI_PY), str(SIMPLEPOD_SCRIPT), "init-auth"])
    run([str(CLI_PY), str(SIMPLEPOD_SCRIPT), "preflight-visual-prompt", "--port", str(port)])


def run_matrix_branch(
    *,
    branch_dir: Path,
    target_image: str,
    port: int,
    wait: int,
    builder_args: list[str],
) -> None:
    cmd = [
        str(CLI_PY),
        str(MATRIX_SCRIPT),
        "--skip-setup",
        "--include-default-excluded",
        "--target-image",
        target_image,
        "--port",
        str(port),
        "--wait",
        str(wait),
        "--local-dir",
        str(branch_dir),
    ]
    for builder_arg in builder_args:
        cmd.extend(["--builder-arg", builder_arg])
    run(cmd)


def load_results(results_path: Path) -> dict[str, str]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return {slugify(item["subject"]): item["subject"] for item in data}


def choose_output(subject_dir: Path) -> Path:
    for name in PREFERRED_OUTPUT_NAMES:
        candidate = subject_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No preferred output found in {subject_dir}")


def choose_mask(subject_dirs: list[Path]) -> Path | None:
    for subject_dir in subject_dirs:
        candidate = subject_dir / "inner_face_mask_00001_.png"
        if candidate.exists():
            return candidate
    return None


def mask_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    arr = mask.convert("L")
    bbox = arr.getbbox()
    width, height = arr.size
    if bbox is None:
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        return x1, y1, x1 + side, y1 + side
    x1, y1, x2, y2 = bbox
    pad_x = max((x2 - x1) // 2, 48)
    pad_y = max((y2 - y1) // 2, 48)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def crop_source_face(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop = (
        max(0, int(0.12 * width)),
        max(0, int(0.04 * height)),
        min(width, int(0.88 * width)),
        min(height, int(0.82 * height)),
    )
    return image.crop(crop)


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)


def render_subject_board(
    *,
    subject_name: str,
    source_path: Path,
    baseline_path: Path,
    reactor_path: Path,
    no_reactor_path: Path,
    mask_path: Path | None,
    output_path: Path,
    face_output_path: Path,
) -> None:
    panel_width = 320
    panel_height = 320
    label_height = 36
    pad = 18
    title_height = 44
    bg = (248, 248, 248)
    text = (20, 20, 20)
    border = (210, 210, 210)

    labels = [
        ("Source", source_path),
        ("VPH-005 Baseline", baseline_path),
        ("ARC-001 ReActor", reactor_path),
        ("ARC-001 No ReActor", no_reactor_path),
    ]
    canvas = Image.new(
        "RGB",
        (pad + len(labels) * (panel_width + pad), title_height + pad + panel_height + label_height + pad),
        bg,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), subject_name, fill=text)

    face_bbox = None
    if mask_path is not None and mask_path.exists():
        face_bbox = mask_bbox(Image.open(mask_path))

    face_canvas = Image.new("RGB", canvas.size, bg)
    face_draw = ImageDraw.Draw(face_canvas)
    face_draw.text((pad, 12), f"{subject_name} face crop", fill=text)

    for index, (label, path) in enumerate(labels):
        image = Image.open(path).convert("RGB")
        if label == "Source":
            face_image = crop_source_face(image)
        elif face_bbox is not None:
            face_image = image.crop(face_bbox)
        else:
            face_image = image

        full_panel = Image.new("RGB", (panel_width, panel_height + label_height), bg)
        fitted = fit_panel(image, panel_width, panel_height)
        full_panel.paste(fitted, ((panel_width - fitted.width) // 2, (panel_height - fitted.height) // 2))
        full_draw = ImageDraw.Draw(full_panel)
        full_draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=border, width=2)
        full_draw.text((10, panel_height + 8), label, fill=text)
        canvas.paste(full_panel, (pad + index * (panel_width + pad), title_height))

        face_panel = Image.new("RGB", (panel_width, panel_height + label_height), bg)
        face_fitted = fit_panel(face_image, panel_width, panel_height)
        face_panel.paste(face_fitted, ((panel_width - face_fitted.width) // 2, (panel_height - face_fitted.height) // 2))
        face_panel_draw = ImageDraw.Draw(face_panel)
        face_panel_draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=border, width=2)
        face_panel_draw.text((10, panel_height + 8), label, fill=text)
        face_canvas.paste(face_panel, (pad + index * (panel_width + pad), title_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    face_output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    face_canvas.save(face_output_path)


def render_all_subjects_face_matrix(
    *,
    rows: list[dict[str, Path | str]],
    output_path: Path,
) -> None:
    cell_width = 240
    cell_height = 240
    row_label_width = 170
    header_height = 48
    label_height = 28
    pad = 18
    bg = (248, 248, 248)
    text = (20, 20, 20)
    border = (210, 210, 210)
    columns = ["Source", "VPH-005 Baseline", "ARC-001 ReActor", "ARC-001 No ReActor"]
    canvas = Image.new(
        "RGB",
        (row_label_width + pad + len(columns) * (cell_width + pad), header_height + pad + len(rows) * (cell_height + label_height + pad)),
        bg,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), "ARC-001 face matrix", fill=text)

    for col_index, column in enumerate(columns):
        x = row_label_width + pad + col_index * (cell_width + pad)
        draw.text((x, 12), column, fill=text)

    for row_index, row in enumerate(rows):
        y = header_height + row_index * (cell_height + label_height + pad)
        draw.text((pad, y + cell_height // 2), str(row["subject"]), fill=text)
        for col_index, key in enumerate(("source", "baseline", "reactor", "no_reactor")):
            image = Image.open(Path(row[key])).convert("RGB")
            fitted = fit_panel(image, cell_width, cell_height)
            panel = Image.new("RGB", (cell_width, cell_height + label_height), bg)
            panel.paste(fitted, ((cell_width - fitted.width) // 2, (cell_height - fitted.height) // 2))
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rectangle((0, 0, cell_width - 1, cell_height - 1), outline=border, width=2)
            x = row_label_width + pad + col_index * (cell_width + pad)
            canvas.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_report(
    *,
    stamp: str,
    target_image: str,
    output_root: Path,
    baseline_dir: Path,
    board_dir: Path,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# {EXP_ID} {EXP_SLUG}

- Experiment ID: `{EXP_ID}`
- Name: `Target-style explicit IP-Adapter split`
- Purpose: verify whether feeding the target image into IP-Adapter improves target style/color retention across the Superman subject matrix, and compare the result against both the current recommended matrix and a no-ReActor branch.
- Workflow / Script: `scripts/build_visual_prompt_hybrid_workflow.py`, `scripts/run_visual_prompt_subject_matrix.py`, `scripts/experiments/run_{EXP_ID}__{EXP_SLUG}.py`
- Method: regenerate the visual-prompt workflow with target-driven IP-Adapter guidance, run the five-subject Superman matrix with ReActor enabled and disabled, then render side-by-side boards against the current recommended `VPH-005` baseline.
- Key Parameters: `target=superman.png`, `port=8191`, `include_default_excluded=true`, branch A=`default`, branch B=`--disable-reactor`
- Inputs: `test_subjects/*`, baseline matrix `{baseline_dir}`
- Outputs: `{output_root}`, boards under `{board_dir}`
- Status: `in progress`
- Findings: pending visual review of the generated comparison boards.
- Next Decision: compare face crops and full-frame boards, then decide whether `ARC-001` should replace `VPH-005` as the recommended Superman matrix or remain an exploratory branch.

## Latest Run

- Timestamp: `{stamp}`
- Target: `{target_image}`
- Reactor-on matrix: `{output_root / "reactor_on"}`
- Reactor-off matrix: `{output_root / "reactor_off"}`
- Aggregate face matrix: `{board_dir / "all_subjects_face_matrix.png"}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-image", default="superman.png")
    parser.add_argument("--port", type=int, default=8191)
    parser.add_argument("--wait", type=int, default=900)
    parser.add_argument("--skip-remote-setup", action="store_true")
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_slug = slugify(args.target_image)
    output_root = Path(
        args.output_root
        or ROOT / "test_outputs" / "arc" / f"{EXP_ID}__{EXP_SLUG}" / f"{stamp}__{target_slug}"
    )
    baseline_dir = Path(args.baseline_dir)
    reactor_on_dir = output_root / "reactor_on"
    reactor_off_dir = output_root / "reactor_off"
    board_dir = output_root / "boards"

    if not args.skip_remote_setup:
        ensure_remote_ready(args.port)

    run_matrix_branch(
        branch_dir=reactor_on_dir,
        target_image=args.target_image,
        port=args.port,
        wait=args.wait,
        builder_args=[],
    )
    run_matrix_branch(
        branch_dir=reactor_off_dir,
        target_image=args.target_image,
        port=args.port,
        wait=args.wait,
        builder_args=["--disable-reactor"],
    )

    reactor_subjects = load_results(reactor_on_dir / "results.json")
    no_reactor_subjects = load_results(reactor_off_dir / "results.json")
    if reactor_subjects != no_reactor_subjects:
        raise SystemExit("ReActor-on and no-ReActor subject sets do not match.")

    face_rows: list[dict[str, Path | str]] = []
    for slug, subject_name in sorted(reactor_subjects.items()):
        source_path = ROOT / "test_subjects" / subject_name
        baseline_subject_dir = baseline_dir / slug
        reactor_subject_dir = reactor_on_dir / slug
        no_reactor_subject_dir = reactor_off_dir / slug
        mask_path = choose_mask([reactor_subject_dir, no_reactor_subject_dir, baseline_subject_dir])

        baseline_output = choose_output(baseline_subject_dir)
        reactor_output = choose_output(reactor_subject_dir)
        no_reactor_output = choose_output(no_reactor_subject_dir)
        full_board_path = board_dir / "subject_full" / f"{slug}.png"
        face_board_path = board_dir / "subject_face" / f"{slug}.png"
        render_subject_board(
            subject_name=slug,
            source_path=source_path,
            baseline_path=baseline_output,
            reactor_path=reactor_output,
            no_reactor_path=no_reactor_output,
            mask_path=mask_path,
            output_path=full_board_path,
            face_output_path=face_board_path,
        )

        source_face = board_dir / "subject_face_sources" / f"{slug}.png"
        source_face.parent.mkdir(parents=True, exist_ok=True)
        crop_source_face(Image.open(source_path).convert("RGB")).save(source_face)
        if mask_path is not None and mask_path.exists():
            bbox = mask_bbox(Image.open(mask_path))
            baseline_face = Image.open(baseline_output).convert("RGB").crop(bbox)
            reactor_face = Image.open(reactor_output).convert("RGB").crop(bbox)
            no_reactor_face = Image.open(no_reactor_output).convert("RGB").crop(bbox)
        else:
            baseline_face = Image.open(baseline_output).convert("RGB")
            reactor_face = Image.open(reactor_output).convert("RGB")
            no_reactor_face = Image.open(no_reactor_output).convert("RGB")
        baseline_face_path = board_dir / "face_cells" / f"{slug}_baseline.png"
        reactor_face_path = board_dir / "face_cells" / f"{slug}_reactor.png"
        no_reactor_face_path = board_dir / "face_cells" / f"{slug}_no_reactor.png"
        baseline_face_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_face.save(baseline_face_path)
        reactor_face.save(reactor_face_path)
        no_reactor_face.save(no_reactor_face_path)

        face_rows.append(
            {
                "subject": slug,
                "source": source_face,
                "baseline": baseline_face_path,
                "reactor": reactor_face_path,
                "no_reactor": no_reactor_face_path,
            }
        )

    render_all_subjects_face_matrix(
        rows=face_rows,
        output_path=board_dir / "all_subjects_face_matrix.png",
    )

    write_report(
        stamp=stamp,
        target_image=args.target_image,
        output_root=output_root,
        baseline_dir=baseline_dir,
        board_dir=board_dir,
    )

    print(f"ARC-001 outputs: {output_root}")
    print(f"ARC-001 report: {REPORT_PATH}")
    print(f"ARC-001 workflow api: {WORKFLOW_API}")
    print(f"ARC-001 workflow ui: {WORKFLOW_UI}")


if __name__ == "__main__":
    main()
