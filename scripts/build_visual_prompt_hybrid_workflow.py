#!/usr/bin/env python3
"""Generate a PuLID + IP-Adapter visual prompt workflow pair.

This is the active replacement for the older fallback graph. The working path is:
- FaceSegmentation to isolate the target face/head edit region
- mask expansion + feathering to cover hairline and cheeks cleanly
- PuLID for source identity transfer
- IP-Adapter Plus Face to preserve source appearance cues
- masked inpainting from the target latent so the body/background stay intact

SAM remains part of the installed environment but is intentionally not wired into the
active graph until a better detector source is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def api_node(class_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def link(node_id: int | str, slot: int) -> list[Any]:
    return [str(node_id), slot]


def node(
    node_id: int,
    node_type: str,
    pos: list[int],
    order: int,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    widgets_values: list[Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": [330, 120],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {},
        "widgets_values": widgets_values or [],
    }
    if title:
        data["title"] = title
    return data


def build_workflow(
    *,
    subject_image: str,
    target_image: str,
    checkpoint: str,
    positive_prompt: str,
    negative_prompt: str,
    coarse_area: str,
    coarse_grow: int,
    coarse_blur: int,
    preserve_expand: int,
    preserve_feather: int,
    pulid_weight: float,
    pulid_projection: str,
    pulid_fidelity: int,
    ipadapter_preset: str,
    ipadapter_weight: float,
    ipadapter_weight_type: str,
    ipadapter_end_at: float,
    inpaint_grow_mask_by: int,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    sampler_name: str,
    scheduler: str,
    filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    return {
        "1": api_node("LoadImage", {"image": subject_image}),
        "2": api_node("LoadImage", {"image": target_image}),
        "3": api_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint}),
        "4": api_node("CLIPTextEncode", {"clip": link(3, 1), "text": positive_prompt}),
        "5": api_node("CLIPTextEncode", {"clip": link(3, 1), "text": negative_prompt}),
        "6": api_node("FaceAnalysisModels", {"library": "insightface", "provider": "CUDA"}),
        "7": api_node(
            "FaceSegmentation",
            {
                "analysis_models": link(6, 0),
                "image": link(2, 0),
                "area": coarse_area,
                "grow": coarse_grow,
                "grow_tapered": True,
                "blur": coarse_blur,
            },
        ),
        "8": api_node("MaskToImage", {"mask": link(7, 0)}),
        "9": api_node(
            "SaveImage",
            {"images": link(8, 0), "filename_prefix": f"{intermediate_prefix}/target_face_mask_coarse"},
        ),
        "10": api_node(
            "GrowMask",
            {"mask": link(7, 0), "expand": preserve_expand, "tapered_corners": True},
        ),
        "11": api_node(
            "FeatherMask",
            {
                "mask": link(10, 0),
                "left": preserve_feather,
                "top": preserve_feather,
                "right": preserve_feather,
                "bottom": preserve_feather,
            },
        ),
        "12": api_node("MaskToImage", {"mask": link(11, 0)}),
        "13": api_node(
            "SaveImage",
            {"images": link(12, 0), "filename_prefix": f"{intermediate_prefix}/target_face_mask_final"},
        ),
        "14": api_node("PulidModelLoader", {"pulid_file": "ip-adapter_pulid_sdxl_fp16.safetensors"}),
        "15": api_node("PulidEvaClipLoader", {}),
        "16": api_node("PulidInsightFaceLoader", {"provider": "CUDA"}),
        "17": api_node(
            "ApplyPulidAdvanced",
            {
                "model": link(3, 0),
                "pulid": link(14, 0),
                "eva_clip": link(15, 0),
                "face_analysis": link(16, 0),
                "image": link(1, 0),
                "weight": pulid_weight,
                "projection": pulid_projection,
                "fidelity": pulid_fidelity,
                "noise": 0.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "attn_mask": link(11, 0),
            },
        ),
        "18": api_node("IPAdapterUnifiedLoader", {"model": link(17, 0), "preset": ipadapter_preset}),
        "19": api_node(
            "IPAdapterAdvanced",
            {
                "model": link(18, 0),
                "ipadapter": link(18, 1),
                "image": link(1, 0),
                "weight": ipadapter_weight,
                "weight_type": ipadapter_weight_type,
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": ipadapter_end_at,
                "embeds_scaling": "V only",
                "attn_mask": link(11, 0),
            },
        ),
        "20": api_node(
            "VAEEncodeForInpaint",
            {
                "pixels": link(2, 0),
                "vae": link(3, 2),
                "mask": link(11, 0),
                "grow_mask_by": inpaint_grow_mask_by,
            },
        ),
        "21": api_node(
            "KSampler",
            {
                "model": link(19, 0),
                "positive": link(4, 0),
                "negative": link(5, 0),
                "latent_image": link(20, 0),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        ),
        "22": api_node("VAEDecode", {"samples": link(21, 0), "vae": link(3, 2)}),
        "23": api_node("SaveImage", {"images": link(22, 0), "filename_prefix": filename_prefix}),
    }


def build_ui_workflow(
    *,
    subject_image: str,
    target_image: str,
    checkpoint: str,
    positive_prompt: str,
    negative_prompt: str,
    coarse_area: str,
    coarse_grow: int,
    coarse_blur: int,
    preserve_expand: int,
    preserve_feather: int,
    pulid_weight: float,
    pulid_projection: str,
    pulid_fidelity: int,
    ipadapter_preset: str,
    ipadapter_weight: float,
    ipadapter_weight_type: str,
    ipadapter_end_at: float,
    inpaint_grow_mask_by: int,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    sampler_name: str,
    scheduler: str,
    filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    links = [
        [1, 3, 1, 4, 0, "CLIP"],
        [2, 3, 1, 5, 0, "CLIP"],
        [3, 6, 0, 7, 0, "ANALYSIS_MODELS"],
        [4, 2, 0, 7, 1, "IMAGE"],
        [5, 7, 0, 8, 0, "MASK"],
        [6, 8, 0, 9, 0, "IMAGE"],
        [7, 7, 0, 10, 0, "MASK"],
        [8, 10, 0, 11, 0, "MASK"],
        [9, 11, 0, 12, 0, "MASK"],
        [10, 12, 0, 13, 0, "IMAGE"],
        [11, 14, 0, 17, 1, "PULID"],
        [12, 15, 0, 17, 2, "EVA_CLIP"],
        [13, 16, 0, 17, 3, "FACEANALYSIS"],
        [14, 3, 0, 17, 0, "MODEL"],
        [15, 1, 0, 17, 4, "IMAGE"],
        [16, 11, 0, 17, 11, "MASK"],
        [17, 17, 0, 18, 0, "MODEL"],
        [18, 18, 0, 19, 0, "MODEL"],
        [19, 18, 1, 19, 1, "IPADAPTER"],
        [20, 1, 0, 19, 2, "IMAGE"],
        [21, 11, 0, 19, 5, "MASK"],
        [22, 2, 0, 20, 0, "IMAGE"],
        [23, 3, 2, 20, 1, "VAE"],
        [24, 11, 0, 20, 2, "MASK"],
        [25, 19, 0, 21, 0, "MODEL"],
        [26, 4, 0, 21, 1, "CONDITIONING"],
        [27, 5, 0, 21, 2, "CONDITIONING"],
        [28, 20, 0, 21, 3, "LATENT"],
        [29, 21, 0, 22, 0, "LATENT"],
        [30, 3, 2, 22, 1, "VAE"],
        [31, 22, 0, 23, 0, "IMAGE"],
        [32, 22, 0, 24, 0, "IMAGE"],
    ]
    return {
        "id": "visual-prompt-hybrid-experiment-ui",
        "revision": 0,
        "last_node_id": 24,
        "last_link_id": 32,
        "nodes": [
            node(1, "LoadImage", [60, 80], 0, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [15, 20]}], widgets_values=[subject_image, "image"], title="Load Source Identity"),
            node(2, "LoadImage", [60, 420], 1, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [4, 22]}], widgets_values=[target_image, "image"], title="Load Target Template"),
            node(3, "CheckpointLoaderSimple", [60, 760], 2, outputs=[{"name": "MODEL", "type": "MODEL", "links": [14]}, {"name": "CLIP", "type": "CLIP", "links": [1, 2]}, {"name": "VAE", "type": "VAE", "links": [23, 30]}], widgets_values=[checkpoint], title="Load SDXL Inpaint Checkpoint"),
            node(4, "CLIPTextEncode", [420, 680], 3, inputs=[{"name": "clip", "type": "CLIP", "link": 1}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [26]}], widgets_values=[positive_prompt], title="Positive Prompt"),
            node(5, "CLIPTextEncode", [420, 910], 4, inputs=[{"name": "clip", "type": "CLIP", "link": 2}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [27]}], widgets_values=[negative_prompt], title="Negative Prompt"),
            node(6, "FaceAnalysisModels", [420, 420], 5, outputs=[{"name": "ANALYSIS_MODELS", "type": "ANALYSIS_MODELS", "links": [3]}], widgets_values=["insightface", "CUDA"], title="Load Face Analysis"),
            node(7, "FaceSegmentation", [780, 380], 6, inputs=[{"name": "analysis_models", "type": "ANALYSIS_MODELS", "link": 3}, {"name": "image", "type": "IMAGE", "link": 4}], outputs=[{"name": "mask", "type": "MASK", "links": [5, 7]}, {"name": "image", "type": "IMAGE", "links": None}, {"name": "seg_mask", "type": "MASK", "links": None}, {"name": "seg_image", "type": "IMAGE", "links": None}, {"name": "x", "type": "INT", "links": None}, {"name": "y", "type": "INT", "links": None}, {"name": "width", "type": "INT", "links": None}, {"name": "height", "type": "INT", "links": None}], widgets_values=[coarse_area, coarse_grow, True, coarse_blur], title="Coarse Target Head Mask"),
            node(8, "MaskToImage", [1140, 320], 7, inputs=[{"name": "mask", "type": "MASK", "link": 5}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [6]}], title="Convert Coarse Mask"),
            node(9, "SaveImage", [1500, 320], 8, inputs=[{"name": "images", "type": "IMAGE", "link": 6}], widgets_values=[f"{intermediate_prefix}/target_face_mask_coarse"], title="Save Coarse Mask"),
            node(10, "GrowMask", [1140, 540], 9, inputs=[{"name": "mask", "type": "MASK", "link": 7}], outputs=[{"name": "MASK", "type": "MASK", "links": [8]}], widgets_values=[preserve_expand, True], title="Expand Preserve Mask"),
            node(11, "FeatherMask", [1500, 540], 10, inputs=[{"name": "mask", "type": "MASK", "link": 8}], outputs=[{"name": "MASK", "type": "MASK", "links": [9, 16, 21, 24]}], widgets_values=[preserve_feather, preserve_feather, preserve_feather, preserve_feather], title="Feather Preserve Mask"),
            node(12, "MaskToImage", [1860, 500], 11, inputs=[{"name": "mask", "type": "MASK", "link": 9}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [10]}], title="Convert Final Mask"),
            node(13, "SaveImage", [2220, 500], 12, inputs=[{"name": "images", "type": "IMAGE", "link": 10}], widgets_values=[f"{intermediate_prefix}/target_face_mask_final"], title="Save Final Mask"),
            node(14, "PulidModelLoader", [1860, 820], 13, outputs=[{"name": "PULID", "type": "PULID", "links": [11]}], widgets_values=["ip-adapter_pulid_sdxl_fp16.safetensors"], title="Load PuLID Model"),
            node(15, "PulidEvaClipLoader", [1860, 960], 14, outputs=[{"name": "EVA_CLIP", "type": "EVA_CLIP", "links": [12]}], title="Load EVA CLIP"),
            node(16, "PulidInsightFaceLoader", [1860, 1100], 15, outputs=[{"name": "FACEANALYSIS", "type": "FACEANALYSIS", "links": [13]}], widgets_values=["CUDA"], title="Load PuLID InsightFace"),
            node(17, "ApplyPulidAdvanced", [2580, 900], 16, inputs=[{"name": "model", "type": "MODEL", "link": 14}, {"name": "pulid", "type": "PULID", "link": 11}, {"name": "eva_clip", "type": "EVA_CLIP", "link": 12}, {"name": "face_analysis", "type": "FACEANALYSIS", "link": 13}, {"name": "image", "type": "IMAGE", "link": 15}, {"name": "attn_mask", "type": "MASK", "link": 16}], outputs=[{"name": "MODEL", "type": "MODEL", "links": [17]}], widgets_values=[pulid_weight, pulid_projection, pulid_fidelity, 0.0, 0.0, 1.0], title="Apply PuLID Identity"),
            node(18, "IPAdapterUnifiedLoader", [2940, 900], 17, inputs=[{"name": "model", "type": "MODEL", "link": 17}, {"name": "ipadapter", "type": "IPADAPTER", "link": None}], outputs=[{"name": "model", "type": "MODEL", "links": [18]}, {"name": "ipadapter", "type": "IPADAPTER", "links": [19]}], widgets_values=[ipadapter_preset], title="Load IP-Adapter Face Preset"),
            node(19, "IPAdapterAdvanced", [3300, 900], 18, inputs=[{"name": "model", "type": "MODEL", "link": 18}, {"name": "ipadapter", "type": "IPADAPTER", "link": 19}, {"name": "image", "type": "IMAGE", "link": 20}, {"name": "attn_mask", "type": "MASK", "link": 21}], outputs=[{"name": "MODEL", "type": "MODEL", "links": [25]}], widgets_values=[ipadapter_weight, ipadapter_weight_type, "concat", 0.0, ipadapter_end_at, "V only"], title="Apply IP-Adapter Face Appearance"),
            node(20, "VAEEncodeForInpaint", [2940, 1160], 19, inputs=[{"name": "pixels", "type": "IMAGE", "link": 22}, {"name": "vae", "type": "VAE", "link": 23}, {"name": "mask", "type": "MASK", "link": 24}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [28]}], widgets_values=[inpaint_grow_mask_by], title="Encode Target For Masked Inpaint"),
            node(21, "KSampler", [3660, 1040], 20, inputs=[{"name": "model", "type": "MODEL", "link": 25}, {"name": "positive", "type": "CONDITIONING", "link": 26}, {"name": "negative", "type": "CONDITIONING", "link": 27}, {"name": "latent_image", "type": "LATENT", "link": 28}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [29]}], widgets_values=[seed, "fixed", steps, cfg, sampler_name, scheduler, denoise], title="Sample Identity-Guided Inpaint"),
            node(22, "VAEDecode", [4020, 1040], 21, inputs=[{"name": "samples", "type": "LATENT", "link": 29}, {"name": "vae", "type": "VAE", "link": 30}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [31, 32]}], title="Decode Final"),
            node(23, "SaveImage", [4380, 960], 22, inputs=[{"name": "images", "type": "IMAGE", "link": 31}], widgets_values=[filename_prefix], title="Save Final Visual Prompt Result"),
            node(24, "PreviewImage", [4380, 1120], 23, inputs=[{"name": "images", "type": "IMAGE", "link": 32}], title="Preview Final"),
        ],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "note": (
                "Active visual-prompt stack: FaceSegmentation mask -> PuLID identity + "
                "IP-Adapter face guidance -> target-latent masked inpaint. SAM is installed "
                "but intentionally not on the hot path."
            )
        },
        "version": 0.4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="workflows/visual_prompt_hybrid_experiment_api.json")
    parser.add_argument("--ui-output", default="workflows/visual_prompt_hybrid_experiment_ui.json")
    parser.add_argument("--subject-image", default="subject_5 year curly.webp")
    parser.add_argument("--target-image", default="superman.png")
    parser.add_argument("--checkpoint", default="sd_xl_base_1.0_inpainting_0.1.safetensors")
    parser.add_argument(
        "--positive-prompt",
        default=(
            "comic book illustration, cinematic superhero portrait, preserve the source child identity exactly, "
            "curly dark hair, big toothy smile, warm brown skin tone, bright happy expression, "
            "young child face, rounded cheeks, clean facial features, cohesive lighting"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "adult face, target hairstyle, slicked hair, straight hair, flat hair, closed mouth, neutral expression, "
            "pale skin, desaturated skin, gray face plate, blank face, faceless, abstract artifacts, green glitch, blur, "
            "malformed eyes, seam, watermark, text"
        ),
    )
    parser.add_argument("--coarse-area", default="face+forehead (if available)")
    parser.add_argument("--coarse-grow", type=int, default=34)
    parser.add_argument("--coarse-blur", type=int, default=21)
    parser.add_argument("--preserve-expand", type=int, default=40)
    parser.add_argument("--preserve-feather", type=int, default=28)
    parser.add_argument("--pulid-weight", type=float, default=1.0)
    parser.add_argument("--pulid-projection", default="ortho_v2")
    parser.add_argument("--pulid-fidelity", type=int, default=8)
    parser.add_argument("--ipadapter-preset", default="PLUS FACE (portraits)")
    parser.add_argument("--ipadapter-weight", type=float, default=0.6)
    parser.add_argument("--ipadapter-weight-type", default="style transfer precise")
    parser.add_argument("--ipadapter-end-at", type=float, default=0.7)
    parser.add_argument("--inpaint-grow-mask-by", type=int, default=16)
    parser.add_argument("--seed", type=int, default=981273412)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--denoise", type=float, default=0.72)
    parser.add_argument("--sampler-name", default="dpmpp_2m_sde")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--filename-prefix", default="faceswap/visual_prompt_hybrid/final")
    parser.add_argument("--intermediate-prefix", default="faceswap/visual_prompt_hybrid/intermediate")
    args = parser.parse_args()

    workflow_args = vars(args).copy()
    output = Path(workflow_args.pop("output"))
    ui_output = Path(workflow_args.pop("ui_output"))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_workflow(**workflow_args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")

    ui_output.parent.mkdir(parents=True, exist_ok=True)
    ui_output.write_text(json.dumps(build_ui_workflow(**workflow_args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ui_output}")


if __name__ == "__main__":
    main()
