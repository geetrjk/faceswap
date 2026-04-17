#!/usr/bin/env python3
"""Generate an experimental SDXL + InstantID face-region workflow.

This graph is intentionally separate from the ReActor baseline. The subject
image is the identity anchor, the target image provides the preserved template
and face keypoints, and a soft target face mask confines InstantID to the
face/neck transition area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def api_node(class_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def build_workflow(
    *,
    subject_image: str,
    target_image: str,
    checkpoint: str,
    instantid_model: str,
    instantid_controlnet: str,
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    instantid_weight: float,
    pose_strength: float,
    instantid_start: float,
    instantid_end: float,
    instantid_noise: float,
    face_mask_area: str,
    face_mask_grow: int,
    face_mask_blur: int,
    inpaint_grow_mask_by: int,
    denoise: float,
    filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    return {
        "1": api_node("LoadImage", {"image": subject_image}),
        "2": api_node("LoadImage", {"image": target_image}),
        "3": api_node("SaveImage", {"images": ["1", 0], "filename_prefix": f"{intermediate_prefix}/subject_identity"}),
        "4": api_node("SaveImage", {"images": ["2", 0], "filename_prefix": f"{intermediate_prefix}/target_pose_style"}),
        "5": api_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint}),
        "6": api_node("CLIPTextEncode", {"clip": ["5", 1], "text": positive_prompt}),
        "7": api_node("CLIPTextEncode", {"clip": ["5", 1], "text": negative_prompt}),
        "8": api_node("InstantIDModelLoader", {"instantid_file": instantid_model}),
        "9": api_node("InstantIDFaceAnalysis", {"provider": "CUDA"}),
        "10": api_node("ControlNetLoader", {"control_net_name": instantid_controlnet}),
        "11": api_node("FaceKeypointsPreprocessor", {"faceanalysis": ["9", 0], "image": ["2", 0]}),
        "12": api_node("FaceAnalysisModels", {"library": "insightface", "provider": "CUDA"}),
        "13": api_node(
            "FaceSegmentation",
            {
                "analysis_models": ["12", 0],
                "image": ["2", 0],
                "area": face_mask_area,
                "grow": face_mask_grow,
                "grow_tapered": True,
                "blur": face_mask_blur,
            },
        ),
        "14": api_node("MaskToImage", {"mask": ["13", 0]}),
        "15": api_node("SaveImage", {"images": ["14", 0], "filename_prefix": f"{intermediate_prefix}/target_face_mask"}),
        "16": api_node(
            "ApplyInstantIDAdvanced",
            {
                "instantid": ["8", 0],
                "insightface": ["9", 0],
                "control_net": ["10", 0],
                "image": ["1", 0],
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "ip_weight": instantid_weight,
                "cn_strength": pose_strength,
                "start_at": instantid_start,
                "end_at": instantid_end,
                "noise": instantid_noise,
                "combine_embeds": "average",
                "image_kps": ["11", 0],
                "mask": ["13", 0],
            },
        ),
        "17": api_node(
            "VAEEncodeForInpaint",
            {
                "pixels": ["2", 0],
                "vae": ["5", 2],
                "mask": ["13", 0],
                "grow_mask_by": inpaint_grow_mask_by,
            },
        ),
        "18": api_node(
            "KSampler",
            {
                "model": ["16", 0],
                "positive": ["16", 1],
                "negative": ["16", 2],
                "latent_image": ["17", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        ),
        "19": api_node("VAEDecode", {"samples": ["18", 0], "vae": ["5", 2]}),
        "20": api_node("ImageCompositeMasked", {"destination": ["2", 0], "source": ["19", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["13", 0]}),
        "21": api_node("PreviewImage", {"images": ["11", 0]}),
        "22": api_node("SaveImage", {"images": ["11", 0], "filename_prefix": f"{intermediate_prefix}/target_face_keypoints"}),
        "23": api_node("PreviewImage", {"images": ["19", 0]}),
        "24": api_node("SaveImage", {"images": ["19", 0], "filename_prefix": f"{intermediate_prefix}/face_inpaint_raw"}),
        "25": api_node("PreviewImage", {"images": ["20", 0]}),
        "26": api_node("SaveImage", {"images": ["20", 0], "filename_prefix": filename_prefix}),
    }


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


def build_ui_workflow(
    *,
    subject_image: str,
    target_image: str,
    checkpoint: str,
    instantid_model: str,
    instantid_controlnet: str,
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    instantid_weight: float,
    pose_strength: float,
    instantid_start: float,
    instantid_end: float,
    instantid_noise: float,
    face_mask_area: str,
    face_mask_grow: int,
    face_mask_blur: int,
    inpaint_grow_mask_by: int,
    denoise: float,
    filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    links = [
        [1, 1, 0, 3, 0, "IMAGE"],
        [2, 2, 0, 4, 0, "IMAGE"],
        [3, 5, 1, 6, 0, "CLIP"],
        [4, 5, 1, 7, 0, "CLIP"],
        [5, 9, 0, 11, 0, "FACEANALYSIS"],
        [6, 2, 0, 11, 1, "IMAGE"],
        [7, 8, 0, 12, 0, "INSTANTID"],
        [8, 9, 0, 12, 1, "FACEANALYSIS"],
        [9, 10, 0, 12, 2, "CONTROL_NET"],
        [10, 1, 0, 12, 3, "IMAGE"],
        [11, 5, 0, 12, 4, "MODEL"],
        [12, 6, 0, 12, 5, "CONDITIONING"],
        [13, 7, 0, 12, 6, "CONDITIONING"],
        [14, 11, 0, 12, 7, "IMAGE"],
        [15, 13, 0, 14, 0, "MASK"],
        [16, 14, 0, 15, 0, "IMAGE"],
        [17, 12, 0, 16, 0, "MODEL"],
        [18, 12, 1, 16, 1, "CONDITIONING"],
        [19, 12, 2, 16, 2, "CONDITIONING"],
        [20, 2, 0, 15, 0, "IMAGE"],
        [21, 5, 2, 15, 1, "VAE"],
        [22, 26, 0, 15, 2, "MASK"],
        [23, 15, 0, 16, 3, "LATENT"],
        [24, 16, 0, 17, 0, "LATENT"],
        [25, 5, 2, 17, 1, "VAE"],
        [26, 2, 0, 18, 0, "IMAGE"],
        [27, 17, 0, 18, 1, "IMAGE"],
        [28, 26, 0, 18, 5, "MASK"],
        [29, 11, 0, 20, 0, "IMAGE"],
        [30, 11, 0, 21, 0, "IMAGE"],
        [31, 17, 0, 22, 0, "IMAGE"],
        [32, 17, 0, 23, 0, "IMAGE"],
        [33, 18, 0, 24, 0, "IMAGE"],
        [34, 18, 0, 25, 0, "IMAGE"],
        [36, 26, 0, 12, 8, "MASK"],
        [37, 25, 0, 26, 0, "ANALYSIS_MODELS"],
        [38, 2, 0, 26, 1, "IMAGE"],
        [39, 26, 0, 13, 0, "MASK"],
    ]
    return {
        "id": "instantid-face-region-ui",
        "revision": 0,
        "last_node_id": 26,
        "last_link_id": 39,
        "nodes": [
            node(1, "LoadImage", [80, 120], 0, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [1, 10]}], widgets_values=[subject_image, "image"], title="Load Subject Identity"),
            node(2, "LoadImage", [80, 420], 1, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [2, 6, 20, 26, 38]}], widgets_values=[target_image, "image"], title="Load Target Template"),
            node(3, "SaveImage", [450, 120], 2, inputs=[{"name": "images", "type": "IMAGE", "link": 1}], widgets_values=[f"{intermediate_prefix}/subject_identity"], title="Save Subject Input"),
            node(4, "SaveImage", [450, 420], 3, inputs=[{"name": "images", "type": "IMAGE", "link": 2}], widgets_values=[f"{intermediate_prefix}/target_pose_style"], title="Save Target Input"),
            node(5, "CheckpointLoaderSimple", [80, 760], 4, outputs=[{"name": "MODEL", "type": "MODEL", "links": [11]}, {"name": "CLIP", "type": "CLIP", "links": [3, 4]}, {"name": "VAE", "type": "VAE", "links": [21, 25]}], widgets_values=[checkpoint], title="Load SDXL Checkpoint"),
            node(6, "CLIPTextEncode", [450, 700], 5, inputs=[{"name": "clip", "type": "CLIP", "link": 3}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [12]}], widgets_values=[positive_prompt], title="Positive Prompt"),
            node(7, "CLIPTextEncode", [450, 900], 6, inputs=[{"name": "clip", "type": "CLIP", "link": 4}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [13]}], widgets_values=[negative_prompt], title="Negative Prompt"),
            node(8, "InstantIDModelLoader", [820, 80], 7, outputs=[{"name": "INSTANTID", "type": "INSTANTID", "links": [7]}], widgets_values=[instantid_model], title="Load InstantID Adapter"),
            node(9, "InstantIDFaceAnalysis", [820, 240], 8, outputs=[{"name": "FACEANALYSIS", "type": "FACEANALYSIS", "links": [5, 8]}], widgets_values=["CUDA"], title="Load AntelopeV2 Face Analysis"),
            node(10, "ControlNetLoader", [820, 400], 9, outputs=[{"name": "CONTROL_NET", "type": "CONTROL_NET", "links": [9]}], widgets_values=[instantid_controlnet], title="Load InstantID ControlNet"),
            node(11, "FaceKeypointsPreprocessor", [820, 580], 10, inputs=[{"name": "faceanalysis", "type": "FACEANALYSIS", "link": 5}, {"name": "image", "type": "IMAGE", "link": 6}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [14, 29, 30]}], title="Extract Target Face Keypoints"),
            node(12, "ApplyInstantIDAdvanced", [1320, 340], 11, inputs=[{"name": "instantid", "type": "INSTANTID", "link": 7}, {"name": "insightface", "type": "FACEANALYSIS", "link": 8}, {"name": "control_net", "type": "CONTROL_NET", "link": 9}, {"name": "image", "type": "IMAGE", "link": 10}, {"name": "model", "type": "MODEL", "link": 11}, {"name": "positive", "type": "CONDITIONING", "link": 12}, {"name": "negative", "type": "CONDITIONING", "link": 13}, {"name": "image_kps", "type": "IMAGE", "link": 14}, {"name": "mask", "type": "MASK", "link": 36}], outputs=[{"name": "MODEL", "type": "MODEL", "links": [17]}, {"name": "positive", "type": "CONDITIONING", "links": [18]}, {"name": "negative", "type": "CONDITIONING", "links": [19]}], widgets_values=[instantid_weight, pose_strength, instantid_start, instantid_end, instantid_noise, "average"], title="Apply InstantID Only Inside Face Mask"),
            node(13, "MaskToImage", [1300, 820], 12, inputs=[{"name": "mask", "type": "MASK", "link": 39}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [16]}], title="Convert Face Mask To Image"),
            node(14, "SaveImage", [1650, 820], 13, inputs=[{"name": "images", "type": "IMAGE", "link": 16}], widgets_values=[f"{intermediate_prefix}/target_face_mask"], title="Save Target Face Mask"),
            node(15, "VAEEncodeForInpaint", [1700, 520], 14, inputs=[{"name": "pixels", "type": "IMAGE", "link": 20}, {"name": "vae", "type": "VAE", "link": 21}, {"name": "mask", "type": "MASK", "link": 22}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [23]}], widgets_values=[inpaint_grow_mask_by], title="Encode Target With Face Noise Mask"),
            node(16, "KSampler", [2100, 420], 15, inputs=[{"name": "model", "type": "MODEL", "link": 17}, {"name": "positive", "type": "CONDITIONING", "link": 18}, {"name": "negative", "type": "CONDITIONING", "link": 19}, {"name": "latent_image", "type": "LATENT", "link": 23}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [24]}], widgets_values=[seed, "fixed", steps, cfg, sampler_name, scheduler, denoise], title="Inpaint Face Region"),
            node(17, "VAEDecode", [2480, 420], 16, inputs=[{"name": "samples", "type": "LATENT", "link": 24}, {"name": "vae", "type": "VAE", "link": 25}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [27, 31, 32]}], title="Decode Face Inpaint"),
            node(18, "ImageCompositeMasked", [2860, 420], 17, inputs=[{"name": "destination", "type": "IMAGE", "link": 26}, {"name": "source", "type": "IMAGE", "link": 27}, {"name": "mask", "type": "MASK", "link": 28}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [33, 34]}], widgets_values=[0, 0, False], title="Composite Face Onto Unchanged Target"),
            node(19, "PreviewImage", [1220, 620], 18, inputs=[{"name": "images", "type": "IMAGE", "link": 29}], title="Preview Target Keypoints"),
            node(20, "SaveImage", [1220, 760], 19, inputs=[{"name": "images", "type": "IMAGE", "link": 30}], widgets_values=[f"{intermediate_prefix}/target_face_keypoints"], title="Save Target Keypoints"),
            node(21, "PreviewImage", [2860, 700], 20, inputs=[{"name": "images", "type": "IMAGE", "link": 31}], title="Preview Raw Face Inpaint"),
            node(22, "SaveImage", [2860, 860], 21, inputs=[{"name": "images", "type": "IMAGE", "link": 32}], widgets_values=[f"{intermediate_prefix}/face_inpaint_raw"], title="Save Raw Face Inpaint"),
            node(23, "PreviewImage", [3260, 360], 22, inputs=[{"name": "images", "type": "IMAGE", "link": 33}], title="Preview Final Composite"),
            node(24, "SaveImage", [3260, 540], 23, inputs=[{"name": "images", "type": "IMAGE", "link": 34}], widgets_values=[filename_prefix], title="Save Final Face-Only Composite"),
            node(25, "FaceAnalysisModels", [820, 760], 24, outputs=[{"name": "ANALYSIS_MODELS", "type": "ANALYSIS_MODELS", "links": [37]}], widgets_values=["insightface", "CUDA"], title="Load Face Mask Analysis"),
            node(26, "FaceSegmentation", [1020, 900], 25, inputs=[{"name": "analysis_models", "type": "ANALYSIS_MODELS", "link": 37}, {"name": "image", "type": "IMAGE", "link": 38}], outputs=[{"name": "mask", "type": "MASK", "links": [22, 28, 36, 39]}, {"name": "image", "type": "IMAGE", "links": []}, {"name": "seg_mask", "type": "MASK", "links": []}, {"name": "seg_image", "type": "IMAGE", "links": []}, {"name": "x", "type": "INT", "links": []}, {"name": "y", "type": "INT", "links": []}, {"name": "width", "type": "INT", "links": []}, {"name": "height", "type": "INT", "links": []}], widgets_values=[face_mask_area, face_mask_grow, True, face_mask_blur], title="Create Soft Target Face Mask"),
        ],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="workflows/instantid_subject_pose_style_api.json")
    parser.add_argument("--ui-output", default="workflows/instantid_subject_pose_style_ui.json")
    parser.add_argument("--subject-image", default="subject_5 year curly.webp")
    parser.add_argument("--target-image", default="superman.png")
    parser.add_argument("--checkpoint", default="sd_xl_base_1.0_inpainting_0.1.safetensors")
    parser.add_argument("--instantid-model", default="ip-adapter.bin")
    parser.add_argument("--instantid-controlnet", default="instantid_controlnet.safetensors")
    parser.add_argument(
        "--positive-prompt",
        default=(
            "the same target character and body, only the face replaced by a young child with curly hair, "
            "preserve the target costume, body, pose, background, lighting, and composition, "
            "seamless face and neck blend, preserve the target illustration style"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "changed body, changed costume, changed cape, changed pose, changed background, full body transformation, "
            "target face identity, target-specific facial structure, target-specific hairstyle, adult face, older person, "
            "beard, mustache, strong jawline, hard mask edge, visible seam, distorted face, blank face, gray face, "
            "missing eyes, missing mouth, faceless, mask plate, flat color, watermark, text"
        ),
    )
    parser.add_argument("--steps", type=int, default=34)
    parser.add_argument("--cfg", type=float, default=5.2)
    parser.add_argument("--sampler-name", default="dpmpp_2m_sde")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--seed", type=int, default=470391928)
    parser.add_argument("--instantid-weight", type=float, default=1.35)
    parser.add_argument("--pose-strength", type=float, default=0.2)
    parser.add_argument("--instantid-start", type=float, default=0.0)
    parser.add_argument("--instantid-end", type=float, default=0.82)
    parser.add_argument("--instantid-noise", type=float, default=0.25)
    parser.add_argument("--face-mask-area", default="face+forehead (if available)")
    parser.add_argument("--face-mask-grow", type=int, default=16)
    parser.add_argument("--face-mask-blur", type=int, default=31)
    parser.add_argument("--inpaint-grow-mask-by", type=int, default=12)
    parser.add_argument("--denoise", type=float, default=0.86)
    parser.add_argument("--filename-prefix", default="faceswap/instantid/final")
    parser.add_argument("--intermediate-prefix", default="faceswap/instantid/intermediate")
    args = parser.parse_args()

    workflow_args = vars(args).copy()
    output = Path(workflow_args.pop("output"))
    ui_output = Path(workflow_args.pop("ui_output"))

    workflow = build_workflow(**workflow_args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")

    ui_workflow = build_ui_workflow(**workflow_args)
    ui_output.parent.mkdir(parents=True, exist_ok=True)
    ui_output.write_text(json.dumps(ui_workflow, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ui_output}")


if __name__ == "__main__":
    main()
