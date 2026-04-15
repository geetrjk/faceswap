#!/usr/bin/env python3
"""Generate a sidecar ReActor swap-and-bake inpaint experiment."""

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
    swap_model: str,
    checkpoint: str,
    positive_prompt: str,
    negative_prompt: str,
    face_mask_area: str,
    face_mask_grow: int,
    face_mask_blur: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    return {
        "1": api_node("LoadImage", {"image": subject_image}),
        "2": api_node("LoadImage", {"image": target_image}),
        "3": api_node(
            "ReActorFaceSwap",
            {
                "enabled": True,
                "input_image": link(2, 0),
                "source_image": link(1, 0),
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "none",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "swap_model": swap_model,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
            },
        ),
        "13": api_node("SaveImage", {"images": link(3, 0), "filename_prefix": f"{intermediate_prefix}/reactor_swap"}),
        "4": api_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint}),
        "5": api_node("CLIPTextEncode", {"clip": link(4, 1), "text": positive_prompt}),
        "6": api_node("CLIPTextEncode", {"clip": link(4, 1), "text": negative_prompt}),
        "7": api_node("FaceAnalysisModels", {"library": "insightface", "provider": "CUDA"}),
        "8": api_node(
            "FaceSegmentation",
            {
                "analysis_models": link(7, 0),
                "image": link(3, 0),
                "area": face_mask_area,
                "grow": face_mask_grow,
                "grow_tapered": True,
                "blur": face_mask_blur,
            },
        ),
        "14": api_node("MaskToImage", {"mask": link(8, 0)}),
        "15": api_node("SaveImage", {"images": link(14, 0), "filename_prefix": f"{intermediate_prefix}/bake_mask"}),
        "9": api_node("VAEEncode", {"pixels": link(3, 0), "vae": link(4, 2)}),
        "10": api_node(
            "KSampler",
            {
                "model": link(4, 0),
                "positive": link(5, 0),
                "negative": link(6, 0),
                "latent_image": link(9, 0),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        ),
        "11": api_node("VAEDecode", {"samples": link(10, 0), "vae": link(4, 2)}),
        "12": api_node("SaveImage", {"images": link(11, 0), "filename_prefix": filename_prefix}),
    }


def build_ui_workflow(
    *,
    subject_image: str,
    target_image: str,
    swap_model: str,
    checkpoint: str,
    positive_prompt: str,
    negative_prompt: str,
    face_mask_area: str,
    face_mask_grow: int,
    face_mask_blur: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    links = [
        [1, 1, 0, 3, 1, "IMAGE"],
        [2, 2, 0, 3, 0, "IMAGE"],
        [3, 3, 0, 13, 0, "IMAGE"],
        [4, 4, 1, 5, 0, "CLIP"],
        [5, 4, 1, 6, 0, "CLIP"],
        [6, 7, 0, 8, 0, "ANALYSIS_MODELS"],
        [7, 3, 0, 8, 1, "IMAGE"],
        [8, 8, 0, 14, 0, "MASK"],
        [9, 14, 0, 15, 0, "IMAGE"],
        [10, 3, 0, 9, 0, "IMAGE"],
        [11, 4, 2, 9, 1, "VAE"],
        [13, 4, 0, 10, 0, "MODEL"],
        [14, 5, 0, 10, 1, "CONDITIONING"],
        [15, 6, 0, 10, 2, "CONDITIONING"],
        [16, 9, 0, 10, 3, "LATENT"],
        [17, 10, 0, 11, 0, "LATENT"],
        [18, 4, 2, 11, 1, "VAE"],
        [19, 11, 0, 12, 0, "IMAGE"],
        [20, 11, 0, 16, 0, "IMAGE"],
    ]
    return {
        "id": "swap-and-bake-experiment-ui",
        "revision": 0,
        "last_node_id": 16,
        "last_link_id": 20,
        "nodes": [
            node(1, "LoadImage", [80, 120], 0, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [1]}], widgets_values=[subject_image, "image"], title="Load Subject Identity"),
            node(2, "LoadImage", [80, 420], 1, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [2]}], widgets_values=[target_image, "image"], title="Load Target Template"),
            node(
                3,
                "ReActorFaceSwap",
                [450, 320],
                2,
                inputs=[
                    {"name": "input_image", "type": "IMAGE", "link": 2},
                    {"name": "source_image", "type": "IMAGE", "link": 1},
                ],
                outputs=[{"name": "SWAPPED_IMAGE", "type": "IMAGE", "links": [3, 7, 10]}],
                widgets_values=[True, swap_model, "retinaface_resnet50", "none", 1.0, 0.5, "no", "no", "0", "0", 1],
                title="ReActor Identity Swap",
            ),
            node(13, "SaveImage", [820, 220], 3, inputs=[{"name": "images", "type": "IMAGE", "link": 3}], widgets_values=[f"{intermediate_prefix}/reactor_swap"], title="Save ReActor Swap"),
            node(4, "CheckpointLoaderSimple", [80, 760], 4, outputs=[{"name": "MODEL", "type": "MODEL", "links": [13]}, {"name": "CLIP", "type": "CLIP", "links": [4, 5]}, {"name": "VAE", "type": "VAE", "links": [11, 18]}], widgets_values=[checkpoint], title="Load Inpaint Checkpoint"),
            node(5, "CLIPTextEncode", [450, 700], 5, inputs=[{"name": "clip", "type": "CLIP", "link": 4}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [14]}], widgets_values=[positive_prompt], title="Positive Bake Prompt"),
            node(6, "CLIPTextEncode", [450, 900], 6, inputs=[{"name": "clip", "type": "CLIP", "link": 5}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [15]}], widgets_values=[negative_prompt], title="Negative Bake Prompt"),
            node(7, "FaceAnalysisModels", [820, 520], 7, outputs=[{"name": "ANALYSIS_MODELS", "type": "ANALYSIS_MODELS", "links": [6]}], widgets_values=["insightface", "CUDA"], title="Load Face Analysis"),
            node(8, "FaceSegmentation", [1180, 420], 8, inputs=[{"name": "analysis_models", "type": "ANALYSIS_MODELS", "link": 6}, {"name": "image", "type": "IMAGE", "link": 7}], outputs=[{"name": "mask", "type": "MASK", "links": [8]}, {"name": "image", "type": "IMAGE", "links": []}, {"name": "seg_mask", "type": "MASK", "links": []}, {"name": "seg_image", "type": "IMAGE", "links": []}, {"name": "x", "type": "INT", "links": []}, {"name": "y", "type": "INT", "links": []}, {"name": "width", "type": "INT", "links": []}, {"name": "height", "type": "INT", "links": []}], widgets_values=[face_mask_area, face_mask_grow, True, face_mask_blur], title="Find Swapped Face Mask"),
            node(14, "MaskToImage", [1540, 420], 9, inputs=[{"name": "mask", "type": "MASK", "link": 8}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [9]}], title="Convert Bake Mask"),
            node(15, "SaveImage", [1900, 420], 10, inputs=[{"name": "images", "type": "IMAGE", "link": 9}], widgets_values=[f"{intermediate_prefix}/bake_mask"], title="Save Bake Mask"),
            node(9, "VAEEncode", [1180, 700], 11, inputs=[{"name": "pixels", "type": "IMAGE", "link": 10}, {"name": "vae", "type": "VAE", "link": 11}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [16]}], title="Encode Swap For Low-Denoise Bake"),
            node(10, "KSampler", [1540, 700], 12, inputs=[{"name": "model", "type": "MODEL", "link": 13}, {"name": "positive", "type": "CONDITIONING", "link": 14}, {"name": "negative", "type": "CONDITIONING", "link": 15}, {"name": "latent_image", "type": "LATENT", "link": 16}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [17]}], widgets_values=[seed, "fixed", steps, cfg, sampler_name, scheduler, denoise], title="Bake Swap"),
            node(11, "VAEDecode", [1900, 700], 13, inputs=[{"name": "samples", "type": "LATENT", "link": 17}, {"name": "vae", "type": "VAE", "link": 18}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [19, 20]}], title="Decode Baked Swap"),
            node(12, "SaveImage", [2260, 620], 14, inputs=[{"name": "images", "type": "IMAGE", "link": 19}], widgets_values=[filename_prefix], title="Save Final Baked Swap"),
            node(16, "PreviewImage", [2260, 780], 15, inputs=[{"name": "images", "type": "IMAGE", "link": 20}], title="Preview Final Baked Swap"),
        ],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="workflows/swap_and_bake_experiment_api.json")
    parser.add_argument("--ui-output", default="workflows/swap_and_bake_experiment_ui.json")
    parser.add_argument("--subject-image", default="subject_5 year curly.webp")
    parser.add_argument("--target-image", default="superman.png")
    parser.add_argument("--swap-model", default="inswapper_128.onnx")
    parser.add_argument("--checkpoint", default="sd_xl_base_1.0_inpainting_0.1.safetensors")
    parser.add_argument(
        "--positive-prompt",
        default="comic book illustration, high quality, cinematic lighting, perfectly blended face, coherent shadows, cohesive art style",
    )
    parser.add_argument(
        "--negative-prompt",
        default="realistic photograph, visible seam, mismatched skin tone, distorted face, abstract artifacts, green glitch, latent noise, poorly drawn",
    )
    parser.add_argument("--face-mask-area", default="face")
    parser.add_argument("--face-mask-grow", type=int, default=24)
    parser.add_argument("--face-mask-blur", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--sampler-name", default="dpmpp_2m_sde")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--denoise", type=float, default=0.2)
    parser.add_argument("--filename-prefix", default="faceswap/swap_and_bake/final")
    parser.add_argument("--intermediate-prefix", default="faceswap/swap_and_bake/intermediate")
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
