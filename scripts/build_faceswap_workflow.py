#!/usr/bin/env python3
"""Generate a ComfyUI API workflow for subject->character face swapping.

This workflow is intentionally compact and includes explicit intermediate
preview/save stages so the face detection and swap can be verified quickly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_workflow(
    checkpoint: str,
    vae: str,
    subject_image: str,
    target_image: str,
    positive_prompt: str,
    negative_prompt: str,
) -> dict:
    # ComfyUI API prompt format: each node id maps to class_type + inputs.
    # Custom nodes expected:
    # - ReActorFaceSwap (Gourieff/comfyui-reactor-node)
    # - UltralyticsDetectorProvider + FaceDetailer (ComfyUI-Impact-Pack)
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": subject_image},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": target_image},
        },
        "3": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["2", 0]},
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": positive_prompt,
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": negative_prompt,
            },
        },
        "8": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["5", 0],
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": 2121212121,
                "steps": 18,
                "cfg": 3.8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.35,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["5", 0],
            },
        },
        "11": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {
                "model_name": "face_yolov8m.pt",
            },
        },
        "12": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["10", 0],
                "source_image": ["1", 0],
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "GFPGANv1.4.pth",
                "face_restore_visibility": 0.7,
                "codeformer_weight": 0.5,
                "swap_model": "inswapper_128.onnx",
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
            },
        },
        "13": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["12", 0]},
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["12", 0],
                "filename_prefix": "faceswap/final",
            },
        },
        "15": {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["12", 0],
                "model": ["4", 0],
                "clip": ["4", 1],
                "vae": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "bbox_detector": ["11", 0],
                "sam_model_opt": None,
                "segm_detector_opt": None,
                "guide_size": 512,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": 222222,
                "steps": 10,
                "cfg": 3.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.25,
                "feather": 8,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.35,
                "bbox_dilation": 18,
                "bbox_crop_factor": 2.2,
            },
        },
        "16": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["15", 0]},
        },
        "17": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["15", 0],
                "filename_prefix": "faceswap/refined",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="workflows/faceswap_subject_on_character_api.json")
    parser.add_argument("--checkpoint", default="flux1-dev-fp8.safetensors")
    parser.add_argument("--vae", default="ae.safetensors")
    parser.add_argument("--subject-image", default="subject.jpg")
    parser.add_argument("--target-image", default="target.png")
    parser.add_argument(
        "--positive-prompt",
        default=(
            "high detail portrait, keep target scene composition, preserve costume and background, "
            "identity-consistent face"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default="lowres, blurry, deformed face, extra eyes, bad anatomy",
    )
    args = parser.parse_args()

    workflow = build_workflow(
        checkpoint=args.checkpoint,
        vae=args.vae,
        subject_image=args.subject_image,
        target_image=args.target_image,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
