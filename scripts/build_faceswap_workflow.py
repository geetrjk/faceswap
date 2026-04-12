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
    subject_image: str,
    target_image: str,
    swap_model: str,
    face_restore_model: str,
    face_restore_visibility: float,
    filename_prefix: str,
    face_boost: bool,
    face_boost_visibility: float,
    intermediate_prefix: str,
) -> dict:
    # ComfyUI API prompt format: each node id maps to class_type + inputs.
    # Custom nodes expected:
    # - ReActorFaceSwap (Gourieff/comfyui-reactor-node)
    workflow = {
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
            "inputs": {"images": ["1", 0]},
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["1", 0],
                "filename_prefix": f"{intermediate_prefix}/subject_input",
            },
        },
        "5": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["2", 0]},
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["2", 0],
                "filename_prefix": f"{intermediate_prefix}/target_input",
            },
        },
        "7": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["2", 0],
                "source_image": ["1", 0],
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
        },
        "8": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["7", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": f"{intermediate_prefix}/plain_swap",
            },
        },
        "10": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["2", 0],
                "source_image": ["1", 0],
                "facedetection": "retinaface_resnet50",
                "face_restore_model": face_restore_model,
                "face_restore_visibility": face_restore_visibility,
                "codeformer_weight": 0.5,
                "swap_model": swap_model,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
            },
        },
        "11": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["10", 0]},
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }
    if face_boost:
        workflow["10"]["inputs"]["face_boost"] = ["13", 0]
        workflow["13"] = {
            "class_type": "ReActorFaceBoost",
            "inputs": {
                "enabled": True,
                "boost_model": face_restore_model,
                "interpolation": "Bicubic",
                "visibility": face_boost_visibility,
                "codeformer_weight": 0.5,
                "restore_with_main_after": False,
            },
        }
    return workflow


def node(
    node_id: int,
    node_type: str,
    pos: list[int],
    order: int,
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
    widgets_values: list | None = None,
    title: str | None = None,
) -> dict:
    data = {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": [315, 120],
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
    subject_image: str,
    target_image: str,
    swap_model: str,
    face_restore_model: str,
    face_restore_visibility: float,
    filename_prefix: str,
    face_boost: bool,
    face_boost_visibility: float,
    intermediate_prefix: str,
) -> dict:
    # ComfyUI UI graph format. This is what the browser needs for manual runs.
    links = [
        [1, 1, 0, 3, 0, "IMAGE"],
        [2, 1, 0, 4, 0, "IMAGE"],
        [3, 1, 0, 7, 1, "IMAGE"],
        [4, 1, 0, 10, 1, "IMAGE"],
        [5, 2, 0, 5, 0, "IMAGE"],
        [6, 2, 0, 6, 0, "IMAGE"],
        [7, 2, 0, 7, 0, "IMAGE"],
        [8, 2, 0, 10, 0, "IMAGE"],
        [9, 7, 0, 8, 0, "IMAGE"],
        [10, 7, 0, 9, 0, "IMAGE"],
        [11, 10, 0, 11, 0, "IMAGE"],
        [12, 10, 0, 12, 0, "IMAGE"],
    ]
    face_boost_input = []
    if face_boost:
        links.append([13, 13, 0, 10, 2, "FACE_BOOST"])
        face_boost_input = [{"name": "face_boost", "type": "FACE_BOOST", "link": 13}]

    return {
        "id": "faceswap-subject-on-character-ui",
        "revision": 0,
        "last_node_id": 13 if face_boost else 12,
        "last_link_id": 13 if face_boost else 12,
        "nodes": [
            node(
                1,
                "LoadImage",
                [80, 160],
                0,
                outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [1, 2, 3, 4]}],
                widgets_values=[subject_image, "image"],
                title="Load Subject Identity",
            ),
            node(
                2,
                "LoadImage",
                [80, 520],
                1,
                outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [5, 6, 7, 8]}],
                widgets_values=[target_image, "image"],
                title="Load Target Character",
            ),
            node(
                3,
                "PreviewImage",
                [450, 80],
                2,
                inputs=[{"name": "images", "type": "IMAGE", "link": 1}],
                title="Preview Subject Input",
            ),
            node(
                4,
                "SaveImage",
                [450, 220],
                3,
                inputs=[{"name": "images", "type": "IMAGE", "link": 2}],
                widgets_values=[f"{intermediate_prefix}/subject_input"],
                title="Save Subject Input",
            ),
            node(
                5,
                "PreviewImage",
                [450, 460],
                4,
                inputs=[{"name": "images", "type": "IMAGE", "link": 5}],
                title="Preview Target Input",
            ),
            node(
                6,
                "SaveImage",
                [450, 600],
                5,
                inputs=[{"name": "images", "type": "IMAGE", "link": 6}],
                widgets_values=[f"{intermediate_prefix}/target_input"],
                title="Save Target Input",
            ),
            node(
                7,
                "ReActorFaceSwap",
                [820, 420],
                6,
                inputs=[
                    {"name": "input_image", "type": "IMAGE", "link": 7},
                    {"name": "source_image", "type": "IMAGE", "link": 3},
                ],
                outputs=[{"name": "SWAPPED_IMAGE", "type": "IMAGE", "links": [9, 10]}],
                widgets_values=[
                    True,
                    swap_model,
                    "retinaface_resnet50",
                    "none",
                    1.0,
                    0.5,
                    "no",
                    "no",
                    "0",
                    "0",
                    1,
                ],
                title="Plain Swap Diagnostic",
            ),
            node(
                8,
                "PreviewImage",
                [1180, 360],
                7,
                inputs=[{"name": "images", "type": "IMAGE", "link": 9}],
                title="Preview Plain Swap",
            ),
            node(
                9,
                "SaveImage",
                [1180, 500],
                8,
                inputs=[{"name": "images", "type": "IMAGE", "link": 10}],
                widgets_values=[f"{intermediate_prefix}/plain_swap"],
                title="Save Plain Swap",
            ),
            node(
                10,
                "ReActorFaceSwap",
                [820, 780],
                9,
                inputs=[
                    {"name": "input_image", "type": "IMAGE", "link": 8},
                    {"name": "source_image", "type": "IMAGE", "link": 4},
                    *face_boost_input,
                ],
                outputs=[{"name": "SWAPPED_IMAGE", "type": "IMAGE", "links": [11, 12]}],
                widgets_values=[
                    True,
                    swap_model,
                    "retinaface_resnet50",
                    face_restore_model,
                    face_restore_visibility,
                    0.5,
                    "no",
                    "no",
                    "0",
                    "0",
                    1,
                ],
                title="Boosted Final Swap",
            ),
            node(
                11,
                "PreviewImage",
                [1180, 740],
                10,
                inputs=[{"name": "images", "type": "IMAGE", "link": 11}],
                title="Preview Final",
            ),
            node(
                12,
                "SaveImage",
                [1180, 880],
                11,
                inputs=[{"name": "images", "type": "IMAGE", "link": 12}],
                widgets_values=[filename_prefix],
                title="Save Final",
            ),
            *(
                [
                    node(
                        13,
                        "ReActorFaceBoost",
                        [450, 800],
                        12,
                        outputs=[{"name": "FACE_BOOST", "type": "FACE_BOOST", "links": [13]}],
                        widgets_values=[True, face_restore_model, "Bicubic", face_boost_visibility, 0.5, False],
                        title="FaceBoost Settings",
                    )
                ]
                if face_boost
                else []
            ),
        ],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="workflows/faceswap_subject_on_character_api.json")
    parser.add_argument("--ui-output", default="workflows/faceswap_subject_on_character_ui.json")
    parser.add_argument("--subject-image", default="subject_5 year curly.webp")
    parser.add_argument("--target-image", default="superman.png")
    parser.add_argument("--swap-model", default="inswapper_128.onnx")
    parser.add_argument("--face-restore-model", default="GFPGANv1.4.pth")
    parser.add_argument("--face-restore-visibility", type=float, default=1.0)
    parser.add_argument("--filename-prefix", default="faceswap/final")
    parser.add_argument("--no-face-boost", dest="face_boost", action="store_false")
    parser.set_defaults(face_boost=True)
    parser.add_argument("--face-boost-visibility", type=float, default=1.0)
    parser.add_argument("--intermediate-prefix", default="faceswap/intermediate")
    args = parser.parse_args()

    workflow = build_workflow(
        subject_image=args.subject_image,
        target_image=args.target_image,
        swap_model=args.swap_model,
        face_restore_model=args.face_restore_model,
        face_restore_visibility=args.face_restore_visibility,
        filename_prefix=args.filename_prefix,
        face_boost=args.face_boost,
        face_boost_visibility=args.face_boost_visibility,
        intermediate_prefix=args.intermediate_prefix,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

    ui_workflow = build_ui_workflow(
        subject_image=args.subject_image,
        target_image=args.target_image,
        swap_model=args.swap_model,
        face_restore_model=args.face_restore_model,
        face_restore_visibility=args.face_restore_visibility,
        filename_prefix=args.filename_prefix,
        face_boost=args.face_boost,
        face_boost_visibility=args.face_boost_visibility,
        intermediate_prefix=args.intermediate_prefix,
    )
    ui_out = Path(args.ui_output)
    ui_out.parent.mkdir(parents=True, exist_ok=True)
    ui_out.write_text(json.dumps(ui_workflow, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ui_out}")


if __name__ == "__main__":
    main()
