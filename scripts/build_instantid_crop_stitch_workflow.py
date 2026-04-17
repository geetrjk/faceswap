#!/usr/bin/env python3
"""Generate an InstantID crop-first stitch-back experiment.

This workflow keeps the existing InstantID and ReActor workflows intact. It
uses FaceSegmentation to find a generous target face/head region, crops that
region, runs InstantID inpainting on the crop, and composites the edited crop
back into the full target image.
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


def build_workflow(
    *,
    subject_image: str,
    target_image: str,
    checkpoint: str,
    instantid_model: str,
    instantid_controlnet: str,
    structural_controlnet: str,
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
    structural_strength: float,
    structural_start: float,
    structural_end: float,
    canny_low_threshold: float,
    canny_high_threshold: float,
    face_mask_area: str,
    face_mask_grow: int,
    face_mask_blur: int,
    crop_mask_shrink: int,
    inpaint_grow_mask_by: int,
    denoise: float,
    filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    # FaceSegmentation outputs: mask, image, seg_mask, seg_image, x, y, width, height.
    bbox_x = link(13, 4)
    bbox_y = link(13, 5)
    bbox_width = link(13, 6)
    bbox_height = link(13, 7)

    return {
        "1": api_node("LoadImage", {"image": subject_image}),
        "2": api_node("LoadImage", {"image": target_image}),
        "3": api_node("SaveImage", {"images": link(1, 0), "filename_prefix": f"{intermediate_prefix}/subject_identity"}),
        "4": api_node("SaveImage", {"images": link(2, 0), "filename_prefix": f"{intermediate_prefix}/target_full"}),
        "5": api_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint}),
        "6": api_node("CLIPTextEncode", {"clip": link(5, 1), "text": positive_prompt}),
        "7": api_node("CLIPTextEncode", {"clip": link(5, 1), "text": negative_prompt}),
        "8": api_node("InstantIDModelLoader", {"instantid_file": instantid_model}),
        "9": api_node("InstantIDFaceAnalysis", {"provider": "CUDA"}),
        "10": api_node("ControlNetLoader", {"control_net_name": instantid_controlnet}),
        "11": api_node("FaceAnalysisModels", {"library": "insightface", "provider": "CUDA"}),
        "36": api_node(
            "Canny",
            {"image": link(16, 0), "low_threshold": canny_low_threshold, "high_threshold": canny_high_threshold},
        ),
        "37": api_node("SaveImage", {"images": link(36, 0), "filename_prefix": f"{intermediate_prefix}/target_crop_canny"}),
        "38": api_node("ControlNetLoader", {"control_net_name": structural_controlnet}),
        "13": api_node(
            "FaceSegmentation",
            {
                "analysis_models": link(11, 0),
                "image": link(2, 0),
                "area": face_mask_area,
                "grow": face_mask_grow,
                "grow_tapered": True,
                "blur": face_mask_blur,
            },
        ),
        "14": api_node("MaskToImage", {"mask": link(13, 0)}),
        "15": api_node("SaveImage", {"images": link(14, 0), "filename_prefix": f"{intermediate_prefix}/target_full_mask"}),
        "16": api_node("ImageCrop", {"image": link(2, 0), "width": bbox_width, "height": bbox_height, "x": bbox_x, "y": bbox_y}),
        "17": api_node("SaveImage", {"images": link(16, 0), "filename_prefix": f"{intermediate_prefix}/target_crop"}),
        "18": api_node("CropMask", {"mask": link(13, 0), "x": bbox_x, "y": bbox_y, "width": bbox_width, "height": bbox_height}),
        "19": api_node("MaskToImage", {"mask": link(18, 0)}),
        "20": api_node("SaveImage", {"images": link(19, 0), "filename_prefix": f"{intermediate_prefix}/target_crop_region_mask"}),
        "21": api_node("FaceKeypointsPreprocessor", {"faceanalysis": link(9, 0), "image": link(16, 0)}),
        "22": api_node("SaveImage", {"images": link(21, 0), "filename_prefix": f"{intermediate_prefix}/target_crop_keypoints"}),
        "33": api_node("GrowMask", {"mask": link(18, 0), "expand": crop_mask_shrink, "tapered_corners": True}),
        "34": api_node("MaskToImage", {"mask": link(33, 0)}),
        "35": api_node("SaveImage", {"images": link(34, 0), "filename_prefix": f"{intermediate_prefix}/target_crop_edit_mask"}),
        "23": api_node(
            "ApplyInstantIDAdvanced",
            {
                "instantid": link(8, 0),
                "insightface": link(9, 0),
                "control_net": link(10, 0),
                "image": link(1, 0),
                "model": link(5, 0),
                "positive": link(6, 0),
                "negative": link(7, 0),
                "ip_weight": instantid_weight,
                "cn_strength": pose_strength,
                "start_at": instantid_start,
                "end_at": instantid_end,
                "noise": instantid_noise,
                "combine_embeds": "average",
                "image_kps": link(21, 0),
                "mask": link(33, 0),
            },
        ),
        "39": api_node(
            "ControlNetApplyAdvanced",
            {
                "positive": link(23, 1),
                "negative": link(23, 2),
                "control_net": link(38, 0),
                "image": link(36, 0),
                "strength": structural_strength,
                "start_percent": structural_start,
                "end_percent": structural_end,
            },
        ),
        "24": api_node(
            "VAEEncodeForInpaint",
            {"pixels": link(16, 0), "vae": link(5, 2), "mask": link(33, 0), "grow_mask_by": inpaint_grow_mask_by},
        ),
        "25": api_node(
            "KSampler",
            {
                "model": link(23, 0),
                "positive": link(39, 0),
                "negative": link(39, 1),
                "latent_image": link(24, 0),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        ),
        "26": api_node("VAEDecode", {"samples": link(25, 0), "vae": link(5, 2)}),
        "27": api_node("SaveImage", {"images": link(26, 0), "filename_prefix": f"{intermediate_prefix}/crop_inpaint_raw"}),
        "28": api_node(
            "ImageCompositeMasked",
            {"destination": link(16, 0), "source": link(26, 0), "x": 0, "y": 0, "resize_source": False, "mask": link(33, 0)},
        ),
        "29": api_node("SaveImage", {"images": link(28, 0), "filename_prefix": f"{intermediate_prefix}/crop_composite"}),
        "30": api_node(
            "ImageCompositeMasked",
            {"destination": link(2, 0), "source": link(28, 0), "x": bbox_x, "y": bbox_y, "resize_source": False, "mask": link(33, 0)},
        ),
        "31": api_node("PreviewImage", {"images": link(30, 0)}),
        "32": api_node("SaveImage", {"images": link(30, 0), "filename_prefix": filename_prefix}),
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
    structural_controlnet: str,
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
    structural_strength: float,
    structural_start: float,
    structural_end: float,
    canny_low_threshold: float,
    canny_high_threshold: float,
    face_mask_area: str,
    face_mask_grow: int,
    face_mask_blur: int,
    crop_mask_shrink: int,
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
        [5, 11, 0, 13, 0, "ANALYSIS_MODELS"],
        [6, 2, 0, 13, 1, "IMAGE"],
        [7, 13, 0, 14, 0, "MASK"],
        [8, 14, 0, 15, 0, "IMAGE"],
        [9, 2, 0, 16, 0, "IMAGE"],
        [10, 13, 6, 16, 1, "INT"],
        [11, 13, 7, 16, 2, "INT"],
        [12, 13, 4, 16, 3, "INT"],
        [13, 13, 5, 16, 4, "INT"],
        [14, 16, 0, 17, 0, "IMAGE"],
        [15, 13, 0, 18, 0, "MASK"],
        [16, 13, 4, 18, 1, "INT"],
        [17, 13, 5, 18, 2, "INT"],
        [18, 13, 6, 18, 3, "INT"],
        [19, 13, 7, 18, 4, "INT"],
        [20, 18, 0, 19, 0, "MASK"],
        [21, 19, 0, 20, 0, "IMAGE"],
        [22, 9, 0, 21, 0, "FACEANALYSIS"],
        [23, 16, 0, 21, 1, "IMAGE"],
        [24, 21, 0, 22, 0, "IMAGE"],
        [25, 8, 0, 23, 0, "INSTANTID"],
        [26, 9, 0, 23, 1, "FACEANALYSIS"],
        [27, 10, 0, 23, 2, "CONTROL_NET"],
        [28, 1, 0, 23, 3, "IMAGE"],
        [29, 5, 0, 23, 4, "MODEL"],
        [30, 6, 0, 23, 5, "CONDITIONING"],
        [31, 7, 0, 23, 6, "CONDITIONING"],
        [32, 21, 0, 23, 7, "IMAGE"],
        [33, 33, 0, 23, 8, "MASK"],
        [34, 16, 0, 24, 0, "IMAGE"],
        [35, 5, 2, 24, 1, "VAE"],
        [36, 33, 0, 24, 2, "MASK"],
        [37, 23, 0, 25, 0, "MODEL"],
        [38, 23, 1, 39, 0, "CONDITIONING"],
        [39, 23, 2, 39, 1, "CONDITIONING"],
        [40, 24, 0, 25, 3, "LATENT"],
        [41, 25, 0, 26, 0, "LATENT"],
        [42, 5, 2, 26, 1, "VAE"],
        [43, 26, 0, 27, 0, "IMAGE"],
        [44, 16, 0, 28, 0, "IMAGE"],
        [45, 26, 0, 28, 1, "IMAGE"],
        [46, 33, 0, 28, 5, "MASK"],
        [47, 28, 0, 29, 0, "IMAGE"],
        [48, 2, 0, 30, 0, "IMAGE"],
        [49, 28, 0, 30, 1, "IMAGE"],
        [50, 13, 4, 30, 2, "INT"],
        [51, 13, 5, 30, 3, "INT"],
        [52, 33, 0, 30, 5, "MASK"],
        [53, 30, 0, 31, 0, "IMAGE"],
        [54, 30, 0, 32, 0, "IMAGE"],
        [55, 18, 0, 33, 0, "MASK"],
        [56, 33, 0, 34, 0, "MASK"],
        [57, 34, 0, 35, 0, "IMAGE"],
        [58, 16, 0, 36, 0, "IMAGE"],
        [59, 36, 0, 37, 0, "IMAGE"],
        [60, 38, 0, 39, 2, "CONTROL_NET"],
        [61, 36, 0, 39, 3, "IMAGE"],
        [62, 39, 0, 25, 1, "CONDITIONING"],
        [63, 39, 1, 25, 2, "CONDITIONING"],
    ]
    return {
        "id": "instantid-crop-stitch-experiment-ui",
        "revision": 0,
        "last_node_id": 39,
        "last_link_id": 63,
        "nodes": [
            node(1, "LoadImage", [80, 120], 0, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [1, 28]}], widgets_values=[subject_image, "image"], title="Load Subject Identity"),
            node(2, "LoadImage", [80, 420], 1, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [2, 6, 9, 48]}], widgets_values=[target_image, "image"], title="Load Target Template"),
            node(3, "SaveImage", [450, 120], 2, inputs=[{"name": "images", "type": "IMAGE", "link": 1}], widgets_values=[f"{intermediate_prefix}/subject_identity"], title="Save Subject Input"),
            node(4, "SaveImage", [450, 420], 3, inputs=[{"name": "images", "type": "IMAGE", "link": 2}], widgets_values=[f"{intermediate_prefix}/target_full"], title="Save Full Target Input"),
            node(5, "CheckpointLoaderSimple", [80, 760], 4, outputs=[{"name": "MODEL", "type": "MODEL", "links": [29]}, {"name": "CLIP", "type": "CLIP", "links": [3, 4]}, {"name": "VAE", "type": "VAE", "links": [35, 42]}], widgets_values=[checkpoint], title="Load SDXL Checkpoint"),
            node(6, "CLIPTextEncode", [450, 700], 5, inputs=[{"name": "clip", "type": "CLIP", "link": 3}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [30]}], widgets_values=[positive_prompt], title="Positive Prompt For Crop"),
            node(7, "CLIPTextEncode", [450, 900], 6, inputs=[{"name": "clip", "type": "CLIP", "link": 4}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [31]}], widgets_values=[negative_prompt], title="Negative Prompt"),
            node(8, "InstantIDModelLoader", [820, 80], 7, outputs=[{"name": "INSTANTID", "type": "INSTANTID", "links": [25]}], widgets_values=[instantid_model], title="Load InstantID Adapter"),
            node(9, "InstantIDFaceAnalysis", [820, 240], 8, outputs=[{"name": "FACEANALYSIS", "type": "FACEANALYSIS", "links": [22, 26]}], widgets_values=["CUDA"], title="Load AntelopeV2 Face Analysis"),
            node(10, "ControlNetLoader", [820, 400], 9, outputs=[{"name": "CONTROL_NET", "type": "CONTROL_NET", "links": [27]}], widgets_values=[instantid_controlnet], title="Load InstantID ControlNet"),
            node(11, "FaceAnalysisModels", [820, 620], 10, outputs=[{"name": "ANALYSIS_MODELS", "type": "ANALYSIS_MODELS", "links": [5]}], widgets_values=["insightface", "CUDA"], title="Load Face Mask Analysis"),
            node(38, "ControlNetLoader", [820, 1080], 34, outputs=[{"name": "CONTROL_NET", "type": "CONTROL_NET", "links": [60]}], widgets_values=[structural_controlnet], title="Load SDXL Canny ControlNet"),
            node(13, "FaceSegmentation", [1200, 520], 11, inputs=[{"name": "analysis_models", "type": "ANALYSIS_MODELS", "link": 5}, {"name": "image", "type": "IMAGE", "link": 6}], outputs=[{"name": "mask", "type": "MASK", "links": [7, 15]}, {"name": "image", "type": "IMAGE", "links": []}, {"name": "seg_mask", "type": "MASK", "links": []}, {"name": "seg_image", "type": "IMAGE", "links": []}, {"name": "x", "type": "INT", "links": [12, 16, 50]}, {"name": "y", "type": "INT", "links": [13, 17, 51]}, {"name": "width", "type": "INT", "links": [10, 18]}, {"name": "height", "type": "INT", "links": [11, 19]}], widgets_values=[face_mask_area, face_mask_grow, True, face_mask_blur], title="Find Generous Target Head Region"),
            node(14, "MaskToImage", [1550, 360], 12, inputs=[{"name": "mask", "type": "MASK", "link": 7}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [8]}], title="Convert Full Mask To Image"),
            node(15, "SaveImage", [1900, 360], 13, inputs=[{"name": "images", "type": "IMAGE", "link": 8}], widgets_values=[f"{intermediate_prefix}/target_full_mask"], title="Save Full Target Mask"),
            node(16, "ImageCrop", [1550, 560], 14, inputs=[{"name": "image", "type": "IMAGE", "link": 9}, {"name": "width", "type": "INT", "link": 10}, {"name": "height", "type": "INT", "link": 11}, {"name": "x", "type": "INT", "link": 12}, {"name": "y", "type": "INT", "link": 13}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [14, 23, 34, 44, 58]}], widgets_values=[], title="Crop Target Head Region"),
            node(17, "SaveImage", [1900, 560], 15, inputs=[{"name": "images", "type": "IMAGE", "link": 14}], widgets_values=[f"{intermediate_prefix}/target_crop"], title="Save Target Crop"),
            node(18, "CropMask", [1550, 760], 16, inputs=[{"name": "mask", "type": "MASK", "link": 15}, {"name": "x", "type": "INT", "link": 16}, {"name": "y", "type": "INT", "link": 17}, {"name": "width", "type": "INT", "link": 18}, {"name": "height", "type": "INT", "link": 19}], outputs=[{"name": "MASK", "type": "MASK", "links": [20, 55]}], title="Crop Target Region Mask"),
            node(19, "MaskToImage", [1900, 760], 17, inputs=[{"name": "mask", "type": "MASK", "link": 20}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [21]}], title="Convert Crop Mask To Image"),
            node(20, "SaveImage", [2250, 760], 18, inputs=[{"name": "images", "type": "IMAGE", "link": 21}], widgets_values=[f"{intermediate_prefix}/target_crop_region_mask"], title="Save Crop Region Mask"),
            node(21, "FaceKeypointsPreprocessor", [1900, 980], 19, inputs=[{"name": "faceanalysis", "type": "FACEANALYSIS", "link": 22}, {"name": "image", "type": "IMAGE", "link": 23}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [24, 32]}], title="Extract Crop Keypoints"),
            node(22, "SaveImage", [2250, 980], 20, inputs=[{"name": "images", "type": "IMAGE", "link": 24}], widgets_values=[f"{intermediate_prefix}/target_crop_keypoints"], title="Save Crop Keypoints"),
            node(23, "ApplyInstantIDAdvanced", [2350, 460], 21, inputs=[{"name": "instantid", "type": "INSTANTID", "link": 25}, {"name": "insightface", "type": "FACEANALYSIS", "link": 26}, {"name": "control_net", "type": "CONTROL_NET", "link": 27}, {"name": "image", "type": "IMAGE", "link": 28}, {"name": "model", "type": "MODEL", "link": 29}, {"name": "positive", "type": "CONDITIONING", "link": 30}, {"name": "negative", "type": "CONDITIONING", "link": 31}, {"name": "image_kps", "type": "IMAGE", "link": 32}, {"name": "mask", "type": "MASK", "link": 33}], outputs=[{"name": "MODEL", "type": "MODEL", "links": [37]}, {"name": "positive", "type": "CONDITIONING", "links": [38]}, {"name": "negative", "type": "CONDITIONING", "links": [39]}], widgets_values=[instantid_weight, pose_strength, instantid_start, instantid_end, instantid_noise, "average"], title="Apply InstantID To Crop"),
            node(24, "VAEEncodeForInpaint", [2350, 700], 22, inputs=[{"name": "pixels", "type": "IMAGE", "link": 34}, {"name": "vae", "type": "VAE", "link": 35}, {"name": "mask", "type": "MASK", "link": 36}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [40]}], widgets_values=[inpaint_grow_mask_by], title="Encode Crop With Local Mask"),
            node(25, "KSampler", [2750, 560], 23, inputs=[{"name": "model", "type": "MODEL", "link": 37}, {"name": "positive", "type": "CONDITIONING", "link": 62}, {"name": "negative", "type": "CONDITIONING", "link": 63}, {"name": "latent_image", "type": "LATENT", "link": 40}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [41]}], widgets_values=[seed, "fixed", steps, cfg, sampler_name, scheduler, denoise], title="Inpaint Crop"),
            node(26, "VAEDecode", [3150, 560], 24, inputs=[{"name": "samples", "type": "LATENT", "link": 41}, {"name": "vae", "type": "VAE", "link": 42}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [43, 45]}], title="Decode Crop Inpaint"),
            node(27, "SaveImage", [3500, 500], 25, inputs=[{"name": "images", "type": "IMAGE", "link": 43}], widgets_values=[f"{intermediate_prefix}/crop_inpaint_raw"], title="Save Raw Crop Inpaint"),
            node(28, "ImageCompositeMasked", [3500, 700], 26, inputs=[{"name": "destination", "type": "IMAGE", "link": 44}, {"name": "source", "type": "IMAGE", "link": 45}, {"name": "mask", "type": "MASK", "link": 46}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [47, 49]}], widgets_values=[0, 0, False], title="Composite Inpaint Back Into Crop"),
            node(29, "SaveImage", [3900, 700], 27, inputs=[{"name": "images", "type": "IMAGE", "link": 47}], widgets_values=[f"{intermediate_prefix}/crop_composite"], title="Save Crop Composite"),
            node(30, "ImageCompositeMasked", [3900, 420], 28, inputs=[{"name": "destination", "type": "IMAGE", "link": 48}, {"name": "source", "type": "IMAGE", "link": 49}, {"name": "x", "type": "INT", "link": 50}, {"name": "y", "type": "INT", "link": 51}, {"name": "mask", "type": "MASK", "link": 52}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [53, 54]}], widgets_values=[False], title="Stitch Crop Back Into Full Target"),
            node(31, "PreviewImage", [4300, 320], 29, inputs=[{"name": "images", "type": "IMAGE", "link": 53}], title="Preview Final Stitched Composite"),
            node(32, "SaveImage", [4300, 500], 30, inputs=[{"name": "images", "type": "IMAGE", "link": 54}], widgets_values=[filename_prefix], title="Save Final Stitched Composite"),
            node(33, "GrowMask", [1900, 840], 31, inputs=[{"name": "mask", "type": "MASK", "link": 55}], outputs=[{"name": "MASK", "type": "MASK", "links": [33, 36, 46, 52, 56]}], widgets_values=[crop_mask_shrink, True], title="Shrink Crop Mask For Local Edit"),
            node(34, "MaskToImage", [2250, 840], 32, inputs=[{"name": "mask", "type": "MASK", "link": 56}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [57]}], title="Convert Edit Mask To Image"),
            node(35, "SaveImage", [2600, 840], 33, inputs=[{"name": "images", "type": "IMAGE", "link": 57}], widgets_values=[f"{intermediate_prefix}/target_crop_edit_mask"], title="Save Crop Edit Mask"),
            node(36, "Canny", [1900, 1180], 35, inputs=[{"name": "image", "type": "IMAGE", "link": 58}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [59, 61]}], widgets_values=[canny_low_threshold, canny_high_threshold], title="Extract Crop Edges"),
            node(37, "SaveImage", [2250, 1180], 36, inputs=[{"name": "images", "type": "IMAGE", "link": 59}], widgets_values=[f"{intermediate_prefix}/target_crop_canny"], title="Save Crop Edges"),
            node(39, "ControlNetApplyAdvanced", [2350, 1080], 37, inputs=[{"name": "positive", "type": "CONDITIONING", "link": 38}, {"name": "negative", "type": "CONDITIONING", "link": 39}, {"name": "control_net", "type": "CONTROL_NET", "link": 60}, {"name": "image", "type": "IMAGE", "link": 61}], outputs=[{"name": "positive", "type": "CONDITIONING", "links": [62]}, {"name": "negative", "type": "CONDITIONING", "links": [63]}], widgets_values=[structural_strength, structural_start, structural_end], title="Apply Crop Edge Control"),
        ],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="workflows/instantid_crop_stitch_experiment_api.json")
    parser.add_argument("--ui-output", default="workflows/instantid_crop_stitch_experiment_ui.json")
    parser.add_argument("--subject-image", default="subject_5 year curly.webp")
    parser.add_argument("--target-image", default="superman.png")
    parser.add_argument("--checkpoint", default="sd_xl_base_1.0_inpainting_0.1.safetensors")
    parser.add_argument("--instantid-model", default="ip-adapter.bin")
    parser.add_argument("--instantid-controlnet", default="instantid_controlnet.safetensors")
    parser.add_argument("--structural-controlnet", default="controlnet-canny-sdxl-1.0-small.safetensors")
    parser.add_argument(
        "--positive-prompt",
        default=(
            "local head and face crop of the same target character, replace the face with a young child with curly hair, "
            "preserve head angle, crop framing, surrounding hair silhouette, neck transition, preserve the target illustration style"
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
    parser.add_argument("--pose-strength", type=float, default=0.35)
    parser.add_argument("--instantid-start", type=float, default=0.0)
    parser.add_argument("--instantid-end", type=float, default=0.82)
    parser.add_argument("--instantid-noise", type=float, default=0.25)
    parser.add_argument("--structural-strength", type=float, default=0.45)
    parser.add_argument("--structural-start", type=float, default=0.0)
    parser.add_argument("--structural-end", type=float, default=0.7)
    parser.add_argument("--canny-low-threshold", type=float, default=0.2)
    parser.add_argument("--canny-high-threshold", type=float, default=0.6)
    parser.add_argument("--face-mask-area", default="face+forehead (if available)")
    parser.add_argument("--face-mask-grow", type=int, default=72)
    parser.add_argument("--face-mask-blur", type=int, default=31)
    parser.add_argument("--crop-mask-shrink", type=int, default=-56)
    parser.add_argument("--inpaint-grow-mask-by", type=int, default=12)
    parser.add_argument("--denoise", type=float, default=0.65)
    parser.add_argument("--filename-prefix", default="faceswap/instantid_crop_stitch/final")
    parser.add_argument("--intermediate-prefix", default="faceswap/instantid_crop_stitch/intermediate")
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
