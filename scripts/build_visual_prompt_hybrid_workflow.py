#!/usr/bin/env python3
"""Generate the subject-agnostic visual prompt hybrid workflow pair."""

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
        "size": [340, 120],
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
    detail_checkpoint: str,
    reactor_enabled: bool,
    reactor_face_restore_model: str,
    reactor_face_restore_visibility: float,
    reactor_codeformer_weight: float,
    positive_prompt: str,
    negative_prompt: str,
    semantic_mask_text: str,
    clipseg_blur: float,
    clipseg_threshold: float,
    clipseg_dilation_factor: int,
    primary_inpaint_grow_mask_by: int,
    pulid_weight: float,
    pulid_projection: str,
    pulid_fidelity: int,
    ipadapter_preset: str,
    ipadapter_weight: float,
    ipadapter_weight_type: str,
    ipadapter_end_at: float,
    ipadapter_embeds_scaling: str,
    primary_seed: int,
    primary_steps: int,
    primary_cfg: float,
    primary_denoise: float,
    secondary_seed: int,
    secondary_steps: int,
    secondary_cfg: float,
    secondary_denoise: float,
    inner_face_area: str,
    inner_face_grow: int,
    inner_face_blur: int,
    secondary_inpaint_grow_mask_by: int,
    skin_mask_text: str,
    skin_mask_blur: float,
    skin_mask_threshold: float,
    skin_mask_dilation_factor: int,
    hires_upscale_method: str,
    hires_scale_by: float,
    hires_seed: int,
    hires_steps: int,
    hires_cfg: float,
    hires_denoise: float,
    sampler_name: str,
    scheduler: str,
    filename_prefix: str,
    hires_filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    return {
        "1": api_node("LoadImage", {"image": subject_image}),
        "2": api_node("LoadImage", {"image": target_image}),
        "3": api_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint}),
        "4": api_node("CLIPTextEncode", {"clip": link(3, 1), "text": positive_prompt}),
        "5": api_node("CLIPTextEncode", {"clip": link(3, 1), "text": negative_prompt}),
        "6": api_node(
            "CLIPSeg",
            {
                "image": link(2, 0),
                "text": semantic_mask_text,
                "blur": clipseg_blur,
                "threshold": clipseg_threshold,
                "dilation_factor": clipseg_dilation_factor,
            },
        ),
        "7": api_node("ImageToMask", {"image": link(6, 2), "channel": "red"}),
        "8": api_node("MaskToImage", {"mask": link(7, 0)}),
        "9": api_node(
            "SaveImage",
            {"images": link(8, 0), "filename_prefix": f"{intermediate_prefix}/target_head_mask_semantic"},
        ),
        "10": api_node(
            "VAEEncodeForInpaint",
            {
                "pixels": link(2, 0),
                "vae": link(3, 2),
                "mask": link(7, 0),
                "grow_mask_by": primary_inpaint_grow_mask_by,
            },
        ),
        "11": api_node("PulidModelLoader", {"pulid_file": "ip-adapter_pulid_sdxl_fp16.safetensors"}),
        "12": api_node("PulidEvaClipLoader", {}),
        "13": api_node("PulidInsightFaceLoader", {"provider": "CUDA"}),
        "14": api_node(
            "ApplyPulidAdvanced",
            {
                "model": link(3, 0),
                "pulid": link(11, 0),
                "eva_clip": link(12, 0),
                "face_analysis": link(13, 0),
                "image": link(1, 0),
                "weight": pulid_weight,
                "projection": pulid_projection,
                "fidelity": pulid_fidelity,
                "noise": 0.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "attn_mask": link(7, 0),
            },
        ),
        "15": api_node("IPAdapterUnifiedLoader", {"model": link(14, 0), "preset": ipadapter_preset}),
        "16": api_node(
            "IPAdapterAdvanced",
            {
                "model": link(15, 0),
                "ipadapter": link(15, 1),
                "image": link(2, 0),
                "weight": ipadapter_weight,
                "weight_type": ipadapter_weight_type,
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": ipadapter_end_at,
                "embeds_scaling": ipadapter_embeds_scaling,
                "attn_mask": link(7, 0),
            },
        ),
        "17": api_node(
            "KSampler",
            {
                "model": link(16, 0),
                "positive": link(4, 0),
                "negative": link(5, 0),
                "latent_image": link(10, 0),
                "seed": primary_seed,
                "steps": primary_steps,
                "cfg": primary_cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": primary_denoise,
            },
        ),
        "18": api_node("VAEDecode", {"samples": link(17, 0), "vae": link(3, 2)}),
        "19": api_node(
            "SaveImage",
            {"images": link(18, 0), "filename_prefix": f"{intermediate_prefix}/generated_head"},
        ),
        "20": api_node(
            "ReActorFaceSwap",
            {
                "enabled": reactor_enabled,
                "input_image": link(18, 0),
                "source_image": link(1, 0),
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": reactor_face_restore_model,
                "face_restore_visibility": reactor_face_restore_visibility,
                "codeformer_weight": reactor_codeformer_weight,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
            },
        ),
        "21": api_node(
            "SaveImage",
            {"images": link(20, 0), "filename_prefix": f"{intermediate_prefix}/reactor_bake"},
        ),
        "22": api_node("FaceAnalysisModels", {"library": "insightface", "provider": "CUDA"}),
        "23": api_node(
            "FaceSegmentation",
            {
                "analysis_models": link(22, 0),
                "image": link(20, 0),
                "area": inner_face_area,
                "grow": inner_face_grow,
                "grow_tapered": True,
                "blur": inner_face_blur,
            },
        ),
        "24": api_node("MaskToImage", {"mask": link(23, 0)}),
        "25": api_node(
            "SaveImage",
            {"images": link(24, 0), "filename_prefix": f"{intermediate_prefix}/inner_face_mask"},
        ),
        "43": api_node("CheckpointLoaderSimple", {"ckpt_name": detail_checkpoint}),
        "44": api_node("CLIPTextEncode", {"clip": link(43, 1), "text": positive_prompt}),
        "45": api_node("CLIPTextEncode", {"clip": link(43, 1), "text": negative_prompt}),
        "26": api_node("VAEEncode", {"pixels": link(20, 0), "vae": link(43, 2)}),
        "27": api_node(
            "KSampler",
            {
                "model": link(43, 0),
                "positive": link(44, 0),
                "negative": link(45, 0),
                "latent_image": link(26, 0),
                "seed": secondary_seed,
                "steps": secondary_steps,
                "cfg": secondary_cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": secondary_denoise,
            },
        ),
        "28": api_node("VAEDecode", {"samples": link(27, 0), "vae": link(43, 2)}),
        "46": api_node(
            "SaveImage",
            {"images": link(28, 0), "filename_prefix": f"{intermediate_prefix}/inner_face_sdedit"},
        ),
        "29": api_node(
            "ImageCompositeMasked",
            {
                "destination": link(20, 0),
                "source": link(28, 0),
                "x": 0,
                "y": 0,
                "resize_source": False,
                "mask": link(23, 0),
            },
        ),
        "30": api_node(
            "SaveImage",
            {"images": link(29, 0), "filename_prefix": f"{intermediate_prefix}/pre_skin_harmonize"},
        ),
        "31": api_node(
            "CLIPSeg",
            {
                "image": link(29, 0),
                "text": skin_mask_text,
                "blur": skin_mask_blur,
                "threshold": skin_mask_threshold,
                "dilation_factor": skin_mask_dilation_factor,
            },
        ),
        "32": api_node("ImageToMask", {"image": link(31, 2), "channel": "red"}),
        "33": api_node("MaskToImage", {"mask": link(32, 0)}),
        "34": api_node(
            "SaveImage",
            {"images": link(33, 0), "filename_prefix": f"{intermediate_prefix}/target_skin_mask"},
        ),
        "35": api_node("SaveImage", {"images": link(29, 0), "filename_prefix": filename_prefix}),
        "36": api_node(
            "ImageScaleBy",
            {
                "image": link(29, 0),
                "upscale_method": hires_upscale_method,
                "scale_by": hires_scale_by,
            },
        ),
        "37": api_node(
            "SaveImage",
            {"images": link(36, 0), "filename_prefix": f"{intermediate_prefix}/hires_upscaled_base"},
        ),
        "38": api_node("VAEEncode", {"pixels": link(36, 0), "vae": link(43, 2)}),
        "39": api_node(
            "KSampler",
            {
                "model": link(43, 0),
                "positive": link(44, 0),
                "negative": link(45, 0),
                "latent_image": link(38, 0),
                "seed": hires_seed,
                "steps": hires_steps,
                "cfg": hires_cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": hires_denoise,
            },
        ),
        "40": api_node("VAEDecode", {"samples": link(39, 0), "vae": link(43, 2)}),
        "41": api_node(
            "ReActorFaceSwap",
            {
                "enabled": reactor_enabled,
                "input_image": link(40, 0),
                "source_image": link(1, 0),
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": reactor_face_restore_model,
                "face_restore_visibility": reactor_face_restore_visibility,
                "codeformer_weight": reactor_codeformer_weight,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
            },
        ),
        "42": api_node("SaveImage", {"images": link(41, 0), "filename_prefix": hires_filename_prefix}),
    }


def build_ui_workflow(
    *,
    subject_image: str,
    target_image: str,
    checkpoint: str,
    detail_checkpoint: str,
    reactor_enabled: bool,
    reactor_face_restore_model: str,
    reactor_face_restore_visibility: float,
    reactor_codeformer_weight: float,
    positive_prompt: str,
    negative_prompt: str,
    semantic_mask_text: str,
    clipseg_blur: float,
    clipseg_threshold: float,
    clipseg_dilation_factor: int,
    primary_inpaint_grow_mask_by: int,
    pulid_weight: float,
    pulid_projection: str,
    pulid_fidelity: int,
    ipadapter_preset: str,
    ipadapter_weight: float,
    ipadapter_weight_type: str,
    ipadapter_end_at: float,
    ipadapter_embeds_scaling: str,
    primary_seed: int,
    primary_steps: int,
    primary_cfg: float,
    primary_denoise: float,
    secondary_seed: int,
    secondary_steps: int,
    secondary_cfg: float,
    secondary_denoise: float,
    inner_face_area: str,
    inner_face_grow: int,
    inner_face_blur: int,
    secondary_inpaint_grow_mask_by: int,
    skin_mask_text: str,
    skin_mask_blur: float,
    skin_mask_threshold: float,
    skin_mask_dilation_factor: int,
    hires_upscale_method: str,
    hires_scale_by: float,
    hires_seed: int,
    hires_steps: int,
    hires_cfg: float,
    hires_denoise: float,
    sampler_name: str,
    scheduler: str,
    filename_prefix: str,
    hires_filename_prefix: str,
    intermediate_prefix: str,
) -> dict[str, Any]:
    links = [
        [1, 3, 1, 4, 0, "CLIP"],
        [2, 3, 1, 5, 0, "CLIP"],
        [3, 2, 0, 6, 0, "IMAGE"],
        [4, 6, 2, 7, 0, "IMAGE"],
        [5, 7, 0, 8, 0, "MASK"],
        [6, 8, 0, 9, 0, "IMAGE"],
        [7, 2, 0, 10, 0, "IMAGE"],
        [8, 3, 2, 10, 1, "VAE"],
        [9, 7, 0, 10, 2, "MASK"],
        [10, 11, 0, 14, 1, "PULID"],
        [11, 12, 0, 14, 2, "EVA_CLIP"],
        [12, 13, 0, 14, 3, "FACEANALYSIS"],
        [13, 3, 0, 14, 0, "MODEL"],
        [14, 1, 0, 14, 4, "IMAGE"],
        [15, 7, 0, 14, 11, "MASK"],
        [16, 14, 0, 15, 0, "MODEL"],
        [17, 15, 0, 16, 0, "MODEL"],
        [18, 15, 1, 16, 1, "IPADAPTER"],
        [19, 2, 0, 16, 2, "IMAGE"],
        [20, 7, 0, 16, 5, "MASK"],
        [21, 16, 0, 17, 0, "MODEL"],
        [22, 4, 0, 17, 1, "CONDITIONING"],
        [23, 5, 0, 17, 2, "CONDITIONING"],
        [24, 10, 0, 17, 3, "LATENT"],
        [25, 17, 0, 18, 0, "LATENT"],
        [26, 3, 2, 18, 1, "VAE"],
        [27, 18, 0, 19, 0, "IMAGE"],
        [28, 18, 0, 20, 0, "IMAGE"],
        [29, 1, 0, 20, 1, "IMAGE"],
        [30, 20, 0, 21, 0, "IMAGE"],
        [31, 22, 0, 23, 0, "ANALYSIS_MODELS"],
        [32, 20, 0, 23, 1, "IMAGE"],
        [33, 23, 0, 24, 0, "MASK"],
        [34, 24, 0, 25, 0, "IMAGE"],
        [35, 20, 0, 26, 0, "IMAGE"],
        [36, 43, 2, 26, 1, "VAE"],
        [37, 43, 0, 27, 0, "MODEL"],
        [38, 44, 0, 27, 1, "CONDITIONING"],
        [39, 45, 0, 27, 2, "CONDITIONING"],
        [40, 26, 0, 27, 3, "LATENT"],
        [41, 27, 0, 28, 0, "LATENT"],
        [42, 43, 2, 28, 1, "VAE"],
        [43, 20, 0, 29, 0, "IMAGE"],
        [44, 28, 0, 29, 1, "IMAGE"],
        [45, 23, 0, 29, 5, "MASK"],
        [46, 29, 0, 30, 0, "IMAGE"],
        [47, 29, 0, 31, 0, "IMAGE"],
        [48, 31, 2, 32, 0, "IMAGE"],
        [49, 32, 0, 33, 0, "MASK"],
        [50, 33, 0, 34, 0, "IMAGE"],
        [51, 29, 0, 35, 0, "IMAGE"],
        [52, 29, 0, 36, 0, "IMAGE"],
        [53, 36, 0, 37, 0, "IMAGE"],
        [54, 36, 0, 38, 0, "IMAGE"],
        [55, 43, 2, 38, 1, "VAE"],
        [56, 43, 0, 39, 0, "MODEL"],
        [57, 44, 0, 39, 1, "CONDITIONING"],
        [58, 45, 0, 39, 2, "CONDITIONING"],
        [59, 38, 0, 39, 3, "LATENT"],
        [60, 39, 0, 40, 0, "LATENT"],
        [61, 43, 2, 40, 1, "VAE"],
        [62, 40, 0, 41, 0, "IMAGE"],
        [63, 1, 0, 41, 1, "IMAGE"],
        [64, 41, 0, 42, 0, "IMAGE"],
        [65, 28, 0, 46, 0, "IMAGE"],
        [66, 43, 1, 44, 0, "CLIP"],
        [67, 43, 1, 45, 0, "CLIP"],
    ]
    return {
        "id": "visual-prompt-hybrid-experiment-ui",
        "revision": 0,
        "last_node_id": 46,
        "last_link_id": 67,
        "nodes": [
            node(1, "LoadImage", [60, 80], 0, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [14, 29, 63]}], widgets_values=[subject_image, "image"], title="Load Source Identity"),
            node(2, "LoadImage", [60, 360], 1, outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [3, 7, 19]}], widgets_values=[target_image, "image"], title="Load Target Template"),
            node(3, "CheckpointLoaderSimple", [60, 700], 2, outputs=[{"name": "MODEL", "type": "MODEL", "links": [13]}, {"name": "CLIP", "type": "CLIP", "links": [1, 2]}, {"name": "VAE", "type": "VAE", "links": [8, 26]}], widgets_values=[checkpoint], title="Load SDXL Inpaint Checkpoint"),
            node(4, "CLIPTextEncode", [420, 650], 3, inputs=[{"name": "clip", "type": "CLIP", "link": 1}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [22]}], widgets_values=[positive_prompt], title="Primary Style Prompt"),
            node(5, "CLIPTextEncode", [420, 900], 4, inputs=[{"name": "clip", "type": "CLIP", "link": 2}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [23]}], widgets_values=[negative_prompt], title="Primary Negative Prompt"),
            node(6, "CLIPSeg", [420, 280], 5, inputs=[{"name": "image", "type": "IMAGE", "link": 3}], outputs=[{"name": "Mask", "type": "MASK", "links": None}, {"name": "Heatmap Mask", "type": "IMAGE", "links": None}, {"name": "BW Mask", "type": "IMAGE", "links": [4]}], widgets_values=[semantic_mask_text, clipseg_blur, clipseg_threshold, clipseg_dilation_factor], title="Semantic Head Mask"),
            node(7, "ImageToMask", [820, 220], 6, inputs=[{"name": "image", "type": "IMAGE", "link": 4}], outputs=[{"name": "MASK", "type": "MASK", "links": [5, 9, 15, 20]}], widgets_values=["red"], title="Normalize CLIPSeg Mask"),
            node(8, "MaskToImage", [1180, 220], 7, inputs=[{"name": "mask", "type": "MASK", "link": 5}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [6]}], title="Convert Semantic Mask"),
            node(9, "SaveImage", [1540, 220], 8, inputs=[{"name": "images", "type": "IMAGE", "link": 6}], widgets_values=[f"{intermediate_prefix}/target_head_mask_semantic"], title="Save Semantic Head Mask"),
            node(10, "VAEEncodeForInpaint", [1180, 430], 9, inputs=[{"name": "pixels", "type": "IMAGE", "link": 7}, {"name": "vae", "type": "VAE", "link": 8}, {"name": "mask", "type": "MASK", "link": 9}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [24]}], widgets_values=[primary_inpaint_grow_mask_by], title="Encode Target For Primary Inpaint"),
            node(11, "PulidModelLoader", [1180, 780], 10, outputs=[{"name": "PULID", "type": "PULID", "links": [10]}], widgets_values=["ip-adapter_pulid_sdxl_fp16.safetensors"], title="Load PuLID Model"),
            node(12, "PulidEvaClipLoader", [1180, 930], 11, outputs=[{"name": "EVA_CLIP", "type": "EVA_CLIP", "links": [11]}], title="Load EVA CLIP"),
            node(13, "PulidInsightFaceLoader", [1180, 1080], 12, outputs=[{"name": "FACEANALYSIS", "type": "FACEANALYSIS", "links": [12]}], widgets_values=["CUDA"], title="Load PuLID Face Analysis"),
            node(14, "ApplyPulidAdvanced", [1580, 900], 13, inputs=[{"name": "model", "type": "MODEL", "link": 13}, {"name": "pulid", "type": "PULID", "link": 10}, {"name": "eva_clip", "type": "EVA_CLIP", "link": 11}, {"name": "face_analysis", "type": "FACEANALYSIS", "link": 12}, {"name": "image", "type": "IMAGE", "link": 14}, {"name": "attn_mask", "type": "MASK", "link": 15}], outputs=[{"name": "MODEL", "type": "MODEL", "links": [16]}], widgets_values=[pulid_weight, pulid_projection, pulid_fidelity, 0.0, 0.0, 1.0], title="Apply PuLID Identity"),
            node(15, "IPAdapterUnifiedLoader", [1980, 900], 14, inputs=[{"name": "model", "type": "MODEL", "link": 16}, {"name": "ipadapter", "type": "IPADAPTER", "link": None}], outputs=[{"name": "model", "type": "MODEL", "links": [17]}, {"name": "ipadapter", "type": "IPADAPTER", "links": [18]}], widgets_values=[ipadapter_preset], title="Load IP-Adapter"),
            node(16, "IPAdapterAdvanced", [2380, 900], 15, inputs=[{"name": "model", "type": "MODEL", "link": 17}, {"name": "ipadapter", "type": "IPADAPTER", "link": 18}, {"name": "image", "type": "IMAGE", "link": 19}, {"name": "attn_mask", "type": "MASK", "link": 20}], outputs=[{"name": "MODEL", "type": "MODEL", "links": [21]}], widgets_values=[ipadapter_weight, ipadapter_weight_type, "concat", 0.0, ipadapter_end_at, ipadapter_embeds_scaling], title="Apply Target Style IP-Adapter Guidance"),
            node(17, "KSampler", [2780, 780], 16, inputs=[{"name": "model", "type": "MODEL", "link": 21}, {"name": "positive", "type": "CONDITIONING", "link": 22}, {"name": "negative", "type": "CONDITIONING", "link": 23}, {"name": "latent_image", "type": "LATENT", "link": 24}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [25]}], widgets_values=[primary_seed, "fixed", primary_steps, primary_cfg, sampler_name, scheduler, primary_denoise], title="Primary High-Denoise Generation"),
            node(18, "VAEDecode", [3180, 780], 17, inputs=[{"name": "samples", "type": "LATENT", "link": 25}, {"name": "vae", "type": "VAE", "link": 26}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [27, 28]}], title="Decode Generated Head"),
            node(19, "SaveImage", [3540, 700], 18, inputs=[{"name": "images", "type": "IMAGE", "link": 27}], widgets_values=[f"{intermediate_prefix}/generated_head"], title="Save Generated Head"),
            node(20, "ReActorFaceSwap", [3540, 920], 19, inputs=[{"name": "input_image", "type": "IMAGE", "link": 28}, {"name": "source_image", "type": "IMAGE", "link": 29}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [30, 32, 35]}], widgets_values=[reactor_enabled, "inswapper_128.onnx", "retinaface_resnet50", reactor_face_restore_model, reactor_face_restore_visibility, reactor_codeformer_weight, "no", "no", "0", "0", 1], title="ReActor Restore Weld"),
            node(21, "SaveImage", [3940, 920], 20, inputs=[{"name": "images", "type": "IMAGE", "link": 30}], widgets_values=[f"{intermediate_prefix}/reactor_bake"], title="Save ReActor Bake"),
            node(22, "FaceAnalysisModels", [3940, 1150], 21, outputs=[{"name": "ANALYSIS_MODELS", "type": "ANALYSIS_MODELS", "links": [31]}], widgets_values=["insightface", "CUDA"], title="Load Face Analysis"),
            node(23, "FaceSegmentation", [4320, 1120], 22, inputs=[{"name": "analysis_models", "type": "ANALYSIS_MODELS", "link": 31}, {"name": "image", "type": "IMAGE", "link": 32}], outputs=[{"name": "mask", "type": "MASK", "links": [33, 37]}, {"name": "image", "type": "IMAGE", "links": None}, {"name": "seg_mask", "type": "MASK", "links": None}, {"name": "seg_image", "type": "IMAGE", "links": None}, {"name": "x", "type": "INT", "links": None}, {"name": "y", "type": "INT", "links": None}, {"name": "width", "type": "INT", "links": None}, {"name": "height", "type": "INT", "links": None}], widgets_values=[inner_face_area, inner_face_grow, True, inner_face_blur], title="Inner Face Blend Mask"),
            node(24, "MaskToImage", [4680, 1060], 23, inputs=[{"name": "mask", "type": "MASK", "link": 33}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [34]}], title="Convert Inner Face Mask"),
            node(25, "SaveImage", [5040, 1060], 24, inputs=[{"name": "images", "type": "IMAGE", "link": 34}], widgets_values=[f"{intermediate_prefix}/inner_face_mask"], title="Save Inner Face Mask"),
            node(26, "VAEEncode", [4680, 1280], 25, inputs=[{"name": "pixels", "type": "IMAGE", "link": 35}, {"name": "vae", "type": "VAE", "link": 36}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [40]}], title="Encode ReActor Output For Inner-Face SDEdit"),
            node(27, "KSampler", [5080, 1280], 26, inputs=[{"name": "model", "type": "MODEL", "link": 37}, {"name": "positive", "type": "CONDITIONING", "link": 38}, {"name": "negative", "type": "CONDITIONING", "link": 39}, {"name": "latent_image", "type": "LATENT", "link": 40}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [41]}], widgets_values=[secondary_seed, "fixed", secondary_steps, secondary_cfg, sampler_name, scheduler, secondary_denoise], title="Precision Inner-Face SDEdit"),
            node(28, "VAEDecode", [5480, 1280], 27, inputs=[{"name": "samples", "type": "LATENT", "link": 41}, {"name": "vae", "type": "VAE", "link": 42}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [44, 65]}], title="Decode Inner-Face SDEdit"),
            node(29, "ImageCompositeMasked", [5880, 1280], 28, inputs=[{"name": "destination", "type": "IMAGE", "link": 43}, {"name": "source", "type": "IMAGE", "link": 44}, {"name": "mask", "type": "MASK", "link": 45}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [46, 47, 52]}], widgets_values=[0, 0, False], title="Blend Inner-Face SDEdit"),
            node(30, "SaveImage", [6280, 1140], 29, inputs=[{"name": "images", "type": "IMAGE", "link": 46}], widgets_values=[f"{intermediate_prefix}/pre_skin_harmonize"], title="Save Pre-Skin Harmonize"),
            node(31, "CLIPSeg", [6280, 1460], 30, inputs=[{"name": "image", "type": "IMAGE", "link": 47}], outputs=[{"name": "Mask", "type": "MASK", "links": None}, {"name": "Heatmap Mask", "type": "IMAGE", "links": None}, {"name": "BW Mask", "type": "IMAGE", "links": [48]}], widgets_values=[skin_mask_text, skin_mask_blur, skin_mask_threshold, skin_mask_dilation_factor], title="Semantic Exposed Skin Mask"),
            node(32, "ImageToMask", [6680, 1460], 31, inputs=[{"name": "image", "type": "IMAGE", "link": 48}], outputs=[{"name": "MASK", "type": "MASK", "links": [49]}], widgets_values=["red"], title="Normalize Skin Mask"),
            node(33, "MaskToImage", [7040, 1400], 32, inputs=[{"name": "mask", "type": "MASK", "link": 49}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [50]}], title="Convert Skin Mask"),
            node(34, "SaveImage", [7400, 1400], 33, inputs=[{"name": "images", "type": "IMAGE", "link": 50}], widgets_values=[f"{intermediate_prefix}/target_skin_mask"], title="Save Exposed Skin Mask"),
            node(35, "SaveImage", [7760, 1520], 34, inputs=[{"name": "images", "type": "IMAGE", "link": 51}], widgets_values=[filename_prefix], title="Save Final Result"),
            node(36, "ImageScaleBy", [6280, 1780], 35, inputs=[{"name": "image", "type": "IMAGE", "link": 52}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [53, 54]}], widgets_values=[hires_upscale_method, hires_scale_by], title="Upscale For Hi-Res Pass"),
            node(37, "SaveImage", [6680, 1720], 36, inputs=[{"name": "images", "type": "IMAGE", "link": 53}], widgets_values=[f"{intermediate_prefix}/hires_upscaled_base"], title="Save Hi-Res Upscaled Base"),
            node(38, "VAEEncode", [6680, 1920], 37, inputs=[{"name": "pixels", "type": "IMAGE", "link": 54}, {"name": "vae", "type": "VAE", "link": 55}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [59]}], title="Encode Hi-Res Base"),
            node(39, "KSampler", [7080, 1920], 38, inputs=[{"name": "model", "type": "MODEL", "link": 56}, {"name": "positive", "type": "CONDITIONING", "link": 57}, {"name": "negative", "type": "CONDITIONING", "link": 58}, {"name": "latent_image", "type": "LATENT", "link": 59}], outputs=[{"name": "LATENT", "type": "LATENT", "links": [60]}], widgets_values=[hires_seed, "fixed", hires_steps, hires_cfg, sampler_name, scheduler, hires_denoise], title="Hi-Res Precision Refine"),
            node(40, "VAEDecode", [7480, 1920], 39, inputs=[{"name": "samples", "type": "LATENT", "link": 60}, {"name": "vae", "type": "VAE", "link": 61}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [62]}], title="Decode Hi-Res Refine"),
            node(41, "ReActorFaceSwap", [7880, 1920], 40, inputs=[{"name": "input_image", "type": "IMAGE", "link": 62}, {"name": "source_image", "type": "IMAGE", "link": 63}], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [64]}], widgets_values=[reactor_enabled, "inswapper_128.onnx", "retinaface_resnet50", reactor_face_restore_model, reactor_face_restore_visibility, reactor_codeformer_weight, "no", "no", "0", "0", 1], title="ReActor Hi-Res Restore Weld"),
            node(42, "SaveImage", [8280, 1920], 41, inputs=[{"name": "images", "type": "IMAGE", "link": 64}], widgets_values=[hires_filename_prefix], title="Save Hi-Res Result"),
            node(43, "CheckpointLoaderSimple", [420, 1160], 41, outputs=[{"name": "MODEL", "type": "MODEL", "links": [37, 56]}, {"name": "CLIP", "type": "CLIP", "links": [66, 67]}, {"name": "VAE", "type": "VAE", "links": [36, 42, 55, 61]}], widgets_values=[detail_checkpoint], title="Load SDXL Detail Checkpoint"),
            node(44, "CLIPTextEncode", [820, 1100], 42, inputs=[{"name": "clip", "type": "CLIP", "link": 66}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [38, 57]}], widgets_values=[positive_prompt], title="Detail Style Prompt"),
            node(45, "CLIPTextEncode", [820, 1340], 43, inputs=[{"name": "clip", "type": "CLIP", "link": 67}], outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [39, 58]}], widgets_values=[negative_prompt], title="Detail Negative Prompt"),
            node(46, "SaveImage", [5880, 1460], 44, inputs=[{"name": "images", "type": "IMAGE", "link": 65}], widgets_values=[f"{intermediate_prefix}/inner_face_sdedit"], title="Save Inner-Face SDEdit"),
        ],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "note": (
                "Rebuilt visual prompt stack: semantic CLIPSeg head mask -> subject-driven PuLID identity + "
                "target-driven IP-Adapter style/composition guidance -> optional ReActor restore weld -> "
                "precision inner-face SDEdit on a base SDXL checkpoint. "
                "Exposed skin harmonization is handled by the remote deterministic postprocess helper. "
                "A separate low-denoise hi-res branch upsamples the clean composite, refines on the base SDXL checkpoint, and reapplies ReActor for identity lock."
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
    parser.add_argument("--detail-checkpoint", default="sd_xl_base_1.0.safetensors")
    parser.add_argument(
        "--disable-reactor",
        action="store_true",
        help="Turn off both ReActor weld stages so the diffusion stack can be evaluated on its own.",
    )
    parser.add_argument("--reactor-face-restore-model", default="GFPGANv1.4.pth")
    parser.add_argument("--reactor-face-restore-visibility", type=float, default=1.0)
    parser.add_argument("--reactor-codeformer-weight", type=float, default=0.5)
    parser.add_argument(
        "--positive-prompt",
        default=(
            "comic book illustration, cinematic lighting, crisp linework, bold color separation, "
            "high facial detail, sharp eyes, clean shading, masterpiece"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "realistic photograph, abstract artifacts, muddy details, soft focus, blurred eyes, "
            "washed out skin texture, visible seam, blank face, gray face plate, duplicate features, "
            "deformed eyes, distorted mouth"
        ),
    )
    parser.add_argument("--semantic-mask-text", default="head, hair, ears, face, neck")
    parser.add_argument("--clipseg-blur", type=float, default=1.5)
    parser.add_argument("--clipseg-threshold", type=float, default=0.35)
    parser.add_argument("--clipseg-dilation-factor", type=int, default=0)
    parser.add_argument("--primary-inpaint-grow-mask-by", type=int, default=10)
    parser.add_argument("--pulid-weight", type=float, default=0.85)
    parser.add_argument("--pulid-projection", default="ortho_v2")
    parser.add_argument("--pulid-fidelity", type=int, default=8)
    parser.add_argument("--ipadapter-preset", default="STANDARD (medium strength)")
    parser.add_argument("--ipadapter-weight", type=float, default=0.55)
    parser.add_argument("--ipadapter-weight-type", default="linear")
    parser.add_argument("--ipadapter-end-at", type=float, default=0.70)
    parser.add_argument("--ipadapter-embeds-scaling", default="K+V")
    parser.add_argument("--primary-seed", type=int, default=84739281)
    parser.add_argument("--primary-steps", type=int, default=30)
    parser.add_argument("--primary-cfg", type=float, default=6.5)
    parser.add_argument("--primary-denoise", type=float, default=0.90)
    parser.add_argument("--secondary-seed", type=int, default=91827364)
    parser.add_argument("--secondary-steps", type=int, default=12)
    parser.add_argument("--secondary-cfg", type=float, default=4.0)
    parser.add_argument("--secondary-denoise", type=float, default=0.16)
    parser.add_argument("--inner-face-area", default="face")
    parser.add_argument("--inner-face-grow", type=int, default=6)
    parser.add_argument("--inner-face-blur", type=int, default=6)
    parser.add_argument("--secondary-inpaint-grow-mask-by", type=int, default=6)
    parser.add_argument("--skin-mask-text", default="face, neck, ears, hands, arms, exposed skin")
    parser.add_argument("--skin-mask-blur", type=float, default=1.5)
    parser.add_argument("--skin-mask-threshold", type=float, default=0.22)
    parser.add_argument("--skin-mask-dilation-factor", type=int, default=2)
    parser.add_argument("--hires-upscale-method", default="lanczos")
    parser.add_argument("--hires-scale-by", type=float, default=1.5)
    parser.add_argument("--hires-seed", type=int, default=27182818)
    parser.add_argument("--hires-steps", type=int, default=16)
    parser.add_argument("--hires-cfg", type=float, default=4.5)
    parser.add_argument("--hires-denoise", type=float, default=0.16)
    parser.add_argument("--sampler-name", default="dpmpp_2m_sde")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--filename-prefix", default="faceswap/visual_prompt_hybrid/final")
    parser.add_argument("--hires-filename-prefix", default="faceswap/visual_prompt_hybrid/final_hires")
    parser.add_argument("--intermediate-prefix", default="faceswap/visual_prompt_hybrid/intermediate")
    args = parser.parse_args()

    workflow_args = vars(args).copy()
    output = Path(workflow_args.pop("output"))
    ui_output = Path(workflow_args.pop("ui_output"))
    workflow_args["reactor_enabled"] = not workflow_args.pop("disable_reactor")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_workflow(**workflow_args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")

    ui_output.parent.mkdir(parents=True, exist_ok=True)
    ui_output.write_text(json.dumps(build_ui_workflow(**workflow_args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ui_output}")


if __name__ == "__main__":
    main()
