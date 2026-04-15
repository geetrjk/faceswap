# Known Mistakes And Fixes

## Workflow Input Names

Problem: The workflow originally loaded `subject.jpg` and `target.png`, but the repo inputs are `subject_5 year curly.webp` and `superman.png`.

Fix: Keep builder defaults aligned with checked-in assets, or deploy aliases and regenerate the workflow to match those aliases before queueing.

## ReActor Node Registration

Problem: A pod may have `ComfyUI-ReActor` installed on disk while live `/object_info` still does not list `ReActorFaceSwap`.

Fix: Use the exact class name `ReActorFaceSwap`, then do a real pod/container restart from the SimplePod control plane. Do not repeatedly poll or reinstall unless `scripts/simplepod.py preflight` still fails after restart.

## Missing `inswapper_128.onnx`

Problem: ReActor can be installed but fail at runtime if `models/insightface/inswapper_128.onnx` is missing.

Fix: Add the model to `/app/ComfyUI/models/insightface/inswapper_128.onnx`, then rerun preflight.

## Missing Authenticator Token

Problem: `/object_info` can return 401 even over localhost when `ComfyUI-Authenticator` has no `/app/ComfyUI/login/PASSWORD` token file loaded.

Fix: Run `.venv/bin/python scripts/simplepod.py init-auth`, then restart the ComfyUI backend so the authenticator loads the token.

## Overbuilt First Workflow

Problem: The first workflow tried to use diffusion cleanup and Impact Pack detailer nodes before the base swap was proven. On a 12 GB RTX 3060 pod, that adds unnecessary checkpoint/model pressure.

Fix: Keep the source workflow ReActor-only until `faceswap/final_*` is produced and visually accepted. Add diffusion/detailer cleanup as a second milestone.

## Face Restore Defaults

Problem: A weaker `GFPGANv1.4.pth` visibility value produced a softer face than needed.

Fix: Use full restore visibility plus ReActor FaceBoost as the default. Keep `--no-face-boost` available for plain diagnostic swaps.

## ReActor As The Wrong Identity Core

Problem: ReActor can make a successful face-region swap while still inheriting the target's hair, skull shape, age cues, and surrounding character geometry. This is the wrong core mechanism when the goal is subject-first identity with only target pose/style guidance.

Fix: Keep ReActor as the baseline or an optional late-stage diagnostic. Use the separate SDXL + InstantID experiment when preserving subject hairstyle, head shape, and age cues matters.

## ComfyUI Output Caching

Problem: A rerun can report success without writing a fresh output if ComfyUI caches unchanged nodes and output branches.

Fix: For verification runs, change `--filename-prefix` and/or `--intermediate-prefix`, change an input, or clear the server cache before assuming an output node is broken.

## API Prompt Loaded As UI Workflow

Problem: Loading `faceswap_subject_on_character_api.json` in the ComfyUI browser can produce an empty graph and the error that the workflow has no output nodes.

Fix: Load `faceswap_subject_on_character_ui.json` in the browser. Keep `_api` for `/prompt` automation and `_ui` for manual ComfyUI graph execution.

## UI Automation

Problem: ComfyUI browser automation can get stuck on auth or SPA state when SimplePod ports change.

Fix: Prefer SSH/API checks for setup and use human-in-the-loop browser review only when visual workflow actions are faster or safer.

## Cartoon Target Keypoints

Problem: InstantID face keypoint extraction can return a very sparse keypoint image for stylized cartoon targets, which makes target pose transfer weak even when the subject identity result is good.

Fix: Keep subject identity strength high and add a stronger pose/composition control branch for the target, such as OpenPose/DWPose/depth/lineart, instead of making the target style or identity weights dominate.

## InstantID Full-Image Drift

Problem: Sampling the InstantID experiment from an empty latent lets the model redraw the entire target, so the body, suit, cape, and background can change even when the desired edit is face-only.

Fix: Build the InstantID path as masked inpainting: generate a soft target-face mask, pass it to `ApplyInstantIDAdvanced` and `VAEEncodeForInpaint`, sample from the target image latent with partial denoise, then composite the decoded result back over the original target through the same mask.

## InstantID Gray Inpaint Plate

Problem: With the current SDXL base masked-inpaint graph, low denoise settings can leave a flat gray or blank mask-shaped plate in the target face region instead of generating facial features.

Fix: For diagnostics, raise denoise high enough to force generation and add negative prompt terms for blank/faceless/mask-plate artifacts. If the face still lands poorly, stop blind API tweaking and inspect the UI workflow checkpoints; the pipeline likely needs a better inpaint checkpoint or stronger pose/composition branch.

## InstantID Colored Control Patch

Problem: Switching from SDXL base to `sd_xl_base_1.0_inpainting_0.1.safetensors` improved the setup correctness but still produced a colorful abstract face patch in the crop-stitch raw inpaint. Lowering InstantID control strength to `0.05` did not remove it, so the failure is not just an over-strong current ControlNet weight.

Fix: Treat the sparse cartoon face keypoints as insufficient structure. Add a stronger pose/composition branch such as DWPose/OpenPose/depth/lineart before more prompt-only or tiny weight sweeps.

## Canny Control Is Not Enough

Problem: Adding a target-crop Canny edge guide with `controlnet-canny-sdxl-1.0-small.safetensors` made the crop-stitch structure signal much richer and ran successfully on `8188`, but the raw inpaint still produced the green/colorful mask-like face artifact. A diagnostic with InstantID pose/control strength lowered to `0.05` and Canny strength raised to `0.65` made the artifact harsher rather than better.

Fix: Keep the Canny branch as an inspectable structural baseline, but do not assume stronger edge control alone solves this failure. Try a different structural/control family or identity/inpaint stack before spending more time on prompt-only or tiny strength sweeps.

## Masked Swap Bake Gray Plate

Problem: The first swap-and-bake template used `VAEEncodeForInpaint` on the ReActor-swapped image with the swapped face mask. In this environment it produced the same flat gray face plate at denoise `0.20` and still did so at `0.45`.

Fix: For the checked-in sidecar experiment, keep the face mask as an inspection output but use plain `VAEEncode` for a full-image low-denoise bake from the swapped image. This avoids the gray plate and keeps the experiment focused on whether a ReActor-first bake is a more stable alternative.

## Missing Visual Prompt Nodes

Problem: The visual-prompt hybrid reference graph expects PuLID, IP-Adapter Plus, and SAM/Impact nodes, but the live `8188` backend did not expose PuLID, IP-Adapter, or SAM/Impact classes in `/object_info`.

Fix: Do not load the reference JSON verbatim on this backend. The checked-in `visual_prompt_hybrid_experiment` is an environment fallback that keeps the automation shape testable with available nodes: FaceSegmentation mask, SDXL head fill, ReActor likeness snap, and full-image low-denoise bake. Install and restart PuLID/IP-Adapter/SAM before attempting a true visual-prompt validation.

## Crop Region Reused As Edit Mask

Problem: In the crop-stitch InstantID experiment, using the same aggressively grown mask for the crop bbox and the actual edit area lets SDXL repaint too much of the crop, including suit/cape pixels.

Fix: Keep the crop region generous, but shrink the crop-local mask before `ApplyInstantIDAdvanced`, `VAEEncodeForInpaint`, and final crop paste. Save both masks separately so the UI makes the distinction inspectable.

## Partial SDXL Inpaint Download

Problem: A large remote checkpoint download can stall and leave a partial `.safetensors` file in `models/checkpoints`, which ComfyUI may list even though it is unusable.

Fix: Use `scripts/setup_simplepod_instantid.sh` or `.venv/bin/python scripts/simplepod.py install-instantid`, which now size-checks required model files and resumes partial downloads with `curl -C -`. Treat any manually downloaded `.partial` file as suspect unless its full size is verified. The InstantID builders default to `sd_xl_base_1.0_inpainting_0.1.safetensors`, so a missing or partial inpaint checkpoint should fail setup/preflight before queueing.

## Slow Fresh-Pod Setup Loops

Problem: A fresh SimplePod container can lose all custom nodes/models. Reinstalling manually caused repeated delays: GitHub `git clone` stalled, `wget` stalled on the SDXL checkpoint, remote command output was buffered until completion, and partial model files were treated as valid by a simple existence check.

Fix: Use the setup helper path instead of ad hoc SSH installs. The helper streams remote install output, times out Git operations, falls back to GitHub zip archives, uses resumable model downloads, and checks minimum file sizes before reporting success.

## New Custom Nodes Need Restart

Problem: Freshly cloned custom nodes can import from disk, but live `/object_info` will not list them if ComfyUI started before they were installed.

Fix: After installing `ComfyUI_InstantID` or `ComfyUI_FaceAnalysis`, do a real SimplePod/ComfyUI restart before preflight and queueing. Do not debug workflow JSON while the live node registry is stale.

## Stale SimplePod SSH Endpoint

Problem: `scripts/simplepod.py profile` can fail even with valid credentials if the pod was restarted and SimplePod assigned a new external SSH port.

Fix: Refresh `SIMPLEPOD_SSH_HOST` and `SIMPLEPOD_SSH_PORT` in `.env` from the SimplePod UI, then rerun `.venv/bin/python scripts/simplepod.py profile`.

## Remote Helper Assumed `python`

Problem: After a fresh server restart, the pod exposed `python3` for non-interactive remote commands but not `python`. `scripts/simplepod.py` used `python` inside remote preflight, install, auth, temp-start, and queue helpers, which made preflight report false failures and blocked reinstall steps even though the base environment was otherwise healthy.

Fix: Use `python3` consistently in remote commands inside `scripts/simplepod.py`. If preflight suddenly reports `python: command not found` after a restart, treat it as a helper issue first, patch the helper, then rerun install and preflight.

## SAM Seeded From Face Mask Produced Bad Regions

Problem: The first true visual-prompt stack converted the `FaceSegmentation` mask into `SEGS` and fed that into `SAMDetectorCombined`. On the current cartoon target, that path produced structurally wrong masks that spilled into body and border regions, which made the inpaint result unusable.

Fix: Do not put `MaskToSEGS -> SAMDetectorCombined` on the active path for this workflow. Keep SAM installed and validated, but drive the current pipeline from the direct `FaceSegmentation -> GrowMask -> FeatherMask` mask until a better detector source is available.
