# Experiment Catalog

This catalog groups the major experiment branches by theme, assigns stable IDs, and identifies the latest useful version inside each family.

## Summary Table

| ID | Name | Theme | Status | Implementation | Latest useful workflow/script | Latest useful outputs |
| --- | --- | --- | --- | --- | --- | --- |
| `IDB-001` | ReActor baseline | Identity baseline | `tried` | built | `build_faceswap_workflow.py` | `test_outputs/final_*.png`, `test_outputs/variant_*` |
| `IDI-001` | InstantID masked inpaint baseline | Identity preservation | `rejected` | built | `build_instantid_workflow.py` | `test_outputs/instantid_*`, `test_outputs/instantid_masked/*` |
| `IDI-002` | InstantID ablations | Identity preservation | `rejected` | built | `build_instantid_workflow.py` | `test_outputs/instantid_nokps_20260412`, `instantid_subjectkps_20260412`, `instantid_globalid_maskpaint_20260412` |
| `IDI-003` | InstantID crop-stitch family | Identity preservation | `rejected` | built | `build_instantid_crop_stitch_workflow.py` | `test_outputs/instantid_crop_stitch_*` |
| `SWB-001` | Swap-and-bake sidecar | Identity / bake | `tried` | built | `build_swap_and_bake_workflow.py` | `test_outputs/swap_and_bake_20260413_full_bake` |
| `VPH-001` | Visual-prompt fallback initial | Visual-prompt generation | `tried` | built | `build_visual_prompt_hybrid_workflow.py` | `test_outputs/visual_prompt_hybrid_20260413` |
| `VPH-002` | Early visual-prompt tuning sweep | Visual-prompt generation | `tried` | built | `build_visual_prompt_hybrid_workflow.py` | `test_outputs/visual_prompt_hybrid_20260414_*`, `20260415_*`, `20260416_*` |
| `VPH-003` | Subject-matrix progression | Visual-prompt generation | `tried` | built | `run_visual_prompt_subject_matrix.py` | `test_outputs/visual_prompt_subject_matrix_20260417_*` |
| `VPH-004` | Spider-Man target-agnostic validation | Visual-prompt generation | `recommended` | built | `run_visual_prompt_subject_matrix.py` | `test_outputs/visual_prompt_subject_matrix_spiderman_final2`, `..._hires_final` |
| `VPH-005` | Superman detail-preservation matrix | Visual-prompt generation | `recommended` | built | `run_visual_prompt_subject_matrix.py` | `test_outputs/visual_prompt_subject_matrix_superman_20260418_131858` |
| `VPH-006` | Stable visual-prompt snapshot | Visual-prompt generation | `in progress` | built | `workflows/stable/visual_prompt_hybrid_v1_{api,ui}.json` | no dedicated run folder yet |
| `SKN-001` | Generative exposed-skin tail | Skin / color | `rejected` | partially built | visual-prompt workflow tail | folded into legacy `visual_prompt_hybrid` and matrix runs |
| `SKN-002` | Deterministic exposed-skin postprocess | Skin / color | `recommended` | built | `remote_skin_tone_postprocess.py` | `final_postprocess_*`, `final_hires_postprocess_*` in visual-prompt matrix folders |
| `SKN-003` | Face reference color method matrix | Skin / color | `in progress` | built | `remote_face_color_reference_postprocess.py`, `run_face_color_method_matrix.py` | `test_outputs/face_color_method_matrix_superman_20260422*` |
| `HIR-001` | Hi-res refine branch | Hi-res / refinement | `recommended` | built | visual-prompt hi-res branch | `test_outputs/visual_prompt_subject_matrix_superman_hires_final`, `...spiderman_hires_final` |
| `HIR-002` | Deterministic sharpen sidecar | Hi-res / refinement | `tried` | built | `remote_hires_sharpen.py` | `final_hires_sharp_*` inside visual-prompt matrices |
| `ARC-001` | Target-style explicit IP-Adapter split | Planned architecture | `planned` | planned | not built | none |
| `ARC-002` | Replace ReActor weld with FaceID or PhotoMaker core | Planned architecture | `planned` | planned | not built | none |
| `ARC-003` | Stronger SDXL detail model / refiner swap | Planned architecture | `planned` | planned | not built | none |
| `ARC-004` | Head relighting / IC-Light harmonization | Planned architecture | `planned` | planned | not built | none |

## Identity Baseline

### `IDB-001` ReActor baseline

- Purpose: prove the repo can do a reliable end-to-end face swap with current pod constraints.
- Workflow or script: `scripts/build_faceswap_workflow.py`, `workflows/faceswap_subject_on_character_{api,ui}.json`
- Key method / idea: ReActor-only swap with GFPGAN / FaceBoost restore variants.
- Important parameters / settings: final promoted baseline used `GFPGANv1.4.pth` with full restore visibility and FaceBoost.
- Input subjects used: initially `subject_5 year curly.webp` against `superman.png`; restore variants also tested on the same default pair.
- Output location: `test_outputs/final_00001_.png`, `test_outputs/final_00002_.png`, `test_outputs/variant_*`
- Status: `tried`
- Short summary of findings: stable baseline and useful setup milestone, but wrong long-term architecture when subject-first identity, age cues, and hair preservation matter.

### `IDI-001` InstantID masked inpaint baseline

- Purpose: replace ReActor-first swapping with subject-first identity conditioning inside SDXL generation.
- Workflow or script: `scripts/build_instantid_workflow.py`, `workflows/instantid_subject_pose_style_{api,ui}.json`
- Key method / idea: InstantID identity from subject, target face keypoints, target latent inpaint, masked composite back over target.
- Important parameters / settings: identity weight `1.35`, pose strength `0.2`, face mask grow `16`, blur `31`, denoise `0.86`
- Input subjects used: default subject-first tests with `subject_5 year curly.webp` on `superman.png`
- Output location: `test_outputs/instantid_iter_20260412`, `instantid_deployed_20260412`, `instantid_highdenoise_20260412`, `instantid_lowpose_20260412`, `instantid_masked/*`
- Status: `rejected`
- Short summary of findings: preserved subject hair and child structure better than ReActor, but sparse cartoon keypoints and SDXL masked inpaint produced gray plates or misaligned face geometry.

### `IDI-002` InstantID ablations

- Purpose: determine whether the main failure was caused by target keypoints, masking, or control strength.
- Workflow or script: `scripts/build_instantid_workflow.py`
- Key method / idea: remove or weaken target guidance and compare no-keypoints, subject-keypoints, global/no-mask, and denoise variants.
- Important parameters / settings: no-keypoints, subject-keypoints, global identity/no-mask, denoise sweeps
- Input subjects used: same default subject-target pair as `IDI-001`
- Output location: `test_outputs/instantid_nokps_20260412`, `instantid_subjectkps_20260412`, `instantid_globalid_maskpaint_20260412`, `instantid_targetimagekps_20260412`
- Status: `rejected`
- Short summary of findings: none of the ablations fixed the failure; they mostly proved that sparse cartoon face structure was the limiting factor, not just one bad weight.

### `IDI-003` InstantID crop-stitch family

- Purpose: improve containment and structure by sampling on a local target crop instead of the whole image.
- Workflow or script: `scripts/build_instantid_crop_stitch_workflow.py`, `workflows/instantid_crop_stitch_experiment_{api,ui}.json`
- Key method / idea: generous crop region, smaller edit mask, local inpaint, stitch back to the unchanged target.
- Important parameters / settings: separate crop-region and edit-mask controls, SDXL inpaint checkpoint trials, pose strength reductions, Canny structural control branch
- Input subjects used: primarily `subject_5 year curly.webp` on `superman.png`
- Output location: `test_outputs/instantid_crop_stitch_20260412`, `instantid_crop_stitch_20260413_setupfix`, `..._pose005`, `..._inpaint_ckpt`, `..._canny`, `..._canny_pose005`
- Status: `rejected`
- Short summary of findings: crop containment worked and Canny improved structural signal, but the raw masked face still collapsed into colorful abstract artifacts. Most useful historical version is the Canny run because it proved the structure branch executed, not because quality was acceptable.

## Swap-And-Bake

### `SWB-001` Swap-and-bake sidecar

- Purpose: test whether a ReActor-first image followed by low-denoise SDXL bake is more stable than InstantID masked inpaint.
- Workflow or script: `scripts/build_swap_and_bake_workflow.py`, `workflows/swap_and_bake_experiment_{api,ui}.json`
- Key method / idea: ReActor swap for likeness, then full-image SDXL bake from the swapped result instead of masked inpaint.
- Important parameters / settings: plain `VAEEncode` retained after `VAEEncodeForInpaint` produced gray face plates at denoise `0.20` and `0.45`
- Input subjects used: default Superman baseline pair
- Output location: `test_outputs/swap_and_bake_20260413`, `swap_and_bake_20260413_denoise045`, `swap_and_bake_20260413_full_bake`, `saved_results/swap_and_bake_20260413`
- Status: `tried`
- Short summary of findings: coherent and stable compared with InstantID, but still inherits ReActor identity limitations. Keep as a fallback sidecar, not the main direction.

## Visual-Prompt Generation

### `VPH-001` Visual-prompt fallback initial

- Purpose: build a testable approximation of the intended PuLID + IP-Adapter + semantic-mask architecture on the current backend.
- Workflow or script: `scripts/build_visual_prompt_hybrid_workflow.py`, `workflows/visual_prompt_hybrid_experiment_{api,ui}.json`
- Key method / idea: semantic head mask, SDXL generation, ReActor restore weld, low-denoise bake fallback.
- Important parameters / settings: early fallback graph before the later subject-matrix and hi-res refinements
- Input subjects used: default Superman pair
- Output location: `test_outputs/visual_prompt_hybrid_20260413`, `saved_results/visual_prompt_hybrid_20260413`
- Status: `tried`
- Short summary of findings: avoided the worst InstantID artifacts and became the basis for later work, but was still an environment fallback rather than a mature target-agnostic architecture.

### `VPH-002` Early visual-prompt tuning sweep

- Purpose: tune ordering, mask handling, and seam behavior before committing to full subject matrices.
- Workflow or script: `scripts/build_visual_prompt_hybrid_workflow.py`
- Key method / idea: early branch ordering and mask strategy sweeps
- Important parameters / settings: source-first ordering, hair-focused variants, true-stack masked variants, clipseg normalization, seam-bake trials
- Input subjects used: mainly single-subject probes around the default Superman target
- Output location: `test_outputs/visual_prompt_hybrid_20260414_source_first`, `..._hairgrow`, `..._hairmid`, `..._hairbake`, `..._20260415_true_stack*`, `..._20260415_tuned`, `..._20260415_finalpass`, `..._20260416_direct_clipseg_normmask`, `..._20260416_seambake`
- Status: `tried`
- Short summary of findings: this family produced the design decisions that later became the stable matrix branch, but the directories are near-duplicate tuning probes. The most useful value is historical reasoning, not any one output folder.

### `VPH-003` Subject-matrix progression

- Purpose: validate visual-prompt behavior across multiple subject complexions and age ranges.
- Workflow or script: `scripts/run_visual_prompt_subject_matrix.py`
- Key method / idea: repeated subject-matrix sweeps while tuning masks, prompts, and restore behavior
- Important parameters / settings: iterative changes across April 17 subject-matrix runs
- Input subjects used: `7_year_old_face`, `african_child`, `south_asian_teenager`, `white_european_child`
- Output location: `test_outputs/visual_prompt_subject_matrix_20260417_185100`, `..._185331`, `..._213315`, `..._215827`, `..._221332`
- Status: `tried`
- Short summary of findings: this is a duplicated iteration family. `20260417_221332` is the latest useful pre-April-18 version; earlier folders are mostly superseded.

### `VPH-004` Spider-Man target-agnostic validation

- Purpose: prove exposed-skin logic does not fire on fully covered costume targets.
- Workflow or script: `scripts/run_visual_prompt_subject_matrix.py`
- Key method / idea: run the visual-prompt branch on `spiderman.png` instead of Superman and check that glove/suit regions are not treated as skin.
- Important parameters / settings: tightened skin gate, texture-smoothness gate, minimum meaningful non-face skin area, base-to-hires dependency
- Input subjects used: `7_year_old_face`, `african_child`, `south_asian_teenager`, `white_european_child`
- Output location: `test_outputs/visual_prompt_subject_matrix_spiderman_20260417`, `..._230003`, `..._final`, `..._final2`, `..._hires_final`
- Status: `recommended`
- Short summary of findings: final base and hi-res matrices correctly skip exposed-skin harmonization with `STATUS=skipped` / `REASON=no_non_face_skin`. This is the main proof that the non-face skin correction became target-agnostic.

### `VPH-005` Superman detail-preservation matrix

- Purpose: preserve sharper facial detail while keeping the target-agnostic exposed-skin logic.
- Workflow or script: `scripts/run_visual_prompt_subject_matrix.py`
- Key method / idea: style-only CLIP prompt, separate detail checkpoint, precision inner-face SDEdit, smaller inner-face mask, later hi-res branch
- Important parameters / settings: inner-face grow/blur reduced to `6/6`, detail checkpoint `sd_xl_base_1.0.safetensors`, hi-res low-denoise refine
- Input subjects used: `7_year_old_face`, `african_child`, `south_asian_teenager`, `white_european_child`, `subject_5_year_curly`
- Output location: `test_outputs/visual_prompt_subject_matrix_superman_20260418_131858`
- Status: `recommended`
- Short summary of findings: current best visually useful Superman matrix. Sharper eyes, mouth edges, and facial shading than earlier matrices while keeping skin postprocess and hi-res sidecars working.

### `VPH-006` Stable visual-prompt snapshot

- Purpose: keep a checked-in stable workflow snapshot separate from the actively edited experiment JSON.
- Workflow or script: `workflows/stable/visual_prompt_hybrid_v1_{api,ui}.json`
- Key method / idea: pin a versioned workflow pair for reuse and deployment
- Important parameters / settings: current branch snapshot from the most recent visual-prompt work
- Input subjects used: not yet tied to a dedicated matrix rerun
- Output location: no dedicated `test_outputs` folder yet
- Status: `in progress`
- Short summary of findings: implementation exists, but it is not yet paired with a dedicated cataloged run folder. It should become the next stable reference once revalidated under the new experiment structure.

## Skin And Face Color

### `SKN-001` Generative exposed-skin tail

- Purpose: solve non-face skin mismatch inside the workflow itself.
- Workflow or script: early visual-prompt workflow tail
- Key method / idea: masked low-denoise generative exposed-skin inpaint
- Important parameters / settings: workflow-tail inpaint on semantic exposed-skin regions
- Input subjects used: full subject matrix on Superman-style target
- Output location: embedded in early `visual_prompt_hybrid` and subject-matrix runs
- Status: `rejected`
- Short summary of findings: repainted hands into gray patches and degraded realism. This branch should remain rejected.

### `SKN-002` Deterministic exposed-skin postprocess

- Purpose: harmonize non-face exposed skin without rediffusing the body.
- Workflow or script: `scripts/remote_skin_tone_postprocess.py`, `scripts/simplepod.py postprocess-skin-tone`
- Key method / idea: refine semantic skin mask, exclude solved face/neck, gate by skin plausibility and texture, transfer solved face tone onto exposed non-face skin deterministically
- Important parameters / settings: minimum region gating, texture-smoothness rejection, base-resolution pass must succeed before hi-res pass runs
- Input subjects used: Superman and Spider-Man subject matrices
- Output location: `final_postprocess_*`, `final_hires_postprocess_*`, `target_skin_mask_refined_*` inside visual-prompt matrix folders
- Status: `recommended`
- Short summary of findings: best current architecture for non-face exposed skin. Works on Superman and correctly skips Spider-Man gloves.

### `SKN-003` Face reference color method matrix

- Purpose: test whether face-local realism can be improved by source-referenced face color correction sidecars.
- Workflow or script: `scripts/remote_face_color_reference_postprocess.py`, `scripts/run_face_color_method_matrix.py`, `scripts/run_visual_prompt_subject_matrix.py`
- Key method / idea: compare direct chroma transfer, mean/std matching, selective LAB shift, RGB gain with luminance preservation, and selective RGB gain
- Important parameters / settings: current default sidecar is `rgb_gain_selective_preserve_y` at strength `0.45`; earlier sweeps also used `lab_mean_shift`, `lab_mean_std`, `ycbcr_mean_shift`, `ycbcr_mean_std`, `rgb_gain_preserve_y`, `lab_selective_shift`
- Input subjects used: `african_child`, `south_asian_teenager`, `white_european_child`, `subject_5_year_curly`
- Output location: `test_outputs/face_color_method_matrix_superman_20260422`, `..._rerun`, `..._altideas`, `..._tuned`, plus single-image probes in `test_outputs/variants/`
- Status: `in progress`
- Short summary of findings: global mean/std transfer is too synthetic. Conservative illuminant-style correction is more plausible, but this family is still not the final answer because the deeper mismatch appears to be generation-conditioning, not just postprocess color transfer.

## Hi-Res Refinement

### `HIR-001` Hi-res refine branch

- Purpose: improve detail and realism after the clean base composite without reintroducing target-agnostic skin failures.
- Workflow or script: visual-prompt hi-res branch inside `build_visual_prompt_hybrid_workflow.py`
- Key method / idea: upscale clean composite, low-denoise SDXL refine, optional ReActor identity weld, optional deterministic hi-res skin postprocess
- Important parameters / settings: scale-by `1.5`, hi-res steps `16`, cfg `4.5`, denoise `0.16`, explicit namespaced hi-res prefixes
- Input subjects used: Superman and Spider-Man subject matrices, plus earlier single-subject hi-res probes
- Output location: `test_outputs/visual_prompt_hires_probe_superman*`, `visual_prompt_hires_probe_spiderman*`, `visual_prompt_subject_matrix_superman_hires_final`, `visual_prompt_subject_matrix_spiderman_hires_final`
- Status: `recommended`
- Short summary of findings: useful and now safe once hi-res skin postprocess is gated on base-pass success. Latest useful family members are the final Superman and Spider-Man hi-res matrices; the earlier `v2`/`v3` probe folders are near-duplicate tuning steps.

### `HIR-002` Deterministic sharpen sidecar

- Purpose: improve eye and hair-edge crispness without confusing quality comparisons by overwriting the main final output.
- Workflow or script: `scripts/remote_hires_sharpen.py`
- Key method / idea: guided unsharp mask / contrast sidecar using the inner-face mask
- Important parameters / settings: separate `final_hires_sharp_*` outputs; keep as sidecar only
- Input subjects used: Superman detail-preservation matrix and later single-run confirmation
- Output location: `final_hires_sharp_*` inside `test_outputs/visual_prompt_subject_matrix_superman_20260418_131858/*` and later matrix runs
- Status: `tried`
- Short summary of findings: visually useful for crispness, but not a structural recovery method. Keep for comparisons only, not as the only final output path.

## Planned Architecture Experiments

### `ARC-001` Target-style explicit IP-Adapter split

- Purpose: make target style and target color first-class conditioning signals instead of relying mostly on masked target latent plus postprocess correction.
- Workflow or script: planned change to `scripts/build_visual_prompt_hybrid_workflow.py`
- Key method / idea: keep subject identity on PuLID or FaceID, but feed the target image into IP-Adapter as explicit style/composition guidance rather than feeding the subject image into both identity and style branches.
- Important parameters / settings: separate identity-vs-style conditioning paths; preserve masked target latent inpaint base
- Input subjects used: planned Superman and Spider-Man subject matrices, starting with the current five-subject Superman set
- Output location: not built
- Status: `planned`
- Short summary of findings: best current next-step architecture because the present graph is over-weighted toward subject identity and under-conditioned on target style/color.

### `ARC-002` Replace ReActor weld with FaceID or PhotoMaker core

- Purpose: reduce the downstream identity snap-back that may be contributing to face-rendering and color mismatch.
- Workflow or script: planned new branch off `build_visual_prompt_hybrid_workflow.py`
- Key method / idea: test `IP-Adapter-FaceID-PlusV2-SDXL` or PhotoMaker V2 as the identity core so identity stays inside generation rather than being rewelded afterward by ReActor.
- Important parameters / settings: compare full replacement vs reduced ReActor dependence
- Input subjects used: same current Superman subject matrix set first
- Output location: not built
- Status: `planned`
- Short summary of findings: high-value architecture branch because it attacks a probable root cause instead of treating the face after the fact.

### `ARC-003` Stronger SDXL detail model / refiner swap

- Purpose: improve facial naturalness, material rendering, and color realism from the generation model itself.
- Workflow or script: planned checkpoint experiment in `build_visual_prompt_hybrid_workflow.py`
- Key method / idea: test `PuLID-v1.1` compatibility plus stronger detail checkpoints or SDXL Refiner for the low-denoise detail stage
- Important parameters / settings: compare current `sd_xl_base_1.0` detail stage against refiner or stronger portrait-capable SDXL alternatives
- Input subjects used: current Superman detail matrix set, then Spider-Man regression set
- Output location: not built
- Status: `planned`
- Short summary of findings: should be treated as an upstream generation-model branch, not a color postprocess branch.

### `ARC-004` Head relighting / IC-Light harmonization

- Purpose: test whether the apparent skin mismatch is really a local lighting mismatch rather than a tone mismatch.
- Workflow or script: planned sidecar or branch after generated-head stage and before final composite
- Key method / idea: relight the generated head or pre-composite face crop instead of recoloring it directly
- Important parameters / settings: keep effect local to the head crop; do not let it repaint the whole body or background
- Input subjects used: start with the current Superman detail matrix set where tone mismatch is easiest to see
- Output location: not built
- Status: `planned`
- Short summary of findings: lower priority than `ARC-001` through `ARC-003`, but a good branch if the remaining error still reads as local white-balance or illumination mismatch after conditioning fixes.

## Duplicate And Near-Duplicate Mapping

- `visual_prompt_subject_matrix_20260417_185100`, `..._185331`, `..._213315`, `..._215827`, and `..._221332` are one iterative family. Latest useful member: `..._221332`. Superseded by `VPH-005`.
- `visual_prompt_hires_probe_superman`, `..._v2`, `..._v3` are one hi-res probe family. Latest useful family outcome is `HIR-001`, specifically `visual_prompt_subject_matrix_superman_hires_final`.
- `visual_prompt_hires_probe_spiderman`, `..._v2`, `..._v3` are the Spider-Man hi-res probe family. Superseded by `visual_prompt_subject_matrix_spiderman_hires_final`.
- `visual_prompt_spiderman_single_retest`, `...retest2`, `...retest3` are single-subject debug reruns. Superseded by `VPH-004`.
- `visual_prompt_subject5_curly_superman_compare` and `visual_prompt_subject5_curly_spiderman_compare` are targeted one-subject comparisons. Useful as spot checks, but superseded by matrix runs.
- `face_color_method_matrix_superman_20260422_rerun` is a reproducibility rerun of the original method board generation. `..._altideas` is the expanded-method exploration. `..._tuned` is the latest useful member.
- `instantid_crop_stitch_20260413_setupfix`, `..._pose005`, `..._inpaint_ckpt`, `..._canny`, and `..._canny_pose005` are one crop-stitch tuning family. Latest historically useful member: `..._canny` because it proves the best structural control branch that still failed.
- `saved_results/visual_prompt_hybrid_20260413` and `saved_results/swap_and_bake_20260413` are preserved handoff snapshots, not the latest recommended runs.

## Current Recommended Starting Points

- Identity / visual generation: `VPH-005`
- Target-agnostic validation: `VPH-004`
- Non-face exposed skin harmonization: `SKN-002`
- Hi-res refinement: `HIR-001`
- Face color sidecar: `SKN-003`, but treat it as provisional and secondary to the planned architecture changes
- Next build candidates: `ARC-001`, then `ARC-002`, then `ARC-003`
