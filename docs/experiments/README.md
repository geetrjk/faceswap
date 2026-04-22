# Experiment Tracking System

This directory is the canonical place to track experiment families, latest useful runs, and planned branches.

Legacy outputs and workflows remain where they are today. The catalog maps those legacy names to stable experiment IDs so future work can stay organized without rewriting history first.

## Status Labels

- `planned`: idea is cataloged but not yet built.
- `in progress`: partially built or partially tested.
- `tried`: executed and documented, but not the current recommendation.
- `recommended`: current best branch for its goal.
- `rejected`: executed and judged not worth pursuing further except as historical reference.

## Theme Codes

- `IDB`: identity baseline / ReActor baseline
- `IDI`: InstantID and crop-stitch identity experiments
- `SWB`: swap-and-bake sidecars
- `VPH`: visual-prompt hybrid generation experiments
- `SKN`: skin and face color correction
- `HIR`: hi-res refinement and sharpening
- `ARC`: planned architecture changes not yet fully built

## Experiment ID Rule

Use `THEME-NNN`, for example `VPH-005`.

One experiment ID should represent one conceptual branch, not every timestamped rerun. Timestamped runs belong under that experiment ID as executions.

## Naming Convention

For future experiments, use the same slug everywhere:

- experiment slug: lowercase, hyphenated, concise
- workflow filenames: `workflows/experiments/<theme>/<EXP_ID>__<slug>_{api,ui}.json`
- run script filenames: `scripts/experiments/run_<EXP_ID>__<slug>.py`
- local output directories: `test_outputs/<theme>/<EXP_ID>__<slug>/<YYYYMMDD_HHMMSS>__<target_slug>/`
- saved handoff snapshots: `saved_results/<theme>/<EXP_ID>__<slug>/<YYYYMMDD_HHMMSS>/`
- report file: `docs/experiments/reports/<EXP_ID>__<slug>.md`

Example:

- workflow: `workflows/experiments/vph/VPH-008__target-style-ipadapter-split_api.json`
- outputs: `test_outputs/vph/VPH-008__target-style-ipadapter-split/20260423_101500__superman/`
- report: `docs/experiments/reports/VPH-008__target-style-ipadapter-split.md`

## Report Structure

Every new experiment should get a short report file with the following fields:

- `Experiment ID`
- `Name`
- `Purpose`
- `Workflow / Script`
- `Method`
- `Key Parameters`
- `Inputs`
- `Outputs`
- `Status`
- `Findings`
- `Next Decision`

## Practical Rules

- Keep one experiment ID per idea and many runs under it.
- Do not create a new workflow filename just because one parameter changed.
- Promote only one latest useful run per experiment in the catalog.
- If a run is only a rerun or a small parameter sweep, keep it under the same experiment ID and note the timestamped output directory.
- When a family is superseded, mark the old entry `tried` or `rejected` and point to the newer ID.
- Keep `docs/STATUS.md` as a short handoff file; keep the durable history in `docs/experiments/CATALOG.md`.

## Legacy Mapping Rule

Current legacy flat folders in `test_outputs/` remain valid. The catalog treats them as historical executions attached to experiment IDs until the repo transitions to the new structure.
