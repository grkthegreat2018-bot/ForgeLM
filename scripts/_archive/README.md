# scripts/_archive

Obsolete one-off scripts and experiment artifacts, retained for reference
but no longer part of the supported workflow. Nothing in `forge/`,
`forge_gui/`, `tests/`, or `scripts/` may import or execute these files.

## Why archived (2026-09 cleanup)

ForgeLM V2 (Jamba-3B hybrid) became the sole base model. All scripts tied
to the deleted model lineage (ForgeLM V2 Light / LFM2.5 / V9 / V10 / V11 /
V12-VLM presets and checkpoints) were moved here along with one-off R&D
round experiment scripts (R25–R48) and old evolution result JSONs.

Categories:
- `bench_*.py`, `test_*.py` — one-off benchmarks and smoke tests for
  deleted checkpoints (`ForgeLM_V2_Light*.safetensors`, LFM2.5 GGUFs).
- `train_r*_qlora.py`, `gen_*_data*.py` — per-round training/data-gen
  scripts for superseded model versions.
- `r2*/r3*/r4*/*.json` — evolution experiment result blobs.
- `port_jamba_to_forgelm_v2.py`, `inspect_jamba_tokenizer.py` — completed
  one-shot port/inspection tasks (kept for provenance).

## Restoration policy

Do not re-point production code at files in this directory. If a script's
functionality is needed, port it to the ForgeLM V2 Jamba checkpoint
(`research/checkpoints/ForgeLM_V2.safetensors`, config `forgelm_v2`) and
place it back under `scripts/` with updated paths.
