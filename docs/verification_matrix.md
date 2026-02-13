# Forge Verification Matrix

## Goal
Prevent untestable feature growth by requiring every shipped capability to map to:
- automated check(s),
- manual validation steps,
- clear pass/fail signals.

Use this document as the source of truth before adding new features.

## Command Entry Point
- `forge verify --mode quick`
- `forge verify --mode full`
- `scripts/run_release_gate.sh`

`quick` validates the current end-to-end platform path.
`full` includes extension hook verification in addition to quick checks.
`run_release_gate.sh` enforces verify + pytest + mypy + ruff in one pass.

On failure, the verification runtime data root is preserved for debugging.
On success, it is deleted unless `--keep-artifacts` is set.

## Check Matrix
| Check ID | Capability | Automated Validation | Manual Validation | Pass Criteria |
|---|---|---|---|---|
| `V001` | Hardware profile | `client.hardware_profile()` fields validated | Run `forge hardware-profile` | Required profile keys exist and values are parseable |
| `V002` | Ingest/filter/versioning | Ingest fixture source, filter, list versions | In Studio run `Ingest -> Filter`, inspect Versions panel | At least 2 dataset versions created |
| `V003` | Run-spec execution | Execute generated run-spec with ingest/filter/versions/hardware-profile | Run `forge run-spec <file>` manually | Run-spec produces expected output rows |
| `V004` | Training artifact pipeline | Train 1 epoch and assert output artifact files | In Studio run Train node, then open Training Curves | `model.pt`, `history.json`, config, tokenizer, contract, reproducibility bundle exist |
| `V005` | Lifecycle + lineage | Validate run state and produced lineage edge | Inspect `.forge/runs` and `.forge/lineage` | Run state is `completed` and produced edge exists |
| `V006` | Training export | Export training shards and verify manifest exists | Run Export node and inspect output folder | `training_manifest.json` exists |
| `V007` | Chat inference path | Run chat against trained model | Use Chat Room panel in Studio | Non-empty model response |
| `V008` (full) | Hook extension points | Run training with generated hooks file and marker output | Train with `--hooks-file` and inspect side effect | Hook side effect file exists after run |

## Feature Gating Rule
A feature is considered releasable only when:
1. It has an automated verification check (`forge verify`) or explicit rationale for deferral.
2. It has a manual validation path documented in this matrix.
3. CI checks pass (`pytest`, `mypy --strict`, `ruff`).

If any condition is missing, mark feature status as `INCOMPLETE`.

## Merge Gate
Before merging feature work, run:

```bash
scripts/run_release_gate.sh
```

Use `--full` for the strictest gate:

```bash
scripts/run_release_gate.sh --full
```

## Manual Studio Validation Pack
Use this sequence to validate the desktop app path:
1. Build pipeline `Ingest -> Filter -> Train -> Export -> Chat`.
2. Run pipeline and verify progress bars update and complete.
3. Open Training Curves and load the generated `history.json`.
4. Verify Run Console includes `run_id` and `artifact_contract_path`.
5. Verify chat returns non-empty response.

## Debugging Workflow
1. Re-run `forge verify --mode quick --keep-artifacts`.
2. Open `verification_report.json` under the printed runtime root.
3. Inspect preserved artifacts and lifecycle files:
   - `<runtime>/runs/index.json`
   - `<runtime>/runs/<run_id>/lifecycle.json`
   - `<runtime>/lineage/model_lineage.json`
4. Fix and re-run until all checks pass.
