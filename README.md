# Forge Phase-One MVP

Forge phase one includes:

- CLI ingest for local paths and optional S3 prefixes.
- Built-in transforms: exact deduplication, language detection, and perplexity quality scoring.
- Immutable dataset snapshots with version catalog and lineage links.
- Python SDK for ingest, version listing, metadata filtering, and optional S3 export.
- PyTorch DataLoader integration with tokenization and shuffling helpers.
- PyTorch training command with default loop, optional custom loop, and loss curves.
- Shared YAML run-spec execution for CLI and Python SDK workflows.
- Training lifecycle registry + lineage graph + artifact contract metadata for each run.
- Hardware capability profiling command with recommended precision and batch defaults.

## Quickstart

```bash
python3 -m pip install -e .
forge ingest tests/fixtures/raw_valid --dataset demo
forge versions --dataset demo
forge filter --dataset demo --language en --min-quality 0.2
forge train --dataset demo --output-dir ./outputs/train/demo
forge hardware-profile
```

## One-Command Smoke Test

```bash
scripts/run_phase1_smoke_test.sh
```

Run this for full validation (tests + typecheck + lint):

```bash
scripts/run_phase1_smoke_test.sh --full
```

PyTorch training smoke test:

```bash
scripts/run_training_smoke_test.sh
```

Platform verification command:

```bash
forge verify --mode quick
forge verify --mode full --keep-artifacts
```

Verification coverage and manual test steps live in `docs/verification_matrix.md`.

Release merge gate:

```bash
scripts/run_release_gate.sh
```

## Optional Dependencies

- S3 ingest/export: `pip install -e .[s3]`
- Structured logging: `pip install -e .[logging]`
- Lance conversion: `pip install -e .[lance]`
- PyTorch serving: `pip install -e .[serve]`

## Training

Run default training loop:

```bash
forge train --dataset demo --output-dir ./outputs/train/demo
```

Run with configurable default architecture:

```bash
forge train \
  --dataset demo \
  --output-dir ./outputs/train/demo \
  --hidden-dim 256 \
  --num-layers 4 \
  --attention-heads 8 \
  --mlp-hidden-dim 1024 \
  --mlp-layers 2 \
  --dropout 0.1 \
  --position-embedding-type learned \
  --vocabulary-size 20000
```

Use custom architecture file:

```bash
forge train \
  --dataset demo \
  --output-dir ./outputs/train/demo \
  --architecture-file ./architectures/my_model.py
```

Use custom training loop:

```bash
forge train \
  --dataset demo \
  --output-dir ./outputs/train/demo \
  --custom-loop-file ./loops/my_training_loop.py
```

Use optional lifecycle hooks (run/epoch/batch/checkpoint/custom loss):

```bash
forge train \
  --dataset demo \
  --output-dir ./outputs/train/demo \
  --hooks-file ./hooks/my_training_hooks.py
```

Fine-tune from existing model weights:

```bash
forge train \
  --dataset demo \
  --output-dir ./outputs/train/demo-finetune \
  --initial-weights-path ./outputs/train/demo/model.pt
```

Run a chat inference check against trained weights:

```bash
forge chat \
  --dataset demo \
  --model-path ./outputs/train/demo/model.pt \
  --position-embedding-type learned \
  --prompt "hello"
```

Run a declarative pipeline spec:

```bash
forge run-spec ./pipeline.yaml
```

Artifacts written to output dir:

- `model.pt` (trained model weights)
- `history.json` (batch-level training loss + epoch train/validation loss)
- `training_curves.png` (loss graph; requires matplotlib)
- `training_config.json` (training architecture/config used for reproducible inference)
- `tokenizer_vocab.json` (fitted tokenizer vocabulary reused by `forge chat`)
- `training_artifacts_manifest.json` (artifact contract with stable paths and run metadata)
- `reproducibility_bundle.json` (config hash + environment snapshot for replay)

Lifecycle and lineage files are stored under your data-root:

- `.forge/runs/index.json` and `.forge/runs/<run_id>/lifecycle.json`
- `.forge/lineage/model_lineage.json`
