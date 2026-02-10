# Forge Phase-One MVP

Forge phase one includes:

- CLI ingest for local paths and optional S3 prefixes.
- Built-in transforms: exact deduplication, language detection, and perplexity quality scoring.
- Immutable dataset snapshots with version catalog and lineage links.
- Python SDK for ingest, version listing, metadata filtering, and optional S3 export.
- PyTorch DataLoader integration with tokenization and shuffling helpers.
- PyTorch training command with default loop, optional custom loop, and loss curves.

## Quickstart

```bash
python3 -m pip install -e .
forge ingest tests/fixtures/raw_valid --dataset demo
forge versions --dataset demo
forge filter --dataset demo --language en --min-quality 0.2
forge train --dataset demo --output-dir ./runs/demo
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

## Optional Dependencies

- S3 ingest/export: `pip install -e .[s3]`
- Structured logging: `pip install -e .[logging]`
- Lance conversion: `pip install -e .[lance]`
- PyTorch serving: `pip install -e .[serve]`

## Training

Run default training loop:

```bash
forge train --dataset demo --output-dir ./runs/demo
```

Use custom architecture file:

```bash
forge train \
  --dataset demo \
  --output-dir ./runs/demo \
  --architecture-file ./architectures/my_model.py
```

Use custom training loop:

```bash
forge train \
  --dataset demo \
  --output-dir ./runs/demo \
  --custom-loop-file ./loops/my_training_loop.py
```

Artifacts written to output dir:

- `model.pt` (trained model weights)
- `history.json` (epoch train/validation loss)
- `training_curves.png` (loss graph; requires matplotlib)
