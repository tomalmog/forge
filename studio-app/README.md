# Forge Studio (Phase 2)

Desktop UI for Forge, built with Tauri + React.

## What It Does Today

- Browse datasets and versions under a chosen Forge `data_root`.
- View dataset dashboard metrics (record count, quality, language/source distributions).
- Inspect record samples from a selected dataset version.
- Compare two versions (added/removed/shared records).
- Build a visual pipeline by dragging nodes:
  - `ingest`
  - `filter`
  - `train`
  - `export-training`
  - `custom` (raw Forge CLI args)
- Run pipeline steps from the UI through whitelisted Forge commands.
- Execute ingest/filter/train/export in background worker tasks so the UI stays responsive.
- Track pipeline and step-level progress with elapsed time and ETA.
- Auto-save Studio session state (selected dataset, pipeline nodes, console output, history path).
- Load and visualize training history (`history.json`) with step-level and epoch-level loss curves.

## Quick Start

1. Install project dependencies:
```bash
python3 -m pip install -e '.[serve]'
npm --prefix studio-app install
```
2. Launch the desktop app:
```bash
cd studio-app
npm run tauri dev
```

## One-Command Smoke Test

From repo root:

```bash
scripts/run_phase2_studio_smoke_test.sh
```

This script:

- Ingests fixture data.
- Creates a filtered version.
- Runs a short PyTorch training pass.
- Exports training shards.
- Builds Studio (`tauri build --debug --no-bundle`).
- Prints values for `data_root`, `dataset`, and `history_path` to paste into the UI.

## Using `first_test_full`

If you already have data in `.forge-firsttest-phase1-full`:

1. Set `data_root` in the sidebar to `.forge-firsttest-phase1-full` and click refresh.
2. Select dataset `first_test_full`.
3. Open the pipeline panel and add/edit nodes.
4. Run a `train` node with an output directory like `.forge-firsttest-train-runs/first_test_full`.
5. Paste the generated `history.json` path into the training curves panel to plot metrics.
