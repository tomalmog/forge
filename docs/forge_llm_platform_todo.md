# Forge LLM Platform TODO (Expanded)

## Purpose
Convert Alejandro's jot notes into an implementation-ready backlog for Forge as an end-to-end LLM training, evaluation, and deployment platform.

This is a planning document only. No code is implied by this file.

## Status Snapshot (2026-02-13)
- Completed now:
  - `F-000` Foundation baseline for reproducible run orchestration.
  - `F-001` Unified run spec execution in CLI + SDK (single shared engine).
  - `F-002` Training lifecycle state machine persistence (`queued/running/checkpointing/completed/failed`).
  - `F-003` Artifact contract + reproducibility bundle persisted for each run.
  - `F-004` Extension/hooks interface (`--hooks-file`, run/epoch/batch/checkpoint hooks, custom loss hook).
  - `F-005` Hardware profiler + capability detector (`forge hardware-profile`, run-spec support).
  - `F-006` Dataset/model lineage graph updates for inputs and produced model artifacts.
  - `S-007` Training checkpoints (periodic save, best checkpoint, resume, retention).
  - `T-007` Optimizer/scheduler customization in CLI, YAML run-spec, and runtime.
  - `S-004` Mixed precision runtime (`auto`, `fp32`, `fp16`, `bf16`) with safe fallback.
  - `Q-001` Verification matrix workflow (`forge verify` + `docs/verification_matrix.md` + release gate script).
  - `U-001` Studio runtime insights views (training lifecycle runs, lineage summary, hardware profile panel).
- In progress now:
  - `S-009` Runtime telemetry (basic progress + ETA done; hardware/utilization metrics still open).
- Next implementation queue:
  1. `S-006` Memory-aware batching (dynamic micro-batch + accumulation).
  2. `S-005` Gradient checkpointing by module/stage.
  3. `S-008` Hardware auto-config profiles (`A100`, `H100`, `L40`) wired into train defaults.

## Priority Order
1. `P0`: Core training methods + reproducibility.
2. `P0`: GPU scale/performance (single node, then multi-node).
3. `P1`: Experimentation (sweeps/configs/benchmarks/comparisons).
4. `P1`: Alignment/safety, model versioning, collaboration workflows.
5. `P2`: Deployment workflows.
6. `P3`: TPU support.

## Foundation (Required Before Most Features)
- [x] `F-000` Foundation baseline for reproducible run orchestration.
- [x] `F-001` Unified run spec used by UI, CLI, and Python SDK.
  - Include model architecture, optimizer, trainer mode, data config, distributed config, eval config.
- [x] `F-002` Training lifecycle state machine and persistence.
  - Standard states: `queued`, `running`, `checkpointing`, `completed`, `failed`, `cancelled`.
- [x] `F-003` Artifact contract for all runs.
  - Standard outputs: checkpoints, final weights, tokenizer, metrics, logs, plots, bench results, config snapshot.
- [x] `F-004` Extension/hooks interface.
  - Support user-injected code at safe extension points: transforms, callbacks, custom losses, custom loop hooks.
- [x] `F-005` Hardware profiler + capability detector.
  - Detect topology/memory and produce recommended defaults.
- [x] `F-006` Dataset/model lineage graph.
  - Every run links exact dataset version, parent model, and config hash.

## Training Methods
- [ ] `T-001` SFT (Supervised Fine-Tuning).
  - Prompt/response training with masking, packing, split handling, reproducible outputs.
- [ ] `T-002` DPO (Direct Preference Optimization).
  - Preference datasets (`chosen`/`rejected`), configurable beta, reference model handling.
- [ ] `T-003` RLHF/RLAIF framework.
  - Policy optimization loop with reward model or AI feedback source.
- [ ] `T-004` Distillation.
  - Teacher-student training, logits distillation, optional hard-label blending.
- [ ] `T-005` Domain adaptation.
  - Continued pretraining/domain SFT with drift checks and forgetting safeguards.
- [ ] `T-006` LoRA/PEFT family.
  - LoRA first, then QLoRA and additional PEFT variants; adapter train/export/merge lifecycle.
- [x] `T-007` Optimizer/scheduler customization.
  - Expose optimizer choices and scheduler/warmup settings in UI/CLI/YAML.

## Scale, Performance, Reliability
- [ ] `S-001` Multi-GPU single-node training (DDP baseline).
- [ ] `S-002` Multi-node training with resilient rendezvous/restart behavior.
- [ ] `S-003` FSDP and DeepSpeed support.
- [x] `S-004` Mixed precision (`bf16`, `fp16`, `fp32`) with safe fallback.
- [ ] `S-005` Gradient checkpointing by module/stage.
- [ ] `S-006` Memory-aware batching.
  - Dynamic micro-batch sizing + grad accumulation based on available memory.
- [x] `S-007` Training checkpoints.
  - Periodic + best-model save, resume-from-checkpoint, retention policy.
- [ ] `S-008` Hardware auto-config profiles.
  - Initial target presets: `A100`, `H100`, `L40`.
- [ ] `S-009` Runtime telemetry. `IN PROGRESS`
  - Tokens/sec, step time, GPU utilization, memory, dataloader stalls.
- [ ] `S-010` TPU support (later phase).

## Compute Connectivity (Cloud and On-Prem)
- [ ] `C-001` Compute target abstraction (local/on-prem/cloud).
- [ ] `C-002` Secure connection profiles and secret-safe handling.
- [ ] `C-003` Remote job lifecycle control (submit/monitor/cancel/resume/artifact fetch).
- [ ] `C-004` Data locality strategy (staging/caching for large corpora).
- [ ] `C-005` Runtime and cost estimate before launch for cloud targets.

## Experimentation and Research Workflow
- [ ] `E-001` Hyperparameter sweeps.
  - Grid/random/Bayesian, parallel trials, top-k summary.
- [ ] `E-002` YAML parity with UI/CLI. `IN PROGRESS`
  - Any UI run should export to YAML; CLI/Python should run same spec without drift.
- [ ] `E-003` Code injection for advanced users.
  - Hook points for custom trainer/eval/loss/schedulers.
- [ ] `E-004` Built-in benchmark suite.
  - Perplexity, instruction quality, domain metrics, latency/throughput.
- [ ] `E-005` Model/run comparisons.
  - Side-by-side charts for quality/speed/cost/safety metrics.
- [ ] `E-006` Reproducibility bundles.
  - Export config, seed, environment, artifact manifest for replay.

## Alignment and Safety
- [ ] `A-001` Safety evaluation integration.
  - Toxicity/jailbreak/harmful request response scoring.
- [ ] `A-002` Alignment toolkit.
  - Preference datasets, reward model management, alignment report.
- [ ] `A-003` Pre-deployment safety gates.
  - Required thresholds before packaging/deployment.
- [ ] `A-004` Red-team regression harness.
  - Prompt suites + failure trend alerts.

## Model Versioning ("Git for Models")
- [ ] `V-001` Model registry with lineage DAG.
- [ ] `V-002` Immutable artifact IDs + semantic model version labels.
- [ ] `V-003` Branch/merge style experiment workflows.
- [ ] `V-004` Model diff tooling.
  - Compare config/tokenizer/bench/safety deltas.
- [ ] `V-005` Rollback and pinning of known-good models.

## Collaboration
- [ ] `L-001` Shared dashboards.
- [ ] `L-002` Comments/annotations on runs, checkpoints, charts.
- [ ] `L-003` Shared dataset workspace with access controls.
- [ ] `L-004` Notifications for run completion/failure/regressions.

## Deployment Tooling
- [ ] `D-001` Quantization pipeline and quality-vs-latency reporting.
- [ ] `D-002` Latency profiling matrix (batch/sequence/device).
- [ ] `D-003` A/B integration artifacts for external serving systems.
- [ ] `D-004` One-click packaging for deployment handoff.
  - Include config/tokenizer/weights/safety report/checksums.
- [ ] `D-005` Deployment readiness checklist and blocking gates.

## Suggested Milestones
1. `M1 (P0)`: SFT + LoRA/QLoRA + checkpoints + mixed precision + memory-aware batching + YAML parity.
2. `M2 (P0/P1)`: DPO + distillation + domain adaptation + multi-GPU + gradient checkpointing + hardware auto-config.
3. `M3 (P1)`: Multi-node + FSDP/DeepSpeed + sweeps + benchmark suite + registry lineage.
4. `M4 (P1/P2)`: Alignment/safety suite + collaboration + deployment pipeline.
5. `M5 (P3)`: TPU support.

## Key Risks and Early Mitigations
- [ ] `R-001` RLHF scope blow-up.
  - Mitigation: ship SFT + DPO first; stage RLHF/RLAIF later.
- [ ] `R-002` Distributed training complexity.
  - Mitigation: lock one stable backend path before supporting multiple advanced backends.
- [ ] `R-003` UI/CLI/YAML config drift.
  - Mitigation: single schema + strict validation + versioned spec.
- [ ] `R-004` Reproducibility gaps.
  - Mitigation: immutable artifacts + seed/environment capture as required metadata.
- [ ] `R-005` Safety/performance tradeoffs.
  - Mitigation: mandatory benchmark + safety gates before deploy.
