export interface VersionSummary {
  version_id: string;
  record_count: number;
  created_at: string;
  parent_version: string | null;
}

export interface SourceCount {
  source: string;
  count: number;
}

export interface DatasetDashboard {
  dataset_name: string;
  version_id: string;
  record_count: number;
  average_quality: number;
  min_quality: number;
  max_quality: number;
  language_counts: Record<string, number>;
  source_counts: SourceCount[];
}

export interface RecordSample {
  record_id: string;
  source_uri: string;
  language: string;
  quality_score: number;
  text: string;
}

export interface VersionDiff {
  dataset_name: string;
  base_version: string;
  target_version: string;
  added_records: number;
  removed_records: number;
  shared_records: number;
}

export interface CommandTaskStart {
  task_id: string;
  estimated_total_seconds: number;
}

export interface CommandTaskStatus {
  task_id: string;
  status: "running" | "completed" | "failed";
  command: string;
  args: string[];
  exit_code: number | null;
  stdout: string;
  stderr: string;
  elapsed_seconds: number;
  estimated_total_seconds: number;
  remaining_seconds: number;
  progress_percent: number;
}

export interface TrainingEpoch {
  epoch: number;
  train_loss: number;
  validation_loss: number;
}

export interface TrainingBatchLoss {
  epoch: number;
  batch_index: number;
  global_step: number;
  train_loss: number;
}

export interface TrainingHistory {
  epochs: TrainingEpoch[];
  batch_losses: TrainingBatchLoss[];
}

export type PipelineNodeType =
  | "ingest"
  | "filter"
  | "train"
  | "export"
  | "chat"
  | "custom";

export interface PipelineNode {
  id: string;
  type: PipelineNodeType;
  title: string;
  canvas_x: number;
  canvas_y: number;
  config: Record<string, string>;
}

export interface PipelineEdge {
  id: string;
  source_node_id: string;
  target_node_id: string;
}

export interface PipelineCanvasExportResult {
  output_path: string;
}
