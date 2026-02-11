import { getForgeCommandStatus, startForgeCommand } from "./api/studioApi";
import { estimateNodeDurationSeconds, toForgeArgs } from "./pipeline";
import { CommandTaskStatus, PipelineNode } from "./types";

export interface PipelineProgressSnapshot {
  is_running: boolean;
  overall_percent: number;
  pipeline_elapsed_seconds: number;
  pipeline_remaining_seconds: number;
  current_step_label: string;
  current_step_percent: number;
  current_step_elapsed_seconds: number;
  current_step_remaining_seconds: number;
}

export interface PipelineRunResult {
  console_output: string;
  history_path: string | null;
}

interface PipelineRunOptions {
  data_root: string;
  nodes: PipelineNode[];
  on_progress: (snapshot: PipelineProgressSnapshot) => void;
}

const POLL_INTERVAL_MS = 900;

export const DEFAULT_PIPELINE_PROGRESS_SNAPSHOT: PipelineProgressSnapshot = {
  is_running: false,
  overall_percent: 0,
  pipeline_elapsed_seconds: 0,
  pipeline_remaining_seconds: 0,
  current_step_label: "Idle",
  current_step_percent: 0,
  current_step_elapsed_seconds: 0,
  current_step_remaining_seconds: 0,
};

export async function runPipelineInBackground(
  options: PipelineRunOptions,
): Promise<PipelineRunResult> {
  if (options.nodes.length === 0) {
    return { console_output: "", history_path: null };
  }

  const estimated_step_seconds = options.nodes.map((node) =>
    estimateNodeDurationSeconds(node.type),
  );
  const started_at_ms = Date.now();
  let history_path: string | null = null;
  const log_chunks: string[] = [];

  for (let index = 0; index < options.nodes.length; index += 1) {
    const node = options.nodes[index];
    const args = toForgeArgs(node);
    const task_start = await startForgeCommand(options.data_root, args);
    estimated_step_seconds[index] = Math.max(
      estimated_step_seconds[index],
      task_start.estimated_total_seconds,
    );
    const task_status = await wait_for_task({
      task_id: task_start.task_id,
      node_title: node.title,
      node_index: index,
      node_count: options.nodes.length,
      estimated_step_seconds,
      started_at_ms,
      on_progress: options.on_progress,
    });
    append_command_output(log_chunks, options.data_root, args, task_status);
    history_path = parse_history_path(task_status.stdout) ?? history_path;
    if (task_status.status !== "completed" || task_status.exit_code !== 0) {
      throw new Error(`Command failed: ${args.join(" ")}`);
    }
  }

  const elapsed_seconds = seconds_since(started_at_ms);
  options.on_progress({
    is_running: false,
    overall_percent: 100,
    pipeline_elapsed_seconds: elapsed_seconds,
    pipeline_remaining_seconds: 0,
    current_step_label: "Pipeline complete",
    current_step_percent: 100,
    current_step_elapsed_seconds: 0,
    current_step_remaining_seconds: 0,
  });

  return {
    console_output:
      log_chunks.length > 0
        ? `${log_chunks.join("\n")}\n`
        : "Pipeline completed with no output.",
    history_path,
  };
}

interface WaitTaskOptions {
  task_id: string;
  node_title: string;
  node_index: number;
  node_count: number;
  estimated_step_seconds: number[];
  started_at_ms: number;
  on_progress: (snapshot: PipelineProgressSnapshot) => void;
}

async function wait_for_task(
  options: WaitTaskOptions,
): Promise<CommandTaskStatus> {
  while (true) {
    const task_status = await getForgeCommandStatus(options.task_id);
    emit_progress(options, task_status);
    if (task_status.status !== "running") {
      return task_status;
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

function emit_progress(
  options: WaitTaskOptions,
  task_status: CommandTaskStatus,
): void {
  const current_step_percent =
    task_status.status === "running" ? task_status.progress_percent : 100;
  const overall_percent =
    ((options.node_index + current_step_percent / 100) / options.node_count) *
    100;
  const future_remaining_seconds = sum_remaining_estimate(
    options.estimated_step_seconds,
    options.node_index + 1,
  );
  options.on_progress({
    is_running: task_status.status === "running",
    overall_percent: clamp(overall_percent, 0, 100),
    pipeline_elapsed_seconds: seconds_since(options.started_at_ms),
    pipeline_remaining_seconds:
      task_status.remaining_seconds + future_remaining_seconds,
    current_step_label: `${options.node_index + 1}/${options.node_count}: ${options.node_title}`,
    current_step_percent: current_step_percent,
    current_step_elapsed_seconds: task_status.elapsed_seconds,
    current_step_remaining_seconds: task_status.remaining_seconds,
  });
}

function append_command_output(
  log_chunks: string[],
  data_root: string,
  args: string[],
  task_status: CommandTaskStatus,
): void {
  log_chunks.push(`$ forge --data-root ${data_root} ${args.join(" ")}`);
  if (task_status.stdout.trim().length > 0) {
    log_chunks.push(task_status.stdout.trim());
  }
  if (task_status.stderr.trim().length > 0) {
    log_chunks.push(task_status.stderr.trim());
  }
}

function parse_history_path(stdout: string): string | null {
  const line = stdout
    .split("\n")
    .find(
      (row) => row.startsWith("history_path=") || row.endsWith("history.json"),
    );
  if (!line) {
    return null;
  }
  return line.includes("=") ? line.split("=")[1].trim() : line.trim();
}

function sum_remaining_estimate(
  estimates: number[],
  start_index: number,
): number {
  return estimates.slice(start_index).reduce((sum, value) => sum + value, 0);
}

function seconds_since(started_at_ms: number): number {
  return Math.max(0, Math.floor((Date.now() - started_at_ms) / 1000));
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function clamp(value: number, min: number, max: number): number {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}
