import { PipelineNode } from "./types";
import { DEFAULT_PANEL_VISIBILITY, PanelVisibility } from "./view_controls";

const SESSION_STORAGE_KEY = "forge_studio_session_v1";

export interface StudioSessionState {
  data_root: string;
  selected_dataset: string | null;
  selected_version: string | null;
  base_version: string | null;
  target_version: string | null;
  nodes: PipelineNode[];
  console_output: string;
  history_path: string;
  is_view_controls_open: boolean;
  panel_visibility: PanelVisibility;
}

export const DEFAULT_SESSION_STATE: StudioSessionState = {
  data_root: ".forge",
  selected_dataset: null,
  selected_version: null,
  base_version: null,
  target_version: null,
  nodes: [],
  console_output: "",
  history_path: "",
  is_view_controls_open: true,
  panel_visibility: DEFAULT_PANEL_VISIBILITY,
};

export function loadSessionState(): StudioSessionState {
  if (typeof window === "undefined" || !window.localStorage) {
    return DEFAULT_SESSION_STATE;
  }
  const rawValue = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (!rawValue) {
    return DEFAULT_SESSION_STATE;
  }
  try {
    const parsed = JSON.parse(rawValue) as Partial<StudioSessionState>;
    return {
      data_root: asString(parsed.data_root, DEFAULT_SESSION_STATE.data_root),
      selected_dataset: asNullableString(parsed.selected_dataset),
      selected_version: asNullableString(parsed.selected_version),
      base_version: asNullableString(parsed.base_version),
      target_version: asNullableString(parsed.target_version),
      nodes: Array.isArray(parsed.nodes) ? parsed.nodes : [],
      console_output: asString(parsed.console_output, ""),
      history_path: asString(parsed.history_path, ""),
      is_view_controls_open: asBoolean(parsed.is_view_controls_open, true),
      panel_visibility: parsePanelVisibility(parsed.panel_visibility),
    };
  } catch {
    return DEFAULT_SESSION_STATE;
  }
}

export function saveSessionState(sessionState: StudioSessionState): void {
  if (typeof window === "undefined" || !window.localStorage) {
    return;
  }
  window.localStorage.setItem(
    SESSION_STORAGE_KEY,
    JSON.stringify(sessionState),
  );
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function asNullableString(value: unknown): string | null {
  if (value === null) {
    return null;
  }
  return typeof value === "string" ? value : null;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function parsePanelVisibility(value: unknown): PanelVisibility {
  const defaultVisibility = DEFAULT_PANEL_VISIBILITY;
  if (!value || typeof value !== "object") {
    return defaultVisibility;
  }
  const raw = value as Partial<Record<keyof PanelVisibility, unknown>>;
  return {
    workspace_header: asBoolean(
      raw.workspace_header,
      defaultVisibility.workspace_header,
    ),
    dashboard_metrics: asBoolean(
      raw.dashboard_metrics,
      defaultVisibility.dashboard_metrics,
    ),
    dashboard_language_mix: asBoolean(
      raw.dashboard_language_mix,
      defaultVisibility.dashboard_language_mix,
    ),
    dashboard_top_sources: asBoolean(
      raw.dashboard_top_sources,
      defaultVisibility.dashboard_top_sources,
    ),
    version_diff: asBoolean(raw.version_diff, defaultVisibility.version_diff),
    sample_inspector: asBoolean(
      raw.sample_inspector,
      defaultVisibility.sample_inspector,
    ),
    pipeline_builder: asBoolean(
      raw.pipeline_builder,
      defaultVisibility.pipeline_builder,
    ),
    training_curves: asBoolean(
      raw.training_curves,
      defaultVisibility.training_curves,
    ),
    run_console: asBoolean(raw.run_console, defaultVisibility.run_console),
  };
}
