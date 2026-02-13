export interface PanelVisibility {
  workspace_header: boolean;
  dashboard_metrics: boolean;
  dashboard_language_mix: boolean;
  dashboard_top_sources: boolean;
  version_diff: boolean;
  sample_inspector: boolean;
  pipeline_builder: boolean;
  chat_room: boolean;
  training_curves: boolean;
  runtime_insights: boolean;
  run_console: boolean;
}

export interface ViewControlItem {
  key: keyof PanelVisibility;
  label: string;
}

export const DEFAULT_PANEL_VISIBILITY: PanelVisibility = {
  workspace_header: true,
  dashboard_metrics: true,
  dashboard_language_mix: true,
  dashboard_top_sources: true,
  version_diff: true,
  sample_inspector: true,
  pipeline_builder: true,
  chat_room: true,
  training_curves: true,
  runtime_insights: true,
  run_console: true,
};

export const VIEW_CONTROL_ITEMS: ViewControlItem[] = [
  { key: "workspace_header", label: "Workspace Header" },
  { key: "dashboard_metrics", label: "Dashboard Metrics" },
  { key: "dashboard_language_mix", label: "Language Mix" },
  { key: "dashboard_top_sources", label: "Top Sources" },
  { key: "version_diff", label: "Version Diff" },
  { key: "sample_inspector", label: "Sample Inspector" },
  { key: "pipeline_builder", label: "Pipeline Builder" },
  { key: "chat_room", label: "Chat Room" },
  { key: "training_curves", label: "Training Curves" },
  { key: "runtime_insights", label: "Runtime Insights" },
  { key: "run_console", label: "Run Console" },
];

export function anyDashboardSectionVisible(
  visibility: PanelVisibility,
): boolean {
  return (
    visibility.dashboard_metrics ||
    visibility.dashboard_language_mix ||
    visibility.dashboard_top_sources
  );
}
