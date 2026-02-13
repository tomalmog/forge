import { DashboardView } from "./DashboardView";
import { ChatRoomView } from "./ChatRoomView";
import { PipelineCanvas } from "./PipelineCanvas";
import { SampleInspector } from "./SampleInspector";
import { StatusConsole } from "./StatusConsole";
import { TrainingCurvesView } from "./TrainingCurvesView";
import { VersionDiffView } from "./VersionDiffView";
import { RuntimeInsightsView } from "./RuntimeInsightsView";
import {
  DatasetDashboard,
  LineageGraphSummary,
  PipelineEdge,
  PipelineNode,
  PipelineNodeType,
  RecordSample,
  TrainingRunSummary,
  TrainingHistory,
  VersionDiff,
  VersionSummary,
} from "../types";
import { anyDashboardSectionVisible, PanelVisibility } from "../view_controls";

interface WorkspacePanelsProps {
  panelVisibility: PanelVisibility;
  dataRoot: string;
  selectedDataset: string | null;
  dashboard: DatasetDashboard | null;
  versions: VersionSummary[];
  baseVersion: string | null;
  targetVersion: string | null;
  diff: VersionDiff | null;
  onBaseVersionChange: (value: string) => void;
  onTargetVersionChange: (value: string) => void;
  onComputeDiff: () => void;
  samples: RecordSample[];
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  startNodeId: string | null;
  selectedNodeId: string | null;
  isPipelineRunning: boolean;
  overallProgressPercent: number;
  pipelineElapsedSeconds: number;
  pipelineRemainingSeconds: number;
  currentStepLabel: string;
  currentStepProgressPercent: number;
  currentStepElapsedSeconds: number;
  currentStepRemainingSeconds: number;
  onAddNode: (type: PipelineNodeType, x?: number, y?: number) => void;
  onMoveNode: (nodeId: string, x: number, y: number) => void;
  onSelectNode: (nodeId: string | null) => void;
  onSetStartNode: (nodeId: string | null) => void;
  onAddEdge: (sourceNodeId: string, targetNodeId: string) => void;
  onRemoveEdge: (edgeId: string) => void;
  onRemoveNode: (nodeId: string) => void;
  onClearCanvas: () => void;
  onUpdateNode: (nodeId: string, key: string, value: string) => void;
  onRunPipeline: () => void;
  lastCanvasExportDir: string;
  onLastCanvasExportDirChange: (value: string) => void;
  historyPath: string;
  history: TrainingHistory | null;
  onHistoryPathChange: (value: string) => void;
  onLoadHistory: () => void;
  consoleOutput: string;
  hardwareProfile: Record<string, string> | null;
  trainingRuns: TrainingRunSummary[];
  lineage: LineageGraphSummary | null;
  onRefreshRuntimeInsights: () => void;
}

export function WorkspacePanels(props: WorkspacePanelsProps) {
  const showDashboard = anyDashboardSectionVisible(props.panelVisibility);
  return (
    <section className="workspace">
      {props.panelVisibility.workspace_header && (
        <header className="workspace-header">
          <h1>Forge Studio Desktop</h1>
          <p>
            Drag pipeline components, train with PyTorch defaults, inspect
            dataset versions, and monitor loss curves.
          </p>
        </header>
      )}
      {showDashboard && (
        <DashboardView
          dashboard={props.dashboard}
          showMetrics={props.panelVisibility.dashboard_metrics}
          showLanguageMix={props.panelVisibility.dashboard_language_mix}
          showTopSources={props.panelVisibility.dashboard_top_sources}
        />
      )}
      {props.panelVisibility.version_diff && (
        <VersionDiffView
          versions={props.versions}
          baseVersion={props.baseVersion}
          targetVersion={props.targetVersion}
          diff={props.diff}
          onBaseVersionChange={props.onBaseVersionChange}
          onTargetVersionChange={props.onTargetVersionChange}
          onComputeDiff={props.onComputeDiff}
        />
      )}
      {props.panelVisibility.sample_inspector && (
        <SampleInspector samples={props.samples} />
      )}
      {props.panelVisibility.pipeline_builder && (
        <PipelineCanvas
          dataRoot={props.dataRoot}
          nodes={props.nodes}
          edges={props.edges}
          startNodeId={props.startNodeId}
          selectedNodeId={props.selectedNodeId}
          isRunning={props.isPipelineRunning}
          overallProgressPercent={props.overallProgressPercent}
          pipelineElapsedSeconds={props.pipelineElapsedSeconds}
          pipelineRemainingSeconds={props.pipelineRemainingSeconds}
          currentStepLabel={props.currentStepLabel}
          currentStepProgressPercent={props.currentStepProgressPercent}
          currentStepElapsedSeconds={props.currentStepElapsedSeconds}
          currentStepRemainingSeconds={props.currentStepRemainingSeconds}
          onAddNode={props.onAddNode}
          onMoveNode={props.onMoveNode}
          onSelectNode={props.onSelectNode}
          onSetStartNode={props.onSetStartNode}
          onAddEdge={props.onAddEdge}
          onRemoveEdge={props.onRemoveEdge}
          onRemoveNode={props.onRemoveNode}
          onClearCanvas={props.onClearCanvas}
          onUpdateNode={props.onUpdateNode}
          onRunPipeline={props.onRunPipeline}
          lastCanvasExportDir={props.lastCanvasExportDir}
          onLastCanvasExportDirChange={props.onLastCanvasExportDirChange}
        />
      )}
      {props.panelVisibility.chat_room && (
        <ChatRoomView
          dataRoot={props.dataRoot}
          selectedDataset={props.selectedDataset}
        />
      )}
      {props.panelVisibility.training_curves && (
        <TrainingCurvesView
          historyPath={props.historyPath}
          history={props.history}
          onHistoryPathChange={props.onHistoryPathChange}
          onLoadHistory={props.onLoadHistory}
        />
      )}
      {props.panelVisibility.runtime_insights && (
        <RuntimeInsightsView
          hardwareProfile={props.hardwareProfile}
          trainingRuns={props.trainingRuns}
          lineage={props.lineage}
          onRefresh={props.onRefreshRuntimeInsights}
        />
      )}
      {props.panelVisibility.run_console && (
        <StatusConsole output={props.consoleOutput} />
      )}
    </section>
  );
}
