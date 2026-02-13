import { useEffect, useState } from "react";
import {
  getDatasetDashboard,
  listDatasets,
  listVersions,
  sampleRecords,
  versionDiff,
} from "./api/studioApi";
import { DatasetSidebar } from "./components/DatasetSidebar";
import { ViewControlDrawer } from "./components/ViewControlDrawer";
import { WorkspacePanels } from "./components/WorkspacePanels";
import { usePipelineGraphState } from "./hooks/use_pipeline_graph_state";
import { loadSessionState, saveSessionState } from "./session_state";
import {
  DatasetDashboard,
  LineageGraphSummary,
  RecordSample,
  TrainingRunSummary,
  VersionDiff,
  VersionSummary,
} from "./types";
import { PanelVisibility } from "./view_controls";
import { loadRuntimeInsights } from "./runtime_insights";
import "./App.css";

const INITIAL_SESSION = loadSessionState();
function App() {
  const [dataRoot, setDataRoot] = useState(INITIAL_SESSION.data_root);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(
    INITIAL_SESSION.selected_dataset,
  );
  const [versions, setVersions] = useState<VersionSummary[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<string | null>(
    INITIAL_SESSION.selected_version,
  );
  const [dashboard, setDashboard] = useState<DatasetDashboard | null>(null);
  const [diff, setDiff] = useState<VersionDiff | null>(null);
  const [baseVersion, setBaseVersion] = useState<string | null>(
    INITIAL_SESSION.base_version,
  );
  const [targetVersion, setTargetVersion] = useState<string | null>(
    INITIAL_SESSION.target_version,
  );
  const [samples, setSamples] = useState<RecordSample[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRunSummary[]>([]);
  const [lineageGraph, setLineageGraph] = useState<LineageGraphSummary | null>(
    null,
  );
  const [hardwareProfile, setHardwareProfile] = useState<Record<
    string,
    string
  > | null>(null);
  const [isViewControlsOpen, setIsViewControlsOpen] = useState(
    INITIAL_SESSION.is_view_controls_open,
  );
  const [panelVisibility, setPanelVisibility] = useState<PanelVisibility>(
    INITIAL_SESSION.panel_visibility,
  );
  const [lastCanvasExportDir, setLastCanvasExportDir] = useState(
    INITIAL_SESSION.last_canvas_export_dir,
  );

  const pipeline = usePipelineGraphState({
    data_root: dataRoot,
    initial_state: {
      nodes: INITIAL_SESSION.nodes,
      edges: INITIAL_SESSION.edges,
      start_node_id: INITIAL_SESSION.start_node_id,
      selected_node_id: INITIAL_SESSION.selected_node_id,
      console_output: INITIAL_SESSION.console_output,
      history_path: INITIAL_SESSION.history_path,
    },
    on_pipeline_complete: async () => {
      if (selectedDataset) {
        await refreshDatasetDetails(selectedDataset, selectedVersion);
      }
      await refreshRuntimeInsights();
    },
  });

  useEffect(() => {
    refreshDatasets().catch(logUiError);
  }, []);

  useEffect(() => {
    if (!selectedDataset) {
      return;
    }
    refreshDatasetDetails(selectedDataset, selectedVersion).catch(logUiError);
  }, [selectedDataset, selectedVersion]);

  useEffect(() => {
    refreshRuntimeInsights().catch(logUiError);
  }, [dataRoot]);

  useEffect(() => {
    saveSessionState({
      data_root: dataRoot,
      selected_dataset: selectedDataset,
      selected_version: selectedVersion,
      base_version: baseVersion,
      target_version: targetVersion,
      nodes: pipeline.nodes,
      edges: pipeline.edges,
      start_node_id: pipeline.start_node_id,
      selected_node_id: pipeline.selected_node_id,
      console_output: pipeline.console_output,
      history_path: pipeline.history_path,
      last_canvas_export_dir: lastCanvasExportDir,
      is_view_controls_open: isViewControlsOpen,
      panel_visibility: panelVisibility,
    });
  }, [
    dataRoot,
    selectedDataset,
    selectedVersion,
    baseVersion,
    targetVersion,
    pipeline.nodes,
    pipeline.edges,
    pipeline.start_node_id,
    pipeline.selected_node_id,
    pipeline.console_output,
    pipeline.history_path,
    lastCanvasExportDir,
    isViewControlsOpen,
    panelVisibility,
  ]);

  async function refreshDatasets() {
    const rows = await listDatasets(dataRoot);
    setDatasets(rows);
    if (rows.length === 0) {
      setSelectedDataset(null);
      setVersions([]);
      setDashboard(null);
      setSamples([]);
      setBaseVersion(null);
      setTargetVersion(null);
      return;
    }
    if (!selectedDataset || !rows.includes(selectedDataset)) {
      setSelectedDataset(rows[0]);
      setSelectedVersion(null);
    }
  }

  async function refreshDatasetDetails(
    datasetName: string,
    versionId: string | null,
  ) {
    const versionRows = await listVersions(dataRoot, datasetName);
    setVersions(versionRows);
    if (versionRows.length === 0) {
      setDashboard(null);
      setSamples([]);
      setBaseVersion(null);
      setTargetVersion(null);
      return;
    }
    const dashboardRow = await getDatasetDashboard(
      dataRoot,
      datasetName,
      versionId,
    );
    const sampleRows = await sampleRecords(
      dataRoot,
      datasetName,
      versionId,
      0,
      12,
    );
    setDashboard(dashboardRow);
    setSamples(sampleRows);
    const versionIds = new Set(versionRows.map((row) => row.version_id));
    setBaseVersion((current) =>
      current && versionIds.has(current) ? current : versionRows[0].version_id,
    );
    setTargetVersion((current) =>
      current && versionIds.has(current)
        ? current
        : versionRows[versionRows.length - 1].version_id,
    );
  }

  async function computeVersionDiff() {
    if (!selectedDataset || !baseVersion || !targetVersion) {
      return;
    }
    const result = await versionDiff(
      dataRoot,
      selectedDataset,
      baseVersion,
      targetVersion,
    );
    setDiff(result);
  }

  async function refreshRuntimeInsights() {
    const snapshot = await loadRuntimeInsights(dataRoot);
    setTrainingRuns(snapshot.runs);
    setLineageGraph(snapshot.lineage);
    setHardwareProfile(snapshot.hardwareProfile);
  }

  function onDatasetSelect(datasetName: string) {
    setSelectedDataset(datasetName);
    setSelectedVersion(null);
    setBaseVersion(null);
    setTargetVersion(null);
    setDiff(null);
  }

  function togglePanelVisibility(key: keyof PanelVisibility) {
    setPanelVisibility((current) => ({ ...current, [key]: !current[key] }));
  }

  return (
    <main
      className={`app-root ${isViewControlsOpen ? "view-controls-open" : "view-controls-collapsed"}`}
    >
      <DatasetSidebar
        dataRoot={dataRoot}
        datasets={datasets}
        selectedDataset={selectedDataset}
        versions={versions}
        selectedVersion={selectedVersion}
        onDataRootChange={setDataRoot}
        onRefresh={() => refreshDatasets().catch(logUiError)}
        onDatasetSelect={onDatasetSelect}
        onVersionSelect={setSelectedVersion}
      />
      <WorkspacePanels
        panelVisibility={panelVisibility}
        dataRoot={dataRoot}
        selectedDataset={selectedDataset}
        dashboard={dashboard}
        versions={versions}
        baseVersion={baseVersion}
        targetVersion={targetVersion}
        diff={diff}
        onBaseVersionChange={setBaseVersion}
        onTargetVersionChange={setTargetVersion}
        onComputeDiff={() => computeVersionDiff().catch(logUiError)}
        samples={samples}
        nodes={pipeline.nodes}
        edges={pipeline.edges}
        startNodeId={pipeline.start_node_id}
        selectedNodeId={pipeline.selected_node_id}
        isPipelineRunning={pipeline.progress.is_running}
        overallProgressPercent={pipeline.progress.overall_percent}
        pipelineElapsedSeconds={pipeline.progress.pipeline_elapsed_seconds}
        pipelineRemainingSeconds={pipeline.progress.pipeline_remaining_seconds}
        currentStepLabel={pipeline.progress.current_step_label}
        currentStepProgressPercent={pipeline.progress.current_step_percent}
        currentStepElapsedSeconds={
          pipeline.progress.current_step_elapsed_seconds
        }
        currentStepRemainingSeconds={
          pipeline.progress.current_step_remaining_seconds
        }
        onAddNode={pipeline.add_node}
        onMoveNode={pipeline.move_node}
        onSelectNode={pipeline.set_selected_node_id}
        onSetStartNode={pipeline.set_start_node_id}
        onAddEdge={pipeline.add_edge}
        onRemoveEdge={pipeline.remove_edge}
        onRemoveNode={pipeline.remove_node}
        onClearCanvas={pipeline.clear_canvas}
        onUpdateNode={pipeline.update_node}
        onRunPipeline={() => pipeline.run_pipeline().catch(logUiError)}
        lastCanvasExportDir={lastCanvasExportDir}
        onLastCanvasExportDirChange={setLastCanvasExportDir}
        historyPath={pipeline.history_path}
        history={pipeline.history}
        onHistoryPathChange={pipeline.set_history_path}
        onLoadHistory={() => pipeline.load_history().catch(logUiError)}
        consoleOutput={pipeline.console_output}
        hardwareProfile={hardwareProfile}
        trainingRuns={trainingRuns}
        lineage={lineageGraph}
        onRefreshRuntimeInsights={() =>
          refreshRuntimeInsights().catch(logUiError)
        }
      />
      <ViewControlDrawer
        isOpen={isViewControlsOpen}
        visibility={panelVisibility}
        onToggleOpen={() => setIsViewControlsOpen((current) => !current)}
        onTogglePanel={togglePanelVisibility}
      />
    </main>
  );
}

const logUiError = (error: unknown): void => {
  console.error(error);
};
export default App;
