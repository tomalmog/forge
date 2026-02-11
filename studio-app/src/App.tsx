import { useEffect, useState } from "react";
import {
  getDatasetDashboard,
  listDatasets,
  listVersions,
  loadTrainingHistory,
  sampleRecords,
  versionDiff,
} from "./api/studioApi";
import { DatasetSidebar } from "./components/DatasetSidebar";
import { ViewControlDrawer } from "./components/ViewControlDrawer";
import { WorkspacePanels } from "./components/WorkspacePanels";
import { buildDefaultNode } from "./pipeline";
import {
  DEFAULT_PIPELINE_PROGRESS_SNAPSHOT,
  runPipelineInBackground,
} from "./pipeline_run";
import { loadSessionState, saveSessionState } from "./session_state";
import {
  DatasetDashboard,
  PipelineNode,
  PipelineNodeType,
  RecordSample,
  TrainingHistory,
  VersionDiff,
  VersionSummary,
} from "./types";
import { PanelVisibility } from "./view_controls";
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
  const [samples, setSamples] = useState<RecordSample[]>([]);
  const [diff, setDiff] = useState<VersionDiff | null>(null);
  const [baseVersion, setBaseVersion] = useState<string | null>(
    INITIAL_SESSION.base_version,
  );
  const [targetVersion, setTargetVersion] = useState<string | null>(
    INITIAL_SESSION.target_version,
  );
  const [nodes, setNodes] = useState<PipelineNode[]>(INITIAL_SESSION.nodes);
  const [consoleOutput, setConsoleOutput] = useState(
    INITIAL_SESSION.console_output,
  );
  const [historyPath, setHistoryPath] = useState(INITIAL_SESSION.history_path);
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [isViewControlsOpen, setIsViewControlsOpen] = useState(
    INITIAL_SESSION.is_view_controls_open,
  );
  const [panelVisibility, setPanelVisibility] = useState<PanelVisibility>(
    INITIAL_SESSION.panel_visibility,
  );
  const [pipelineProgress, setPipelineProgress] = useState(
    DEFAULT_PIPELINE_PROGRESS_SNAPSHOT,
  );

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
    saveSessionState({
      data_root: dataRoot,
      selected_dataset: selectedDataset,
      selected_version: selectedVersion,
      base_version: baseVersion,
      target_version: targetVersion,
      nodes,
      console_output: consoleOutput,
      history_path: historyPath,
      is_view_controls_open: isViewControlsOpen,
      panel_visibility: panelVisibility,
    });
  }, [
    dataRoot,
    selectedDataset,
    selectedVersion,
    baseVersion,
    targetVersion,
    nodes,
    consoleOutput,
    historyPath,
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

  function addNode(type: PipelineNodeType) {
    setNodes((current) => [...current, buildDefaultNode(type)]);
  }

  function removeNode(nodeId: string) {
    setNodes((current) => current.filter((node) => node.id !== nodeId));
  }

  function updateNode(nodeId: string, key: string, value: string) {
    setNodes((current) =>
      current.map((node) =>
        node.id === nodeId
          ? { ...node, config: { ...node.config, [key]: value } }
          : node,
      ),
    );
  }

  async function runPipeline() {
    if (nodes.length === 0) {
      setConsoleOutput("Add at least one pipeline node before running.");
      return;
    }
    setPipelineProgress({
      ...DEFAULT_PIPELINE_PROGRESS_SNAPSHOT,
      is_running: true,
      current_step_label: "Preparing pipeline",
    });
    setConsoleOutput("Pipeline started...");
    try {
      const result = await runPipelineInBackground({
        data_root: dataRoot,
        nodes,
        on_progress: setPipelineProgress,
      });
      setConsoleOutput(result.console_output);
      if (result.history_path) {
        setHistoryPath(result.history_path);
      }
      if (selectedDataset) {
        await refreshDatasetDetails(selectedDataset, selectedVersion);
      }
    } catch (error) {
      setPipelineProgress((current) => ({ ...current, is_running: false }));
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      setConsoleOutput((current) =>
        `${current}\nPipeline failed: ${errorMessage}`.trim(),
      );
      throw error;
    }
  }

  async function loadHistoryFromPath() {
    if (!historyPath.trim()) {
      return;
    }
    const row = await loadTrainingHistory(historyPath.trim());
    setHistory(row);
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
        onDatasetSelect={(dataset) => {
          setSelectedDataset(dataset);
          setSelectedVersion(null);
          setBaseVersion(null);
          setTargetVersion(null);
          setDiff(null);
        }}
        onVersionSelect={setSelectedVersion}
      />
      <WorkspacePanels
        panelVisibility={panelVisibility}
        dashboard={dashboard}
        versions={versions}
        baseVersion={baseVersion}
        targetVersion={targetVersion}
        diff={diff}
        onBaseVersionChange={setBaseVersion}
        onTargetVersionChange={setTargetVersion}
        onComputeDiff={() => computeVersionDiff().catch(logUiError)}
        samples={samples}
        nodes={nodes}
        isPipelineRunning={pipelineProgress.is_running}
        overallProgressPercent={pipelineProgress.overall_percent}
        pipelineElapsedSeconds={pipelineProgress.pipeline_elapsed_seconds}
        pipelineRemainingSeconds={pipelineProgress.pipeline_remaining_seconds}
        currentStepLabel={pipelineProgress.current_step_label}
        currentStepProgressPercent={pipelineProgress.current_step_percent}
        currentStepElapsedSeconds={
          pipelineProgress.current_step_elapsed_seconds
        }
        currentStepRemainingSeconds={
          pipelineProgress.current_step_remaining_seconds
        }
        onAddNode={addNode}
        onRemoveNode={removeNode}
        onUpdateNode={updateNode}
        onRunPipeline={() => runPipeline().catch(logUiError)}
        historyPath={historyPath}
        history={history}
        onHistoryPathChange={setHistoryPath}
        onLoadHistory={() => loadHistoryFromPath().catch(logUiError)}
        consoleOutput={consoleOutput}
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

function logUiError(error: unknown) {
  console.error(error);
}

export default App;
