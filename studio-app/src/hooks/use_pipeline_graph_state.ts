import { useEffect, useState } from "react";
import { loadTrainingHistory } from "../api/studioApi";
import { buildDefaultNode } from "../pipeline";
import { resolveNonOverlappingNodePosition } from "../pipeline_layout";
import {
  buildPipelineExecutionPlan,
  sanitizePipelineGraph,
} from "../pipeline_graph";
import {
  DEFAULT_PIPELINE_PROGRESS_SNAPSHOT,
  PipelineProgressSnapshot,
  runPipelineInBackground,
} from "../pipeline_run";
import {
  PipelineEdge,
  PipelineNode,
  PipelineNodeType,
  TrainingHistory,
} from "../types";

interface PipelineGraphInitialState {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  start_node_id: string | null;
  selected_node_id: string | null;
  console_output: string;
  history_path: string;
}

interface UsePipelineGraphStateOptions {
  data_root: string;
  initial_state: PipelineGraphInitialState;
  on_pipeline_complete: () => Promise<void>;
}

export interface PipelineGraphState {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  start_node_id: string | null;
  selected_node_id: string | null;
  console_output: string;
  history_path: string;
  history: TrainingHistory | null;
  progress: PipelineProgressSnapshot;
  set_selected_node_id: (nodeId: string | null) => void;
  set_start_node_id: (nodeId: string | null) => void;
  set_history_path: (path: string) => void;
  add_node: (type: PipelineNodeType, x?: number, y?: number) => void;
  move_node: (nodeId: string, x: number, y: number) => void;
  remove_node: (nodeId: string) => void;
  update_node: (nodeId: string, key: string, value: string) => void;
  add_edge: (sourceNodeId: string, targetNodeId: string) => void;
  remove_edge: (edgeId: string) => void;
  clear_canvas: () => void;
  run_pipeline: () => Promise<void>;
  load_history: () => Promise<void>;
}

export function usePipelineGraphState(
  options: UsePipelineGraphStateOptions,
): PipelineGraphState {
  const [nodes, setNodes] = useState<PipelineNode[]>(
    options.initial_state.nodes,
  );
  const [edges, setEdges] = useState<PipelineEdge[]>(
    sanitizePipelineGraph(
      options.initial_state.nodes,
      options.initial_state.edges,
    ),
  );
  const [startNodeId, setStartNodeId] = useState<string | null>(
    options.initial_state.start_node_id,
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(
    options.initial_state.selected_node_id,
  );
  const [consoleOutput, setConsoleOutput] = useState(
    options.initial_state.console_output,
  );
  const [historyPath, setHistoryPath] = useState(
    options.initial_state.history_path,
  );
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [progress, setProgress] = useState(DEFAULT_PIPELINE_PROGRESS_SNAPSHOT);

  useEffect(() => {
    setEdges((current) => sanitizePipelineGraph(nodes, current));
    const nodeIdSet = new Set(nodes.map((node) => node.id));
    if (!startNodeId && nodes.length > 0) {
      setStartNodeId(nodes[0].id);
    }
    if (startNodeId && !nodeIdSet.has(startNodeId)) {
      setStartNodeId(null);
    }
    if (selectedNodeId && !nodeIdSet.has(selectedNodeId)) {
      setSelectedNodeId(null);
    }
  }, [nodes, startNodeId, selectedNodeId]);

  function addNode(type: PipelineNodeType, x?: number, y?: number) {
    const nextNode = buildDefaultNode(type, nodes.length);
    const preferredX =
      x !== undefined ? Math.max(0, Math.round(x)) : nextNode.canvas_x;
    const preferredY =
      y !== undefined ? Math.max(0, Math.round(y)) : nextNode.canvas_y;
    const resolvedPosition = resolveNonOverlappingNodePosition(
      nodes,
      preferredX,
      preferredY,
    );
    setNodes((current) => [
      ...current,
      {
        ...nextNode,
        canvas_x: resolvedPosition.x,
        canvas_y: resolvedPosition.y,
      },
    ]);
    setSelectedNodeId(nextNode.id);
    if (!startNodeId) {
      setStartNodeId(nextNode.id);
    }
  }

  function moveNode(nodeId: string, x: number, y: number) {
    setNodes((current) =>
      current.map((node) =>
        node.id === nodeId
          ? { ...node, canvas_x: Math.max(0, x), canvas_y: Math.max(0, y) }
          : node,
      ),
    );
  }

  function removeNode(nodeId: string) {
    setNodes((current) => current.filter((node) => node.id !== nodeId));
    setEdges((current) =>
      current.filter(
        (edge) =>
          edge.source_node_id !== nodeId && edge.target_node_id !== nodeId,
      ),
    );
    if (startNodeId === nodeId) {
      setStartNodeId(null);
    }
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(null);
    }
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

  function addEdge(sourceNodeId: string, targetNodeId: string) {
    if (sourceNodeId === targetNodeId) {
      return;
    }
    const nextEdge: PipelineEdge = {
      id: `edge-${Math.random().toString(16).slice(2, 10)}`,
      source_node_id: sourceNodeId,
      target_node_id: targetNodeId,
    };
    setEdges((current) => sanitizePipelineGraph(nodes, [...current, nextEdge]));
  }

  function removeEdge(edgeId: string) {
    setEdges((current) => current.filter((edge) => edge.id !== edgeId));
  }

  function clearCanvas() {
    setNodes([]);
    setEdges([]);
    setStartNodeId(null);
    setSelectedNodeId(null);
  }

  async function runPipeline() {
    if (nodes.length === 0) {
      setConsoleOutput("Add at least one pipeline node before running.");
      return;
    }
    if (!startNodeId) {
      setConsoleOutput(
        "Choose a start node before running. Select a node and click 'Set As Start'.",
      );
      return;
    }
    setProgress({
      ...DEFAULT_PIPELINE_PROGRESS_SNAPSHOT,
      is_running: true,
      current_step_label: "Preparing pipeline",
    });
    setConsoleOutput("Pipeline started...");
    try {
      const executionPlan = buildPipelineExecutionPlan(
        nodes,
        edges,
        startNodeId,
      );
      const result = await runPipelineInBackground({
        data_root: options.data_root,
        nodes: executionPlan.ordered_nodes,
        on_progress: setProgress,
      });
      setConsoleOutput(result.console_output);
      if (result.history_path) {
        setHistoryPath(result.history_path);
      }
      await options.on_pipeline_complete();
    } catch (error) {
      setProgress((current) => ({ ...current, is_running: false }));
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      setConsoleOutput((current) =>
        `${current}\nPipeline failed: ${errorMessage}`.trim(),
      );
      throw error;
    }
  }

  async function loadHistory() {
    if (!historyPath.trim()) {
      return;
    }
    const row = await loadTrainingHistory(historyPath.trim());
    setHistory(row);
  }

  return {
    nodes,
    edges,
    start_node_id: startNodeId,
    selected_node_id: selectedNodeId,
    console_output: consoleOutput,
    history_path: historyPath,
    history,
    progress,
    set_selected_node_id: setSelectedNodeId,
    set_start_node_id: setStartNodeId,
    set_history_path: setHistoryPath,
    add_node: addNode,
    move_node: moveNode,
    remove_node: removeNode,
    update_node: updateNode,
    add_edge: addEdge,
    remove_edge: removeEdge,
    clear_canvas: clearCanvas,
    run_pipeline: runPipeline,
    load_history: loadHistory,
  };
}
