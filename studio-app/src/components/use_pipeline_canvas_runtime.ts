import { RefObject, useEffect, useMemo, useRef, useState } from "react";
import { PipelineEdge, PipelineNode } from "../types";
import {
  PIPELINE_ERASER_RADIUS,
  PIPELINE_GRID_SIZE,
  PIPELINE_NODE_HEIGHT,
  PIPELINE_NODE_WIDTH,
} from "./pipeline_canvas_constants";
import { clamp, snapToGrid } from "./pipeline_canvas_math";
import { detectEraserHits } from "./pipeline_eraser";
import { PipelineToolMode } from "./PipelineCanvasToolbar";

interface UsePipelineCanvasRuntimeOptions {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  selectedNodeId: string | null;
  onMoveNode: (nodeId: string, x: number, y: number) => void;
  onAddEdge: (sourceNodeId: string, targetNodeId: string) => void;
  onRemoveEdge: (edgeId: string) => void;
  onRemoveNode: (nodeId: string) => void;
  onSelectNode: (nodeId: string | null) => void;
}

interface DragState {
  nodeId: string;
  offsetX: number;
  offsetY: number;
}

export interface PipelineCanvasRuntime {
  graphRef: RefObject<HTMLDivElement | null>;
  nodeMap: Map<string, PipelineNode>;
  openEditors: PipelineNode[];
  activeTool: PipelineToolMode;
  isErasing: boolean;
  wireSourceNodeId: string | null;
  openEditorNodeIds: string[];
  setActiveTool: (mode: PipelineToolMode) => void;
  setIsErasing: (isErasing: boolean) => void;
  setWireSourceNodeId: (nodeId: string | null) => void;
  closeEditor: (nodeId: string) => void;
  eraseAtClientPoint: (clientX: number, clientY: number) => void;
  startNodeDrag: (
    node: PipelineNode,
    clientX: number,
    clientY: number,
    nodeLeft: number,
    nodeTop: number,
  ) => void;
  handleNodeClick: (node: PipelineNode) => void;
  handleNodeDoubleClick: (node: PipelineNode) => void;
}

export function usePipelineCanvasRuntime(
  options: UsePipelineCanvasRuntimeOptions,
): PipelineCanvasRuntime {
  const {
    nodes,
    edges,
    selectedNodeId,
    onMoveNode,
    onAddEdge,
    onRemoveEdge,
    onRemoveNode,
    onSelectNode,
  } = options;
  const graphRef = useRef<HTMLDivElement | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [isErasing, setIsErasing] = useState(false);
  const [activeTool, setActiveTool] = useState<PipelineToolMode>("move");
  const [wireSourceNodeId, setWireSourceNodeId] = useState<string | null>(null);
  const [openEditorNodeIds, setOpenEditorNodeIds] = useState<string[]>([]);

  const nodeMap = useMemo(
    () => new Map(nodes.map((node) => [node.id, node])),
    [nodes],
  );
  const openEditors = openEditorNodeIds
    .map((nodeId) => nodeMap.get(nodeId))
    .filter((node): node is PipelineNode => node !== undefined);

  useEffect(() => {
    const nodeIdSet = new Set(nodes.map((node) => node.id));
    setOpenEditorNodeIds((current) =>
      current.filter((id) => nodeIdSet.has(id)),
    );
    if (wireSourceNodeId && !nodeIdSet.has(wireSourceNodeId)) {
      setWireSourceNodeId(null);
    }
  }, [nodes, wireSourceNodeId]);

  useEffect(() => {
    if (!dragState || activeTool !== "move") {
      return;
    }
    const onMouseMove = (event: MouseEvent) => {
      const graphElement = graphRef.current;
      if (!graphElement) {
        return;
      }
      const rect = graphElement.getBoundingClientRect();
      const rawX = event.clientX - rect.left - dragState.offsetX;
      const rawY = event.clientY - rect.top - dragState.offsetY;
      const maxX = Math.max(0, rect.width - PIPELINE_NODE_WIDTH);
      const maxY = Math.max(0, rect.height - PIPELINE_NODE_HEIGHT);
      const nextX = clamp(snapToGrid(rawX, PIPELINE_GRID_SIZE), 0, maxX);
      const nextY = clamp(snapToGrid(rawY, PIPELINE_GRID_SIZE), 0, maxY);
      onMoveNode(dragState.nodeId, nextX, nextY);
    };
    const onMouseUp = () => setDragState(null);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [activeTool, dragState, onMoveNode]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!selectedNodeId) {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        onRemoveNode(selectedNodeId);
        onSelectNode(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onRemoveNode, onSelectNode, selectedNodeId]);

  function eraseAtClientPoint(clientX: number, clientY: number): void {
    const graphElement = graphRef.current;
    if (!graphElement) {
      return;
    }
    const rect = graphElement.getBoundingClientRect();
    const hits = detectEraserHits(
      clientX - rect.left,
      clientY - rect.top,
      PIPELINE_ERASER_RADIUS,
      nodes,
      edges,
      nodeMap,
    );
    for (const edgeId of hits.edge_ids) {
      onRemoveEdge(edgeId);
    }
    for (const nodeId of hits.node_ids) {
      onRemoveNode(nodeId);
      if (selectedNodeId === nodeId) {
        onSelectNode(null);
      }
    }
  }

  function startNodeDrag(
    node: PipelineNode,
    clientX: number,
    clientY: number,
    nodeLeft: number,
    nodeTop: number,
  ): void {
    if (activeTool === "erase") {
      setIsErasing(true);
      eraseAtClientPoint(clientX, clientY);
      return;
    }
    if (activeTool !== "move") {
      return;
    }
    setDragState({
      nodeId: node.id,
      offsetX: clientX - nodeLeft,
      offsetY: clientY - nodeTop,
    });
  }

  function handleNodeClick(node: PipelineNode): void {
    if (activeTool === "erase") {
      return;
    }
    onSelectNode(node.id);
    if (activeTool !== "wire") {
      return;
    }
    if (!wireSourceNodeId) {
      setWireSourceNodeId(node.id);
      return;
    }
    if (wireSourceNodeId === node.id) {
      setWireSourceNodeId(null);
      return;
    }
    onAddEdge(wireSourceNodeId, node.id);
    setWireSourceNodeId(node.id);
  }

  function handleNodeDoubleClick(node: PipelineNode): void {
    const isEditorOpen = openEditorNodeIds.includes(node.id);
    if (isEditorOpen) {
      setOpenEditorNodeIds((current) => current.filter((id) => id !== node.id));
      if (selectedNodeId === node.id) {
        onSelectNode(null);
      }
      return;
    }
    setOpenEditorNodeIds((current) => [
      ...current.filter((id) => id !== node.id),
      node.id,
    ]);
    onSelectNode(node.id);
  }

  function closeEditor(nodeId: string): void {
    setOpenEditorNodeIds((current) => current.filter((id) => id !== nodeId));
  }

  return {
    graphRef,
    nodeMap,
    openEditors,
    activeTool,
    isErasing,
    wireSourceNodeId,
    openEditorNodeIds,
    setActiveTool,
    setIsErasing,
    setWireSourceNodeId,
    closeEditor,
    eraseAtClientPoint,
    startNodeDrag,
    handleNodeClick,
    handleNodeDoubleClick,
  };
}
