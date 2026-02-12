import { useState } from "react";
import { exportPipelineCanvas } from "../api/studioApi";
import { PipelineEdge, PipelineNode, PipelineNodeType } from "../types";
import { PipelineCanvasToolbar } from "./PipelineCanvasToolbar";
import { PipelineEditorList } from "./PipelineEditorList";
import { PipelineGraphCanvas } from "./PipelineGraphCanvas";
import { PipelinePalette } from "./PipelinePalette";
import { PipelineProgressView } from "./PipelineProgressView";
import { PIPELINE_GRID_SIZE } from "./pipeline_canvas_constants";
import { snapToGrid } from "./pipeline_canvas_math";
import { usePipelineCanvasRuntime } from "./use_pipeline_canvas_runtime";
import "./PipelineCanvas.css";

interface PipelineCanvasProps {
  dataRoot: string;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  startNodeId: string | null;
  selectedNodeId: string | null;
  isRunning: boolean;
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
}

export function PipelineCanvas(props: PipelineCanvasProps) {
  const [canvasActionMessage, setCanvasActionMessage] = useState<string>("");
  const runtime = usePipelineCanvasRuntime({
    nodes: props.nodes,
    edges: props.edges,
    selectedNodeId: props.selectedNodeId,
    onMoveNode: props.onMoveNode,
    onAddEdge: props.onAddEdge,
    onRemoveEdge: props.onRemoveEdge,
    onRemoveNode: props.onRemoveNode,
    onSelectNode: props.onSelectNode,
  });

  return (
    <section className="panel">
      <h3>Visual Training Pipeline</h3>
      <p>
        Build a graph with snap-to-grid nodes. Double-click nodes to open
        editors and keep multiple editors open.
      </p>

      <div className="pipeline-graph-shell">
        <div className="pipeline-control-row">
          <div className="pipeline-controls-left">
            <PipelinePalette
              isRunning={props.isRunning}
              onAddNode={(type) => props.onAddNode(type)}
            />
            <span className="pipeline-control-divider">|</span>
            <PipelineCanvasToolbar
              activeTool={runtime.activeTool}
              wireSourceNodeId={runtime.wireSourceNodeId}
              onSetTool={(mode) => {
                runtime.setActiveTool(mode);
                if (mode !== "wire") {
                  runtime.setWireSourceNodeId(null);
                }
                if (mode !== "erase") {
                  runtime.setIsErasing(false);
                }
              }}
            />
          </div>
          <div className="pipeline-control-actions">
            <button
              className="pipeline-control-button"
              onClick={() => {
                props.onClearCanvas();
                setCanvasActionMessage("Canvas cleared.");
              }}
              disabled={props.isRunning || props.nodes.length === 0}
            >
              Clear Canvas
            </button>
            <button
              className="pipeline-control-button"
              onClick={() => handleExportCanvas()}
              disabled={props.nodes.length === 0}
            >
              Export Canvas
            </button>
          </div>
        </div>
        {canvasActionMessage ? (
          <p className="pipeline-control-message">{canvasActionMessage}</p>
        ) : null}

        <PipelineGraphCanvas
          graphRef={runtime.graphRef}
          nodes={props.nodes}
          edges={props.edges}
          nodeMap={runtime.nodeMap}
          startNodeId={props.startNodeId}
          selectedNodeId={props.selectedNodeId}
          wireSourceNodeId={runtime.wireSourceNodeId}
          openEditorNodeIds={runtime.openEditorNodeIds}
          activeTool={runtime.activeTool}
          onBackgroundClick={() => props.onSelectNode(null)}
          onGraphMouseDown={(event) => {
            if (runtime.activeTool !== "erase") {
              return;
            }
            runtime.setIsErasing(true);
            runtime.eraseAtClientPoint(event.clientX, event.clientY);
          }}
          onGraphMouseMove={(event) => {
            if (runtime.activeTool !== "erase" || !runtime.isErasing) {
              return;
            }
            runtime.eraseAtClientPoint(event.clientX, event.clientY);
          }}
          onGraphMouseUp={() => runtime.setIsErasing(false)}
          onGraphMouseLeave={() => runtime.setIsErasing(false)}
          onCanvasDrop={(type, x, y) =>
            props.onAddNode(
              type,
              Math.max(0, snapToGrid(x, PIPELINE_GRID_SIZE)),
              Math.max(0, snapToGrid(y, PIPELINE_GRID_SIZE)),
            )
          }
          onNodeMouseDown={(node, event) => {
            event.stopPropagation();
            const rect = event.currentTarget.getBoundingClientRect();
            runtime.startNodeDrag(
              node,
              event.clientX,
              event.clientY,
              rect.left,
              rect.top,
            );
          }}
          onNodeClick={(node, event) => {
            event.stopPropagation();
            runtime.handleNodeClick(node);
          }}
          onNodeDoubleClick={(node, event) => {
            event.stopPropagation();
            runtime.handleNodeDoubleClick(node);
          }}
        />
      </div>

      {props.isRunning || props.overallProgressPercent > 0 ? (
        <PipelineProgressView
          overallPercent={props.overallProgressPercent}
          pipelineElapsedSeconds={props.pipelineElapsedSeconds}
          pipelineRemainingSeconds={props.pipelineRemainingSeconds}
          currentStepLabel={props.currentStepLabel}
          currentStepPercent={props.currentStepProgressPercent}
          currentStepElapsedSeconds={props.currentStepElapsedSeconds}
          currentStepRemainingSeconds={props.currentStepRemainingSeconds}
        />
      ) : null}

      <PipelineEditorList
        nodes={runtime.openEditors}
        startNodeId={props.startNodeId}
        nodeMap={runtime.nodeMap}
        edges={props.edges}
        onCloseEditor={runtime.closeEditor}
        onSetStartNode={props.onSetStartNode}
        onRemoveNode={props.onRemoveNode}
        onUpdateNode={props.onUpdateNode}
        onRemoveEdge={props.onRemoveEdge}
      />

      <button
        className="button action"
        onClick={props.onRunPipeline}
        disabled={props.isRunning}
      >
        {props.isRunning ? "Pipeline Running..." : "Run Pipeline"}
      </button>
    </section>
  );

  async function handleExportCanvas(): Promise<void> {
    try {
      const result = await exportPipelineCanvas(
        props.dataRoot,
        props.nodes,
        props.edges,
        props.startNodeId,
      );
      setCanvasActionMessage(`Canvas exported to ${result.output_path}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setCanvasActionMessage(`Canvas export failed: ${message}`);
    }
  }
}
