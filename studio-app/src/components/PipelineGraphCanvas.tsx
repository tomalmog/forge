import { MouseEvent, RefObject } from "react";
import { PipelineEdge, PipelineNode, PipelineNodeType } from "../types";
import {
  PIPELINE_NODE_HEIGHT,
  PIPELINE_NODE_WIDTH,
} from "./pipeline_canvas_constants";

interface PipelineGraphCanvasProps {
  graphRef: RefObject<HTMLDivElement | null>;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  nodeMap: Map<string, PipelineNode>;
  startNodeId: string | null;
  selectedNodeId: string | null;
  wireSourceNodeId: string | null;
  openEditorNodeIds: string[];
  activeTool: "move" | "wire" | "erase";
  onBackgroundClick: () => void;
  onGraphMouseDown: (event: MouseEvent<HTMLDivElement>) => void;
  onGraphMouseMove: (event: MouseEvent<HTMLDivElement>) => void;
  onGraphMouseUp: () => void;
  onGraphMouseLeave: () => void;
  onCanvasDrop: (type: PipelineNodeType, x: number, y: number) => void;
  onNodeMouseDown: (
    node: PipelineNode,
    event: MouseEvent<HTMLButtonElement>,
  ) => void;
  onNodeClick: (
    node: PipelineNode,
    event: MouseEvent<HTMLButtonElement>,
  ) => void;
  onNodeDoubleClick: (
    node: PipelineNode,
    event: MouseEvent<HTMLButtonElement>,
  ) => void;
}

export function PipelineGraphCanvas(props: PipelineGraphCanvasProps) {
  return (
    <div
      className={`pipeline-graph tool-${props.activeTool}`}
      ref={props.graphRef}
      onClick={props.onBackgroundClick}
      onMouseDown={props.onGraphMouseDown}
      onMouseMove={props.onGraphMouseMove}
      onMouseUp={props.onGraphMouseUp}
      onMouseLeave={props.onGraphMouseLeave}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        const type = event.dataTransfer.getData(
          "forge-node",
        ) as PipelineNodeType;
        if (!type) {
          return;
        }
        const rect = event.currentTarget.getBoundingClientRect();
        props.onCanvasDrop(
          type,
          event.clientX - rect.left - 90,
          event.clientY - rect.top - 44,
        );
      }}
    >
      <svg className="pipeline-edge-layer">
        <defs>
          <marker
            id="pipeline-arrow"
            markerWidth="10"
            markerHeight="10"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerUnits="strokeWidth"
            orient="auto"
          >
            <path d="M0,0 L10,5 L0,10 Z" fill="#7a8aa6" />
          </marker>
        </defs>
        {props.edges.map((edge) => {
          const source = props.nodeMap.get(edge.source_node_id);
          const target = props.nodeMap.get(edge.target_node_id);
          if (!source || !target) {
            return null;
          }
          return (
            <line
              key={edge.id}
              className="pipeline-edge-line"
              x1={source.canvas_x + PIPELINE_NODE_WIDTH}
              y1={source.canvas_y + PIPELINE_NODE_HEIGHT / 2}
              x2={target.canvas_x}
              y2={target.canvas_y + PIPELINE_NODE_HEIGHT / 2}
              markerEnd="url(#pipeline-arrow)"
            />
          );
        })}
      </svg>

      {props.nodes.map((node) => {
        const isSelected = node.id === props.selectedNodeId;
        const isStartNode = node.id === props.startNodeId;
        const isWireSource = node.id === props.wireSourceNodeId;
        const isEditorOpen = props.openEditorNodeIds.includes(node.id);
        return (
          <button
            key={node.id}
            className={`pipeline-graph-node ${isSelected ? "selected" : ""} ${isStartNode ? "start" : ""} ${isWireSource ? "wire-source" : ""} ${isEditorOpen ? "editor-open" : ""}`}
            style={{
              transform: `translate(${node.canvas_x}px, ${node.canvas_y}px)`,
            }}
            onMouseDown={(event) => props.onNodeMouseDown(node, event)}
            onClick={(event) => props.onNodeClick(node, event)}
            onDoubleClick={(event) => props.onNodeDoubleClick(node, event)}
          >
            <strong>{node.title}</strong>
            <small>{node.type}</small>
            {isStartNode ? <span className="node-badge">start</span> : null}
          </button>
        );
      })}

      {props.nodes.length === 0 ? (
        <p className="pipeline-empty">
          Drop a component here to start building the graph.
        </p>
      ) : null}
    </div>
  );
}
