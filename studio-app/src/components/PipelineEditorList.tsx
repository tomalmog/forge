import { PipelineEdge, PipelineNode } from "../types";
import { PipelineNodeEditor } from "./PipelineNodeEditor";

interface PipelineEditorListProps {
  nodes: PipelineNode[];
  startNodeId: string | null;
  nodeMap: Map<string, PipelineNode>;
  edges: PipelineEdge[];
  onCloseEditor: (nodeId: string) => void;
  onSetStartNode: (nodeId: string | null) => void;
  onRemoveNode: (nodeId: string) => void;
  onUpdateNode: (nodeId: string, key: string, value: string) => void;
  onRemoveEdge: (edgeId: string) => void;
}

export function PipelineEditorList(props: PipelineEditorListProps) {
  if (props.nodes.length === 0) {
    return (
      <div className="pipeline-editor-list">
        <p className="pipeline-empty">
          Double-click any node to open its editor. Multiple editors stay open in click order.
        </p>
      </div>
    );
  }
  return (
    <div className="pipeline-editor-list">
      {props.nodes.map((node) => (
        <PipelineNodeEditor
          key={node.id}
          node={node}
          isStartNode={props.startNodeId === node.id}
          nodeMap={props.nodeMap}
          edges={props.edges}
          onClose={props.onCloseEditor}
          onSetStartNode={props.onSetStartNode}
          onRemoveNode={props.onRemoveNode}
          onUpdateNode={props.onUpdateNode}
          onRemoveEdge={props.onRemoveEdge}
        />
      ))}
    </div>
  );
}
