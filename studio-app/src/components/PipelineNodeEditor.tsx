import { PipelineEdge, PipelineNode } from "../types";

interface PipelineNodeEditorProps {
  node: PipelineNode;
  isStartNode: boolean;
  nodeMap: Map<string, PipelineNode>;
  edges: PipelineEdge[];
  onClose: (nodeId: string) => void;
  onSetStartNode: (nodeId: string) => void;
  onRemoveNode: (nodeId: string) => void;
  onUpdateNode: (nodeId: string, key: string, value: string) => void;
  onRemoveEdge: (edgeId: string) => void;
}

export function PipelineNodeEditor(props: PipelineNodeEditorProps) {
  const relatedEdges = props.edges.filter(
    (edge) =>
      edge.source_node_id === props.node.id ||
      edge.target_node_id === props.node.id,
  );
  return (
    <article className="node-card">
      <header>
        <strong>{props.node.title}</strong>
        <div className="pipeline-editor-header-actions">
          {props.isStartNode ? <small>start</small> : null}
          <button
            className="link-button"
            onClick={() => props.onClose(props.node.id)}
          >
            Close
          </button>
          <button
            className="link-button"
            onClick={() => props.onRemoveNode(props.node.id)}
          >
            Remove
          </button>
        </div>
      </header>

      <div className="pipeline-inspector-actions">
        <button
          className="button"
          onClick={() => props.onSetStartNode(props.node.id)}
        >
          Set As Start
        </button>
      </div>

      {Object.entries(props.node.config).map(([key, value]) => (
        <label key={key}>
          {key}
          <input
            value={value}
            onChange={(event) =>
              props.onUpdateNode(props.node.id, key, event.currentTarget.value)
            }
          />
        </label>
      ))}

      <div className="pipeline-edge-list">
        <strong>Connections</strong>
        {relatedEdges.length === 0 ? (
          <small>No connections.</small>
        ) : (
          relatedEdges.map((edge) => (
            <button
              key={edge.id}
              className="link-button"
              onClick={() => props.onRemoveEdge(edge.id)}
            >
              Remove{" "}
              {edge.source_node_id === props.node.id ? "outgoing" : "incoming"}{" "}
              wire:{" "}
              {props.nodeMap.get(edge.source_node_id)?.title ??
                edge.source_node_id}{" "}
              to{" "}
              {props.nodeMap.get(edge.target_node_id)?.title ??
                edge.target_node_id}
            </button>
          ))
        )}
      </div>
    </article>
  );
}
