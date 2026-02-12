type PipelineToolMode = "move" | "wire" | "erase";

interface PipelineCanvasToolbarProps {
  activeTool: PipelineToolMode;
  wireSourceNodeId: string | null;
  onSetTool: (mode: PipelineToolMode) => void;
}

export function PipelineCanvasToolbar(props: PipelineCanvasToolbarProps) {
  return (
    <div className="pipeline-toolbar">
      <button
        className={`pipeline-tool-button ${props.activeTool === "move" ? "active" : ""}`}
        onClick={() => props.onSetTool("move")}
      >
        Move
      </button>
      <button
        className={`pipeline-tool-button ${props.activeTool === "wire" ? "active" : ""}`}
        onClick={() => props.onSetTool("wire")}
      >
        Wire
      </button>
      <button
        className={`pipeline-tool-button ${props.activeTool === "erase" ? "active" : ""}`}
        onClick={() => props.onSetTool("erase")}
      >
        Erase
      </button>
      {props.wireSourceNodeId ? (
        <small className="pipeline-toolbar-note">
          Wiring mode: source = {props.wireSourceNodeId}. Click next nodes to
          keep chaining.
        </small>
      ) : null}
    </div>
  );
}

export type { PipelineToolMode };
