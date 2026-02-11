import { PipelineNode, PipelineNodeType } from "../types";

const palette: Array<{ type: PipelineNodeType; title: string }> = [
  { type: "ingest", title: "Ingest" },
  { type: "filter", title: "Filter" },
  { type: "train", title: "Train" },
  { type: "export", title: "Export" },
  { type: "custom", title: "Custom Step" },
];

interface PipelineCanvasProps {
  nodes: PipelineNode[];
  isRunning: boolean;
  overallProgressPercent: number;
  pipelineElapsedSeconds: number;
  pipelineRemainingSeconds: number;
  currentStepLabel: string;
  currentStepProgressPercent: number;
  currentStepElapsedSeconds: number;
  currentStepRemainingSeconds: number;
  onAddNode: (type: PipelineNodeType) => void;
  onRemoveNode: (nodeId: string) => void;
  onUpdateNode: (nodeId: string, key: string, value: string) => void;
  onRunPipeline: () => void;
}

export function PipelineCanvas(props: PipelineCanvasProps) {
  return (
    <section className="panel">
      <h3>Visual Training Pipeline</h3>
      <p>
        Drag components into the canvas, configure them, and run sequentially.
      </p>

      <div className="palette">
        {palette.map((entry) => (
          <button
            className="palette-item"
            key={entry.type}
            draggable
            disabled={props.isRunning}
            onDragStart={(event) =>
              event.dataTransfer.setData("forge-node", entry.type)
            }
            onClick={() => props.onAddNode(entry.type)}
          >
            {entry.title}
          </button>
        ))}
      </div>

      {(props.isRunning || props.overallProgressPercent > 0) && (
        <div className="progress-shell">
          <div className="progress-head">
            <strong>
              Pipeline Progress {props.overallProgressPercent.toFixed(1)}%
            </strong>
            <span>
              elapsed {formatDuration(props.pipelineElapsedSeconds)} | eta{" "}
              {formatDuration(props.pipelineRemainingSeconds)}
            </span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${props.overallProgressPercent}%` }}
            />
          </div>
          <div className="step-progress">
            <div className="progress-head">
              <strong>{props.currentStepLabel}</strong>
              <span>
                step {props.currentStepProgressPercent.toFixed(1)}% | elapsed{" "}
                {formatDuration(props.currentStepElapsedSeconds)} | eta{" "}
                {formatDuration(props.currentStepRemainingSeconds)}
              </span>
            </div>
            <div className="progress-track progress-track-step">
              <div
                className="progress-fill progress-fill-step"
                style={{ width: `${props.currentStepProgressPercent}%` }}
              />
            </div>
          </div>
        </div>
      )}

      <div
        className="pipeline-canvas"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          const type = event.dataTransfer.getData(
            "forge-node",
          ) as PipelineNodeType;
          if (type) {
            props.onAddNode(type);
          }
        }}
      >
        {props.nodes.length === 0 ? (
          <p>Drop a component here to start building the pipeline.</p>
        ) : (
          props.nodes.map((node, index) => (
            <PipelineNodeCard
              key={node.id}
              node={node}
              index={index}
              onRemoveNode={props.onRemoveNode}
              onUpdateNode={props.onUpdateNode}
            />
          ))
        )}
      </div>

      <button
        className="button action"
        onClick={props.onRunPipeline}
        disabled={props.isRunning}
      >
        {props.isRunning ? "Pipeline Running..." : "Run Pipeline"}
      </button>
    </section>
  );
}

function PipelineNodeCard({
  node,
  index,
  onRemoveNode,
  onUpdateNode,
}: {
  node: PipelineNode;
  index: number;
  onRemoveNode: (nodeId: string) => void;
  onUpdateNode: (nodeId: string, key: string, value: string) => void;
}) {
  const fieldRows = Object.entries(node.config);
  return (
    <article className="node-card">
      <header>
        <strong>
          {index + 1}. {node.title}
        </strong>
        <button className="link-button" onClick={() => onRemoveNode(node.id)}>
          Remove
        </button>
      </header>
      {fieldRows.map(([key, value]) => (
        <label key={key}>
          {key}
          <input
            value={value}
            onChange={(event) =>
              onUpdateNode(node.id, key, event.currentTarget.value)
            }
          />
        </label>
      ))}
    </article>
  );
}

function formatDuration(totalSeconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;
  return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
}
