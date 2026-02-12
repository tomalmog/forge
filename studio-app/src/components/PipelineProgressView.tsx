import { formatDuration } from "./pipeline_canvas_math";

interface PipelineProgressViewProps {
  overallPercent: number;
  pipelineElapsedSeconds: number;
  pipelineRemainingSeconds: number;
  currentStepLabel: string;
  currentStepPercent: number;
  currentStepElapsedSeconds: number;
  currentStepRemainingSeconds: number;
}

export function PipelineProgressView(props: PipelineProgressViewProps) {
  return (
    <div className="progress-shell">
      <div className="progress-head">
        <strong>Pipeline Progress {props.overallPercent.toFixed(1)}%</strong>
        <span>
          elapsed {formatDuration(props.pipelineElapsedSeconds)} | eta{" "}
          {formatDuration(props.pipelineRemainingSeconds)}
        </span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${props.overallPercent}%` }} />
      </div>
      <div className="step-progress">
        <div className="progress-head">
          <strong>{props.currentStepLabel}</strong>
          <span>
            step {props.currentStepPercent.toFixed(1)}% | elapsed{" "}
            {formatDuration(props.currentStepElapsedSeconds)} | eta{" "}
            {formatDuration(props.currentStepRemainingSeconds)}
          </span>
        </div>
        <div className="progress-track progress-track-step">
          <div
            className="progress-fill progress-fill-step"
            style={{ width: `${props.currentStepPercent}%` }}
          />
        </div>
      </div>
    </div>
  );
}
