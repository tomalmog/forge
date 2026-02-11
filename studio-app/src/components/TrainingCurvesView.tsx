import { TrainingBatchLoss, TrainingEpoch, TrainingHistory } from "../types";
import "./TrainingCurvesView.css";

interface TrainingCurvesViewProps {
  historyPath: string;
  history: TrainingHistory | null;
  onHistoryPathChange: (value: string) => void;
  onLoadHistory: () => void;
}

interface LineSeries {
  label: string;
  color: string;
  values: number[];
}

interface ChartSpec {
  title: string;
  xLabel: string;
  yLabel: string;
  xValues: number[];
  series: LineSeries[];
}

interface Bounds {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

const CHART_WIDTH = 1000;
const CHART_HEIGHT = 430;
const CHART_BOUNDS: Bounds = { top: 58, right: 28, bottom: 78, left: 86 };

export function TrainingCurvesView(props: TrainingCurvesViewProps) {
  return (
    <section className="panel">
      <h3>Training Curves</h3>
      <div className="history-controls">
        <input
          className="input"
          value={props.historyPath}
          onChange={(event) =>
            props.onHistoryPathChange(event.currentTarget.value)
          }
          placeholder="/path/to/history.json"
        />
        <button className="button" onClick={props.onLoadHistory}>
          Load History
        </button>
      </div>
      {props.history ? (
        <CurvesPanel history={props.history} />
      ) : (
        <p>No history loaded.</p>
      )}
    </section>
  );
}

function CurvesPanel({ history }: { history: TrainingHistory }) {
  if (history.epochs.length === 0) {
    return <p>History file has no epoch rows.</p>;
  }
  const charts = buildCharts(history.epochs, history.batch_losses);
  return (
    <div className="training-chart-stack">
      {charts.map((chart) => (
        <LossChart key={chart.title} chart={chart} />
      ))}
    </div>
  );
}

function buildCharts(
  epochs: TrainingEpoch[],
  batchLosses: TrainingBatchLoss[],
): ChartSpec[] {
  const charts: ChartSpec[] = [];
  if (batchLosses.length > 1) {
    charts.push({
      title: "Training Loss by Step",
      xLabel: "Global Step",
      yLabel: "Loss",
      xValues: batchLosses.map((row) => row.global_step),
      series: [
        {
          label: "train loss",
          color: "#7aa2f7",
          values: batchLosses.map((row) => row.train_loss),
        },
      ],
    });
  }
  charts.push({
    title: "Epoch Loss Curves",
    xLabel: "Epoch",
    yLabel: "Loss",
    xValues: epochs.map((row) => row.epoch),
    series: [
      {
        label: "train loss",
        color: "#7aa2f7",
        values: epochs.map((row) => row.train_loss),
      },
      {
        label: "validation loss",
        color: "#d8a46d",
        values: epochs.map((row) => row.validation_loss),
      },
    ],
  });
  return charts;
}

function LossChart({ chart }: { chart: ChartSpec }) {
  const allYValues = chart.series.flatMap((row) => row.values);
  const yRange = yDomain(allYValues);
  const yTicks = createTicks(yRange.min, yRange.max, 5);
  const xTicks = xTickRows(chart.xValues, 6);
  return (
    <article className="training-chart-card">
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        className="training-chart-svg"
      >
        <text className="training-chart-title" x={CHART_WIDTH / 2} y={30}>
          {chart.title}
        </text>
        {yTicks.map((value) => (
          <g key={`y-${value.toFixed(6)}`}>
            <line
              className="training-grid-line"
              x1={CHART_BOUNDS.left}
              x2={CHART_WIDTH - CHART_BOUNDS.right}
              y1={mapY(value, yRange.min, yRange.max)}
              y2={mapY(value, yRange.min, yRange.max)}
            />
            <text
              className="training-axis-tick training-axis-tick-y"
              x={CHART_BOUNDS.left - 10}
              y={mapY(value, yRange.min, yRange.max) + 4}
            >
              {value.toFixed(3)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g key={`x-${tick.index}`}>
            <line
              className="training-grid-line vertical"
              x1={mapX(tick.index, chart.xValues.length)}
              x2={mapX(tick.index, chart.xValues.length)}
              y1={CHART_BOUNDS.top}
              y2={CHART_HEIGHT - CHART_BOUNDS.bottom}
            />
            <text
              className="training-axis-tick training-axis-tick-x"
              x={mapX(tick.index, chart.xValues.length)}
              y={CHART_HEIGHT - CHART_BOUNDS.bottom + 22}
            >
              {tick.label}
            </text>
          </g>
        ))}
        <line
          className="training-axis-line"
          x1={CHART_BOUNDS.left}
          x2={CHART_WIDTH - CHART_BOUNDS.right}
          y1={CHART_HEIGHT - CHART_BOUNDS.bottom}
          y2={CHART_HEIGHT - CHART_BOUNDS.bottom}
        />
        <line
          className="training-axis-line"
          x1={CHART_BOUNDS.left}
          x2={CHART_BOUNDS.left}
          y1={CHART_BOUNDS.top}
          y2={CHART_HEIGHT - CHART_BOUNDS.bottom}
        />
        {chart.series.map((series) => (
          <path
            key={series.label}
            d={seriesPath(series.values, yRange.min, yRange.max)}
            fill="none"
            stroke={series.color}
            strokeWidth={3}
          />
        ))}
        <text
          className="training-axis-label"
          x={CHART_WIDTH / 2}
          y={CHART_HEIGHT - 20}
        >
          {chart.xLabel}
        </text>
        <text
          className="training-axis-label"
          x={24}
          y={CHART_HEIGHT / 2}
          transform={`rotate(-90 24 ${CHART_HEIGHT / 2})`}
        >
          {chart.yLabel}
        </text>
        <g
          transform={`translate(${CHART_WIDTH - CHART_BOUNDS.right - 220}, ${CHART_BOUNDS.top - 28})`}
        >
          {chart.series.map((series, index) => (
            <g
              key={`legend-${series.label}`}
              transform={`translate(0, ${index * 18})`}
            >
              <line
                x1={0}
                y1={9}
                x2={24}
                y2={9}
                stroke={series.color}
                strokeWidth={3}
              />
              <text className="training-legend-label" x={30} y={13}>
                {series.label}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </article>
  );
}

function seriesPath(values: number[], minY: number, maxY: number): string {
  if (values.length === 0) {
    return "";
  }
  return values
    .map(
      (value, index) =>
        `${index === 0 ? "M" : "L"}${mapX(index, values.length).toFixed(2)} ${mapY(value, minY, maxY).toFixed(2)}`,
    )
    .join(" ");
}

function mapX(index: number, totalPoints: number): number {
  const usableWidth = CHART_WIDTH - CHART_BOUNDS.left - CHART_BOUNDS.right;
  if (totalPoints <= 1) {
    return CHART_BOUNDS.left + usableWidth / 2;
  }
  return CHART_BOUNDS.left + (index / (totalPoints - 1)) * usableWidth;
}

function mapY(value: number, minY: number, maxY: number): number {
  const usableHeight = CHART_HEIGHT - CHART_BOUNDS.top - CHART_BOUNDS.bottom;
  const range = Math.max(maxY - minY, 0.000001);
  const normalized = (value - minY) / range;
  return CHART_HEIGHT - CHART_BOUNDS.bottom - normalized * usableHeight;
}

function yDomain(values: number[]): { min: number; max: number } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (Math.abs(max - min) < 0.000001) {
    return { min: min - 0.5, max: max + 0.5 };
  }
  const margin = (max - min) * 0.08;
  return { min: min - margin, max: max + margin };
}

function createTicks(min: number, max: number, count: number): number[] {
  if (count < 2) {
    return [min, max];
  }
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, index) => min + index * step);
}

function xTickRows(
  xValues: number[],
  count: number,
): Array<{ index: number; label: string }> {
  if (xValues.length <= count) {
    return xValues.map((value, index) => ({ index, label: String(value) }));
  }
  const lastIndex = xValues.length - 1;
  return Array.from({ length: count }, (_, tickIndex) => {
    const index = Math.round((tickIndex / (count - 1)) * lastIndex);
    return { index, label: String(xValues[index]) };
  });
}
