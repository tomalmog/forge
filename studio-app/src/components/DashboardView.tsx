import { DatasetDashboard } from "../types";

interface DashboardViewProps {
  dashboard: DatasetDashboard | null;
  showMetrics: boolean;
  showLanguageMix: boolean;
  showTopSources: boolean;
}

export function DashboardView(props: DashboardViewProps) {
  const { dashboard, showMetrics, showLanguageMix, showTopSources } = props;
  if (!dashboard) {
    return (
      <section className="panel">
        <h3>Dataset Dashboard</h3>
        <p>Select a dataset to view quality and source composition.</p>
      </section>
    );
  }

  const sourceMax = Math.max(
    ...dashboard.source_counts.map((row) => row.count),
    1,
  );
  const languageRows = Object.entries(dashboard.language_counts);
  const showSplit = showLanguageMix || showTopSources;
  const splitGridClassName =
    showLanguageMix && showTopSources
      ? "split-grid"
      : "split-grid split-grid-single";

  return (
    <section className="panel">
      <h3>Dataset Dashboard</h3>
      {showMetrics && (
        <div className="stats-grid">
          <MetricCard label="Version" value={dashboard.version_id} />
          <MetricCard label="Records" value={String(dashboard.record_count)} />
          <MetricCard
            label="Avg Quality"
            value={dashboard.average_quality.toFixed(3)}
          />
          <MetricCard
            label="Quality Range"
            value={`${dashboard.min_quality.toFixed(3)} - ${dashboard.max_quality.toFixed(3)}`}
          />
        </div>
      )}

      {showSplit && (
        <div className={splitGridClassName}>
          {showLanguageMix && (
            <div>
              <h4>Language Mix</h4>
              {languageRows.map(([language, count]) => (
                <BarRow
                  key={language}
                  label={language}
                  value={count}
                  max={dashboard.record_count}
                />
              ))}
            </div>
          )}

          {showTopSources && (
            <div>
              <h4>Top Sources</h4>
              {dashboard.source_counts.map((row) => (
                <BarRow
                  key={row.source}
                  label={row.source}
                  value={row.count}
                  max={sourceMax}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}

function BarRow({
  label,
  value,
  max,
}: {
  label: string;
  value: number;
  max: number;
}) {
  const width = Math.max(5, Math.round((value / max) * 100));
  return (
    <div className="bar-row">
      <div className="bar-label">
        <span>{trimLabel(label)}</span>
        <strong>{value}</strong>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function trimLabel(label: string): string {
  return label.length <= 42 ? label : `${label.slice(0, 42)}...`;
}
