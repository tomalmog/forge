import { LineageGraphSummary, TrainingRunSummary } from "../types";

interface RuntimeInsightsViewProps {
  hardwareProfile: Record<string, string> | null;
  trainingRuns: TrainingRunSummary[];
  lineage: LineageGraphSummary | null;
  onRefresh: () => void;
}

export function RuntimeInsightsView(props: RuntimeInsightsViewProps) {
  return (
    <section className="panel">
      <div className="runtime-insights-header">
        <h3>Runtime Insights</h3>
        <button className="button" onClick={props.onRefresh}>
          Refresh
        </button>
      </div>
      <div className="runtime-insights-grid">
        <article className="runtime-insights-block">
          <h4>Hardware Profile</h4>
          {props.hardwareProfile ? (
            <dl className="runtime-key-value-list">
              {Object.entries(props.hardwareProfile).map(([key, value]) => (
                <div key={key} className="runtime-key-value-row">
                  <dt>{key}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p>No hardware profile loaded.</p>
          )}
        </article>
        <article className="runtime-insights-block">
          <h4>Training Runs</h4>
          {props.trainingRuns.length > 0 ? (
            <ul className="runtime-run-list">
              {props.trainingRuns.map((row) => (
                <li key={row.run_id} className="runtime-run-row">
                  <strong>{row.run_id}</strong>
                  <span>
                    {row.dataset_name}:{row.dataset_version_id}
                  </span>
                  <span>state={row.state}</span>
                  <span>{row.updated_at}</span>
                  {row.model_path ? <small>{row.model_path}</small> : null}
                </li>
              ))}
            </ul>
          ) : (
            <p>No run lifecycle records found.</p>
          )}
        </article>
      </div>
      <article className="runtime-insights-block">
        <h4>Lineage Graph</h4>
        {props.lineage ? (
          <>
            <p>
              runs={props.lineage.run_count} edges={props.lineage.edge_count}
            </p>
            {props.lineage.edges.length > 0 ? (
              <ul className="runtime-lineage-list">
                {props.lineage.edges.slice(0, 24).map((edge, index) => (
                  <li key={`${edge.from}-${edge.to}-${index}`}>
                    {edge.from} {"->"} {edge.to} ({edge.type})
                  </li>
                ))}
              </ul>
            ) : (
              <p>No lineage edges found.</p>
            )}
          </>
        ) : (
          <p>No lineage graph loaded.</p>
        )}
      </article>
    </section>
  );
}
