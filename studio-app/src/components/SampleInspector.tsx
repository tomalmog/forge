import { RecordSample } from "../types";

interface SampleInspectorProps {
  samples: RecordSample[];
}

export function SampleInspector({ samples }: SampleInspectorProps) {
  return (
    <section className="panel">
      <h3>Sample Inspector</h3>
      {samples.length === 0 ? (
        <p>No sample rows loaded.</p>
      ) : (
        <div className="sample-list">
          {samples.map((sample) => (
            <article className="sample-card" key={sample.record_id}>
              <header>
                <strong>{sample.record_id.slice(0, 14)}</strong>
                <span>{sample.language}</span>
                <span>{sample.quality_score.toFixed(3)}</span>
              </header>
              <p>{sample.text.slice(0, 280)}...</p>
              <small>{sample.source_uri}</small>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
