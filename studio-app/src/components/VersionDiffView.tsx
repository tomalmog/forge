import { VersionDiff, VersionSummary } from "../types";

interface VersionDiffViewProps {
  versions: VersionSummary[];
  baseVersion: string | null;
  targetVersion: string | null;
  diff: VersionDiff | null;
  onBaseVersionChange: (value: string) => void;
  onTargetVersionChange: (value: string) => void;
  onComputeDiff: () => void;
}

export function VersionDiffView(props: VersionDiffViewProps) {
  return (
    <section className="panel">
      <h3>Version Diff</h3>
      <div className="diff-controls">
        <label>
          Base
          <select
            value={props.baseVersion ?? ""}
            onChange={(event) => props.onBaseVersionChange(event.currentTarget.value)}
          >
            <option value="">Select base version</option>
            {props.versions.map((version) => (
              <option key={version.version_id} value={version.version_id}>
                {version.version_id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Target
          <select
            value={props.targetVersion ?? ""}
            onChange={(event) => props.onTargetVersionChange(event.currentTarget.value)}
          >
            <option value="">Select target version</option>
            {props.versions.map((version) => (
              <option key={version.version_id} value={version.version_id}>
                {version.version_id}
              </option>
            ))}
          </select>
        </label>
        <button className="button" onClick={props.onComputeDiff}>
          Compute Diff
        </button>
      </div>

      {props.diff ? (
        <div className="stats-grid">
          <DiffCard label="Added" value={props.diff.added_records} />
          <DiffCard label="Removed" value={props.diff.removed_records} />
          <DiffCard label="Shared" value={props.diff.shared_records} />
        </div>
      ) : (
        <p>Pick two versions and compute diff.</p>
      )}
    </section>
  );
}

function DiffCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric-card">
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}
