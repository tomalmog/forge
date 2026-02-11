import { VersionSummary } from "../types";

interface DatasetSidebarProps {
  dataRoot: string;
  datasets: string[];
  selectedDataset: string | null;
  versions: VersionSummary[];
  selectedVersion: string | null;
  onDataRootChange: (value: string) => void;
  onRefresh: () => void;
  onDatasetSelect: (dataset: string) => void;
  onVersionSelect: (versionId: string | null) => void;
}

export function DatasetSidebar(props: DatasetSidebarProps) {
  return (
    <aside className="sidebar">
      <h2>Forge Store</h2>
      <label className="label" htmlFor="data-root">
        Data Root
      </label>
      <input
        id="data-root"
        className="input"
        value={props.dataRoot}
        onChange={(event) => props.onDataRootChange(event.currentTarget.value)}
      />
      <button className="button" onClick={props.onRefresh}>
        Refresh Datasets
      </button>

      <div className="section">
        <h3>Datasets</h3>
        <ul className="list">
          {props.datasets.map((dataset) => (
            <li key={dataset}>
              <button
                className={`list-item ${dataset === props.selectedDataset ? "active" : ""}`}
                onClick={() => props.onDatasetSelect(dataset)}
              >
                {dataset}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="section">
        <h3>Versions</h3>
        <button
          className={`list-item ${props.selectedVersion === null ? "active" : ""}`}
          onClick={() => props.onVersionSelect(null)}
        >
          Latest
        </button>
        <ul className="list">
          {props.versions.map((version) => (
            <li key={version.version_id}>
              <button
                className={`list-item ${version.version_id === props.selectedVersion ? "active" : ""}`}
                onClick={() => props.onVersionSelect(version.version_id)}
              >
                <span>{version.version_id.slice(0, 18)}...</span>
                <small>{version.record_count} rows</small>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
