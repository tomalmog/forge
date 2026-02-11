import { PanelVisibility, VIEW_CONTROL_ITEMS } from "../view_controls";

interface ViewControlDrawerProps {
  isOpen: boolean;
  visibility: PanelVisibility;
  onToggleOpen: () => void;
  onTogglePanel: (key: keyof PanelVisibility) => void;
}

export function ViewControlDrawer(props: ViewControlDrawerProps) {
  return (
    <aside className={`view-controls ${props.isOpen ? "open" : "collapsed"}`}>
      <header className="view-controls-header">
        <h3>Views</h3>
        <button className="view-controls-toggle" onClick={props.onToggleOpen}>
          {props.isOpen ? "Hide" : "Show"}
        </button>
      </header>
      {props.isOpen && (
        <div className="view-controls-list">
          {VIEW_CONTROL_ITEMS.map((item) => (
            <label className="view-control-row" key={item.key}>
              <input
                type="checkbox"
                checked={props.visibility[item.key]}
                onChange={() => props.onTogglePanel(item.key)}
              />
              <span>{item.label}</span>
            </label>
          ))}
        </div>
      )}
    </aside>
  );
}
