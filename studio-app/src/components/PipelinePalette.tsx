import { PipelineNodeType } from "../types";

const PALETTE_ENTRIES: Array<{ type: PipelineNodeType; title: string }> = [
  { type: "ingest", title: "Ingest" },
  { type: "filter", title: "Filter" },
  { type: "train", title: "Train" },
  { type: "export", title: "Export" },
  { type: "chat", title: "Chat" },
  { type: "custom", title: "Custom Step" },
];

interface PipelinePaletteProps {
  isRunning: boolean;
  onAddNode: (type: PipelineNodeType) => void;
}

export function PipelinePalette(props: PipelinePaletteProps) {
  return (
    <div className="palette">
      {PALETTE_ENTRIES.map((entry) => (
        <button
          className="palette-item"
          key={entry.type}
          draggable
          disabled={props.isRunning}
          onDragStart={(event) => event.dataTransfer.setData("forge-node", entry.type)}
          onClick={() => props.onAddNode(entry.type)}
        >
          {entry.title}
        </button>
      ))}
    </div>
  );
}
