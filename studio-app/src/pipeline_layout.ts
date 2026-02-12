import { PipelineNode } from "./types";
import {
  PIPELINE_GRID_SIZE,
  PIPELINE_NODE_HEIGHT,
  PIPELINE_NODE_WIDTH,
} from "./components/pipeline_canvas_constants";

const SPAWN_GAP = 8;
const MAX_ATTEMPTS = 240;

export function resolveNonOverlappingNodePosition(
  nodes: PipelineNode[],
  preferredX: number,
  preferredY: number,
): { x: number; y: number } {
  let candidateX = Math.max(0, preferredX);
  let candidateY = Math.max(0, preferredY);
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    if (!overlapsAnyNode(candidateX, candidateY, nodes)) {
      return { x: candidateX, y: candidateY };
    }
    candidateX += PIPELINE_GRID_SIZE;
    if (attempt > 0 && attempt % 8 === 0) {
      candidateX = Math.max(0, preferredX);
      candidateY += PIPELINE_GRID_SIZE;
    }
  }
  return { x: candidateX, y: candidateY };
}

function overlapsAnyNode(
  candidateX: number,
  candidateY: number,
  nodes: PipelineNode[],
): boolean {
  return nodes.some((node) =>
    rectanglesOverlap(
      candidateX,
      candidateY,
      PIPELINE_NODE_WIDTH,
      PIPELINE_NODE_HEIGHT,
      node.canvas_x,
      node.canvas_y,
      PIPELINE_NODE_WIDTH,
      PIPELINE_NODE_HEIGHT,
    ),
  );
}

function rectanglesOverlap(
  ax: number,
  ay: number,
  aw: number,
  ah: number,
  bx: number,
  by: number,
  bw: number,
  bh: number,
): boolean {
  const aRight = ax + aw + SPAWN_GAP;
  const aBottom = ay + ah + SPAWN_GAP;
  const bRight = bx + bw + SPAWN_GAP;
  const bBottom = by + bh + SPAWN_GAP;
  return ax < bRight && aRight > bx && ay < bBottom && aBottom > by;
}
