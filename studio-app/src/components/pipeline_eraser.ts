import { PipelineEdge, PipelineNode } from "../types";
import {
  PIPELINE_NODE_HEIGHT,
  PIPELINE_NODE_WIDTH,
} from "./pipeline_canvas_constants";

export interface EraserHitResult {
  node_ids: string[];
  edge_ids: string[];
}

export function detectEraserHits(
  pointX: number,
  pointY: number,
  radius: number,
  nodes: PipelineNode[],
  edges: PipelineEdge[],
  nodeMap: Map<string, PipelineNode>,
): EraserHitResult {
  const nodeIds = nodes
    .filter((node) => isPointInsideNode(pointX, pointY, radius, node))
    .map((node) => node.id);
  const edgeIds = edges
    .filter((edge) => {
      const source = nodeMap.get(edge.source_node_id);
      const target = nodeMap.get(edge.target_node_id);
      if (!source || !target) {
        return false;
      }
      const startX = source.canvas_x + PIPELINE_NODE_WIDTH;
      const startY = source.canvas_y + PIPELINE_NODE_HEIGHT / 2;
      const endX = target.canvas_x;
      const endY = target.canvas_y + PIPELINE_NODE_HEIGHT / 2;
      return distancePointToSegment(pointX, pointY, startX, startY, endX, endY) <= radius;
    })
    .map((edge) => edge.id);
  return {
    node_ids: nodeIds,
    edge_ids: edgeIds,
  };
}

function isPointInsideNode(
  pointX: number,
  pointY: number,
  radius: number,
  node: PipelineNode,
): boolean {
  const left = node.canvas_x - radius;
  const right = node.canvas_x + PIPELINE_NODE_WIDTH + radius;
  const top = node.canvas_y - radius;
  const bottom = node.canvas_y + PIPELINE_NODE_HEIGHT + radius;
  return pointX >= left && pointX <= right && pointY >= top && pointY <= bottom;
}

function distancePointToSegment(
  pointX: number,
  pointY: number,
  startX: number,
  startY: number,
  endX: number,
  endY: number,
): number {
  const deltaX = endX - startX;
  const deltaY = endY - startY;
  const segmentLengthSquared = deltaX * deltaX + deltaY * deltaY;
  if (segmentLengthSquared === 0) {
    return Math.hypot(pointX - startX, pointY - startY);
  }
  const projection = ((pointX - startX) * deltaX + (pointY - startY) * deltaY) / segmentLengthSquared;
  const clampedProjection = Math.max(0, Math.min(1, projection));
  const nearestX = startX + clampedProjection * deltaX;
  const nearestY = startY + clampedProjection * deltaY;
  return Math.hypot(pointX - nearestX, pointY - nearestY);
}
