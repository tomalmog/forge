import { PipelineEdge, PipelineNode } from "./types";

export interface PipelineExecutionPlan {
  ordered_nodes: PipelineNode[];
  reachable_node_ids: string[];
}

export function sanitizePipelineGraph(
  nodes: PipelineNode[],
  edges: PipelineEdge[],
): PipelineEdge[] {
  const nodeIdSet = new Set(nodes.map((node) => node.id));
  const uniqueKeys = new Set<string>();
  const cleanedEdges: PipelineEdge[] = [];
  for (const edge of edges) {
    if (!nodeIdSet.has(edge.source_node_id) || !nodeIdSet.has(edge.target_node_id)) {
      continue;
    }
    if (edge.source_node_id === edge.target_node_id) {
      continue;
    }
    const key = `${edge.source_node_id}->${edge.target_node_id}`;
    if (uniqueKeys.has(key)) {
      continue;
    }
    uniqueKeys.add(key);
    cleanedEdges.push(edge);
  }
  return cleanedEdges;
}

export function buildPipelineExecutionPlan(
  nodes: PipelineNode[],
  edges: PipelineEdge[],
  startNodeId: string,
): PipelineExecutionPlan {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  if (!nodeMap.has(startNodeId)) {
    throw new Error(`Start node '${startNodeId}' does not exist in current pipeline.`);
  }
  const normalizedEdges = sanitizePipelineGraph(nodes, edges);
  const reachableIds = collectReachableNodeIds(startNodeId, normalizedEdges);
  const sortedNodes = topologicalSortReachableNodes(nodes, normalizedEdges, reachableIds);
  return {
    ordered_nodes: sortedNodes,
    reachable_node_ids: Array.from(reachableIds),
  };
}

function collectReachableNodeIds(
  startNodeId: string,
  edges: PipelineEdge[],
): Set<string> {
  const outgoing = buildOutgoingEdgeMap(edges);
  const visited = new Set<string>();
  const stack = [startNodeId];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current || visited.has(current)) {
      continue;
    }
    visited.add(current);
    const nextNodes = outgoing.get(current);
    if (!nextNodes) {
      continue;
    }
    for (const nextNodeId of nextNodes) {
      if (!visited.has(nextNodeId)) {
        stack.push(nextNodeId);
      }
    }
  }
  return visited;
}

function topologicalSortReachableNodes(
  nodes: PipelineNode[],
  edges: PipelineEdge[],
  reachableIds: Set<string>,
): PipelineNode[] {
  const stableOrderIndex = buildStableOrderIndex(nodes);
  const indegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();
  for (const nodeId of reachableIds) {
    indegree.set(nodeId, 0);
    adjacency.set(nodeId, []);
  }
  for (const edge of edges) {
    if (!reachableIds.has(edge.source_node_id) || !reachableIds.has(edge.target_node_id)) {
      continue;
    }
    adjacency.get(edge.source_node_id)?.push(edge.target_node_id);
    indegree.set(
      edge.target_node_id,
      (indegree.get(edge.target_node_id) ?? 0) + 1,
    );
  }

  const zeroIndegreeQueue = Array.from(reachableIds).filter(
    (nodeId) => (indegree.get(nodeId) ?? 0) === 0,
  );
  zeroIndegreeQueue.sort((left, right) => stableSort(left, right, stableOrderIndex));

  const sortedNodeIds: string[] = [];
  while (zeroIndegreeQueue.length > 0) {
    const currentNodeId = zeroIndegreeQueue.shift();
    if (!currentNodeId) {
      continue;
    }
    sortedNodeIds.push(currentNodeId);
    const targets = adjacency.get(currentNodeId) ?? [];
    targets.sort((left, right) => stableSort(left, right, stableOrderIndex));
    for (const targetNodeId of targets) {
      const nextIndegree = (indegree.get(targetNodeId) ?? 0) - 1;
      indegree.set(targetNodeId, nextIndegree);
      if (nextIndegree === 0) {
        zeroIndegreeQueue.push(targetNodeId);
        zeroIndegreeQueue.sort((left, right) => stableSort(left, right, stableOrderIndex));
      }
    }
  }

  if (sortedNodeIds.length !== reachableIds.size) {
    throw new Error(
      "Pipeline contains a cycle reachable from the selected start node. " +
        "Remove circular connections before running.",
    );
  }

  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  return sortedNodeIds
    .map((nodeId) => nodeMap.get(nodeId))
    .filter((node): node is PipelineNode => node !== undefined);
}

function buildOutgoingEdgeMap(edges: PipelineEdge[]): Map<string, string[]> {
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    const targets = outgoing.get(edge.source_node_id);
    if (targets) {
      targets.push(edge.target_node_id);
      continue;
    }
    outgoing.set(edge.source_node_id, [edge.target_node_id]);
  }
  return outgoing;
}

function buildStableOrderIndex(nodes: PipelineNode[]): Map<string, number> {
  const index = new Map<string, number>();
  nodes.forEach((node, nodeIndex) => index.set(node.id, nodeIndex));
  return index;
}

function stableSort(
  leftNodeId: string,
  rightNodeId: string,
  orderIndex: Map<string, number>,
): number {
  return (orderIndex.get(leftNodeId) ?? 0) - (orderIndex.get(rightNodeId) ?? 0);
}
