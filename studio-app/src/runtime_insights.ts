import {
  getHardwareProfile,
  getLineageGraph,
  listTrainingRuns,
} from "./api/studioApi";
import { LineageGraphSummary, TrainingRunSummary } from "./types";

export interface RuntimeInsightsPayload {
  runs: TrainingRunSummary[];
  lineage: LineageGraphSummary;
  hardwareProfile: Record<string, string>;
}

export async function loadRuntimeInsights(
  dataRoot: string,
): Promise<RuntimeInsightsPayload> {
  const [runs, lineage, hardwareProfile] = await Promise.all([
    listTrainingRuns(dataRoot),
    getLineageGraph(dataRoot),
    getHardwareProfile(dataRoot),
  ]);
  return { runs, lineage, hardwareProfile };
}
