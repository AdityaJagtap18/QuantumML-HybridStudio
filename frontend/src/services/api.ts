import type { QuantumLayerConfig, CircuitPreviewResponse } from "../types/architecture";
import type { DatasetConfigRequest, DatasetInfo, DatasetPreviewResponse } from "../types/dataset";
import type { ExperimentCreateRequest, ExperimentDetail, ExperimentSummary } from "../types/experiment";

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listDatasets: () => request<DatasetInfo[]>("/datasets"),
  previewDataset: (config: DatasetConfigRequest) =>
    request<DatasetPreviewResponse>("/datasets/preview", {
      method: "POST",
      body: JSON.stringify(config),
    }),

  previewCircuit: (config: QuantumLayerConfig) =>
    request<CircuitPreviewResponse>("/architectures/circuit-preview", {
      method: "POST",
      body: JSON.stringify(config),
    }),

  listExperiments: () => request<ExperimentSummary[]>("/experiments"),
  createExperiment: (payload: ExperimentCreateRequest) =>
    request<ExperimentDetail>("/experiments", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getExperiment: (id: string) => request<ExperimentDetail>(`/experiments/${id}`),
  deleteExperiment: (id: string) =>
    request<{ deleted: boolean }>(`/experiments/${id}`, { method: "DELETE" }),
};
