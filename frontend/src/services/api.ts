import type { DatasetConfigRequest, DatasetInfo, DatasetPreviewResponse } from "../types/dataset";

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
};
