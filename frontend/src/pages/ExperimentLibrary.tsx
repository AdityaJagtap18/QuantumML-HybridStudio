import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../services/api";
import type { ExperimentSummary } from "../types/experiment";

const STATUS_COLORS: Record<string, string> = {
  COMPLETED: "text-success",
  FAILED: "text-danger",
  RUNNING: "text-accent",
  QUEUED: "text-muted",
};

export default function ExperimentLibrary() {
  const [experiments, setExperiments] = useState<ExperimentSummary[] | null>(null);
  const [search, setSearch] = useState("");
  const [kindFilter, setKindFilter] = useState("all");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    api.listExperiments().then(setExperiments);
  }, [refreshKey]);

  async function handleDelete(id: string) {
    await api.deleteExperiment(id);
    setRefreshKey((k) => k + 1);
  }

  if (!experiments) return <p className="p-6 text-sm text-muted">Loading…</p>;

  const kinds = Array.from(new Set(experiments.map((e) => e.architecture_kind)));
  const filtered = experiments
    .filter((e) => e.name.toLowerCase().includes(search.toLowerCase()))
    .filter((e) => kindFilter === "all" || e.architecture_kind === kindFilter);

  return (
    <div className="p-6">
      <div className="mb-6 border-b border-border pb-4">
        <h1 className="text-lg font-semibold tracking-tight text-ink">Experiment Library</h1>
        <p className="mt-1 text-sm text-muted">Browse, search, duplicate and inspect saved runs.</p>
      </div>

      {experiments.length === 0 ? (
        <p className="text-sm text-muted">
          No experiments yet.{" "}
          <Link to="/architecture" className="text-accent hover:underline">
            Start one from the Architecture Studio.
          </Link>
        </p>
      ) : (
        <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
          <div className="mb-3 flex flex-wrap gap-3">
            <input
              className="border border-border bg-canvas px-2 py-1.5 text-sm"
              placeholder="Search by name…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select className="border border-border bg-canvas px-2 py-1.5 text-sm" value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
              <option value="all">All architectures</option>
              {kinds.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>

          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-border text-muted">
                <th className="px-2 py-1.5 font-medium">Name</th>
                <th className="px-2 py-1.5 font-medium">Architecture</th>
                <th className="px-2 py-1.5 font-medium">Dataset</th>
                <th className="px-2 py-1.5 font-medium">Status</th>
                <th className="px-2 py-1.5 font-medium">Created</th>
                <th className="px-2 py-1.5 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e) => (
                <tr key={e.id} className="border-b border-border/60">
                  <td className="px-2 py-1.5">
                    <Link to={`/training/${e.id}`} className="font-medium text-accent hover:underline">
                      {e.name}
                    </Link>
                  </td>
                  <td className="px-2 py-1.5 font-mono-tech text-xs">{e.architecture_kind}</td>
                  <td className="px-2 py-1.5 font-mono-tech text-xs">{e.dataset_id}</td>
                  <td className={`px-2 py-1.5 font-mono-tech text-xs ${STATUS_COLORS[e.status] ?? ""}`}>{e.status}</td>
                  <td className="px-2 py-1.5 font-mono-tech text-xs text-muted">{new Date(e.created_at).toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-right">
                    {e.status === "COMPLETED" && (
                      <Link to={`/predictions/${e.id}`} className="mr-3 text-xs text-accent hover:underline">
                        Predictions
                      </Link>
                    )}
                    <button onClick={() => handleDelete(e.id)} className="text-xs text-muted hover:text-danger">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
