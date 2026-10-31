import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../services/api";
import type { ExperimentDetail, ExperimentSummary } from "../types/experiment";

const COLORS = ["#315E9E", "#1F8A5A", "#B5750A", "#8E44AD", "#B3261E"];

export default function BenchmarkWorkspace() {
  const [summaries, setSummaries] = useState<ExperimentSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [details, setDetails] = useState<Record<string, ExperimentDetail>>({});

  useEffect(() => {
    api.listExperiments().then((all) => setSummaries(all.filter((e) => e.status === "COMPLETED")));
  }, []);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = prev.includes(id) ? prev.filter((v) => v !== id) : [...prev, id];
      if (!prev.includes(id)) {
        api.getExperiment(id).then((d) => setDetails((cur) => ({ ...cur, [id]: d })));
      }
      return next;
    });
  }

  const selected = selectedIds.map((id) => details[id]).filter((d): d is ExperimentDetail => !!d && !!d.result);

  const metricData = ["accuracy", "f1", "precision", "recall"].map((metric) => {
    const row: Record<string, string | number> = { metric };
    for (const d of selected) {
      row[d.name] = (d.result!.metrics as any)[metric];
    }
    return row;
  });

  const timingData = [
    { metric: "Train time (s)", ...Object.fromEntries(selected.map((d) => [d.name, d.result!.training.total_seconds])) },
    { metric: "Total params", ...Object.fromEntries(selected.map((d) => [d.name, d.result!.resource_summary.total_parameters])) },
  ];

  const best = selected.length > 0 ? selected.reduce((a, b) => (a.result!.metrics.accuracy >= b.result!.metrics.accuracy ? a : b)) : null;
  const bestHybrid = selected.filter((d) => d.result!.resource_summary.uses_quantum_layer).sort((a, b) => b.result!.metrics.accuracy - a.result!.metrics.accuracy)[0];
  const bestClassical = selected.filter((d) => !d.result!.resource_summary.uses_quantum_layer).sort((a, b) => b.result!.metrics.accuracy - a.result!.metrics.accuracy)[0];

  return (
    <div className="p-6">
      <div className="mb-6 border-b border-border pb-4">
        <h1 className="text-lg font-semibold tracking-tight text-ink">Benchmark Workspace</h1>
        <p className="mt-1 text-sm text-muted">Select completed runs to compare — classical baselines against hybrid models.</p>
      </div>

      <div className="mb-5 border border-border border-t-2 border-t-accent bg-surface p-4">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Completed runs</h2>
        {summaries.length === 0 ? (
          <p className="text-sm text-muted">No completed experiments yet.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {summaries.map((e) => (
              <button
                key={e.id}
                onClick={() => toggle(e.id)}
                className={`border px-2.5 py-1.5 text-left text-xs ${
                  selectedIds.includes(e.id) ? "border-accent bg-accent/10 text-accent" : "border-border text-ink hover:bg-canvas"
                }`}
              >
                <span className="block font-medium">{e.name}</span>
                <span className="block font-mono-tech text-[10px] text-muted">{e.architecture_kind}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {selected.length > 0 && (
        <div className="space-y-5">
          <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Metric comparison</h2>
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-border text-muted">
                  <th className="px-2 py-1 font-medium">Model</th>
                  <th className="px-2 py-1 font-medium">Architecture</th>
                  <th className="px-2 py-1 font-medium">Accuracy</th>
                  <th className="px-2 py-1 font-medium">F1</th>
                  <th className="px-2 py-1 font-medium">Train (s)</th>
                  <th className="px-2 py-1 font-medium">Classical / Quantum params</th>
                </tr>
              </thead>
              <tbody>
                {selected.map((d) => (
                  <tr key={d.id} className="border-b border-border/60">
                    <td className="px-2 py-1 font-medium">{d.name}</td>
                    <td className="px-2 py-1 font-mono-tech text-xs">{d.architecture_kind}</td>
                    <td className="px-2 py-1 font-mono-tech text-xs">{d.result!.metrics.accuracy.toFixed(3)}</td>
                    <td className="px-2 py-1 font-mono-tech text-xs">{d.result!.metrics.f1.toFixed(3)}</td>
                    <td className="px-2 py-1 font-mono-tech text-xs">{d.result!.training.total_seconds.toFixed(2)}</td>
                    <td className="px-2 py-1 font-mono-tech text-xs">
                      {d.result!.resource_summary.classical_parameters}/{d.result!.resource_summary.quantum_parameters}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-2 gap-5">
            <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
              <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Metrics</h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={metricData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E3E7ED" />
                  <XAxis dataKey="metric" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {selected.map((d, i) => (
                    <Bar key={d.id} dataKey={d.name} fill={COLORS[i % COLORS.length]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
              <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Training time &amp; complexity</h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={timingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E3E7ED" />
                  <XAxis dataKey="metric" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {selected.map((d, i) => (
                    <Bar key={d.id} dataKey={d.name} fill={COLORS[i % COLORS.length]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Research interpretation</h2>
            <ul className="list-inside list-disc space-y-1.5 text-sm text-ink">
              {best && (
                <li>
                  Highest accuracy overall: <strong>{best.name}</strong> ({best.result!.metrics.accuracy.toFixed(3)})
                </li>
              )}
              {bestHybrid && bestClassical && (
                <li>
                  {bestHybrid.result!.metrics.accuracy > bestClassical.result!.metrics.accuracy
                    ? `The best hybrid model (${bestHybrid.name}) outperformed the best classical model (${bestClassical.name}) on this comparison (${bestHybrid.result!.metrics.accuracy.toFixed(3)} vs ${bestClassical.result!.metrics.accuracy.toFixed(3)} accuracy).`
                    : bestHybrid.result!.metrics.accuracy < bestClassical.result!.metrics.accuracy
                    ? `The best classical model (${bestClassical.name}) outperformed the best hybrid model (${bestHybrid.name}) on this comparison (${bestClassical.result!.metrics.accuracy.toFixed(3)} vs ${bestHybrid.result!.metrics.accuracy.toFixed(3)} accuracy) — no quantum advantage observed here.`
                    : `The best hybrid and best classical model tied on accuracy (${bestHybrid.result!.metrics.accuracy.toFixed(3)}).`}
                </li>
              )}
              <li>Limitations: small sample counts and simulated quantum execution — treat this as a methodology check, not evidence of quantum advantage.</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
