import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { PixelCanvas } from "../components/image-gallery/PixelCanvas";
import { api } from "../services/api";
import type { ExperimentDetail, ImagePredictionResult } from "../types/experiment";

type Filter = "all" | "correct" | "incorrect";

export default function PredictionAnalysis() {
  const { id } = useParams<{ id: string }>();
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [classFilter, setClassFilter] = useState<string>("all");
  const [selected, setSelected] = useState<ImagePredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    api.getExperiment(id).then((d) => {
      setDetail(d);
      setSelected(d.result?.predictions[0] ?? null);
    }).catch((err) => setError(String(err)));
  }, [id]);

  const classNames = detail?.result?.dataset.class_names ?? [];
  const predictions = detail?.result?.predictions ?? [];

  const filtered = useMemo(() => {
    return predictions
      .filter((p) => (filter === "all" ? true : filter === "correct" ? p.correct : !p.correct))
      .filter((p) => (classFilter === "all" ? true : String(p.true_label) === classFilter));
  }, [predictions, filter, classFilter]);

  if (!id) {
    return (
      <div className="p-6">
        <div className="mb-6 border-b border-border pb-4">
          <h1 className="text-lg font-semibold tracking-tight text-ink">Prediction Analysis</h1>
          <p className="mt-1 text-sm text-muted">No experiment selected.</p>
        </div>
        <Link to="/library" className="inline-block bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover">
          Pick a completed experiment from the Experiment Library
        </Link>
      </div>
    );
  }

  if (error) return <p className="p-6 text-sm text-danger">{error}</p>;
  if (!detail) return <p className="p-6 text-sm text-muted">Loading…</p>;
  if (!detail.result) return <p className="p-6 text-sm text-muted">This experiment has no results yet (status: {detail.status}).</p>;

  const correctCount = predictions.filter((p) => p.correct).length;

  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-y-auto p-5">
        <div className="mb-4 border-b border-border pb-3">
          <h1 className="text-lg font-semibold tracking-tight text-ink">{detail.name}</h1>
          <p className="mt-1 font-mono-tech text-xs text-muted">
            {correctCount}/{predictions.length} correct on the test set ({((correctCount / predictions.length) * 100).toFixed(1)}%)
          </p>
        </div>

        <div className="mb-4 flex flex-wrap gap-2">
          {(["all", "correct", "incorrect"] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`border px-3 py-1 text-xs capitalize ${filter === f ? "border-accent bg-accent/10 text-accent" : "border-border text-ink hover:bg-canvas"}`}
            >
              {f}
            </button>
          ))}
          <select className="border border-border bg-canvas px-2 py-1 text-xs" value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
            <option value="all">All classes</option>
            {classNames.map((name, idx) => (
              <option key={idx} value={idx}>
                {name}
              </option>
            ))}
          </select>
          <span className="self-center font-mono-tech text-xs text-muted">{filtered.length} shown</span>
        </div>

        <div className="grid grid-cols-8 gap-2">
          {filtered.map((p) => (
            <button key={p.index} onClick={() => setSelected(p)} className="flex flex-col items-center gap-1">
              <PixelCanvas pixels={p.pixels} pixelMax={16} size={44} highlight={selected?.index === p.index} />
              <span className={`text-[10px] ${p.correct ? "text-success" : "text-danger"}`}>
                {classNames[p.predicted_label] ?? p.predicted_label}
              </span>
            </button>
          ))}
        </div>
      </div>

      {selected && (
        <aside className="w-64 shrink-0 border-l border-border bg-surface p-4">
          <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Inspect prediction</h2>
          <div className="mb-3 flex justify-center">
            <PixelCanvas pixels={selected.pixels} pixelMax={16} size={160} />
          </div>
          <dl className="grid grid-cols-2 gap-y-1 font-mono-tech text-xs">
            <dt className="text-muted">True label</dt>
            <dd>{classNames[selected.true_label] ?? selected.true_label}</dd>
            <dt className="text-muted">Predicted</dt>
            <dd className={selected.correct ? "text-success" : "text-danger"}>
              {classNames[selected.predicted_label] ?? selected.predicted_label}
            </dd>
            <dt className="text-muted">Confidence</dt>
            <dd>{(selected.confidence * 100).toFixed(1)}%</dd>
            <dt className="text-muted">Status</dt>
            <dd>{selected.correct ? "correct" : "incorrect"}</dd>
          </dl>
        </aside>
      )}
    </div>
  );
}
