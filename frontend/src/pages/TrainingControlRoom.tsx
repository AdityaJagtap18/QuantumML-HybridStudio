import { AlertTriangle, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../services/api";
import type { ExperimentDetail, ExperimentStage } from "../types/experiment";

const STAGES: ExperimentStage[] = [
  "QUEUED",
  "VALIDATING",
  "PREPROCESSING",
  "BUILDING_MODEL",
  "TRAINING",
  "EVALUATING",
  "COMPLETED",
];

export default function TrainingControlRoom() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const poll = () => {
      api.getExperiment(id).then((d) => {
        if (cancelled) return;
        setDetail(d);
        if (d.status === "COMPLETED" || d.status === "FAILED") {
          if (pollRef.current) window.clearInterval(pollRef.current);
        }
      });
    };
    poll();
    pollRef.current = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [id]);

  useEffect(() => {
    if (!detail?.started_at) return;
    const start = new Date(detail.started_at).getTime();
    const tick = () => setElapsed((Date.now() - start) / 1000);
    tick();
    const t = window.setInterval(tick, 500);
    return () => window.clearInterval(t);
  }, [detail?.started_at]);

  if (!id) {
    return (
      <div className="p-6">
        <div className="mb-6 border-b border-border pb-4">
          <h1 className="text-lg font-semibold tracking-tight text-ink">Training Control Room</h1>
          <p className="mt-1 text-sm text-muted">No active run selected.</p>
        </div>
        <Link to="/architecture" className="inline-block bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover">
          Configure an architecture to train
        </Link>
      </div>
    );
  }

  if (!detail) return <p className="p-6 text-sm text-muted">Connecting to job…</p>;

  const history = detail.training_history ?? [];
  const stageIndex = STAGES.indexOf(detail.stage);
  const failed = detail.status === "FAILED";

  return (
    <div className="p-6">
      <div className="mb-6 border-b border-border pb-4">
        <h1 className="text-lg font-semibold tracking-tight text-ink">{detail.name}</h1>
        <p className="mt-1 font-mono-tech text-xs text-muted">
          {detail.architecture_kind} · {detail.dataset_id} · job {detail.id}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
          <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Status</h2>
          <dl className="mb-3 grid grid-cols-2 gap-y-1 font-mono-tech text-xs">
            <dt className="text-muted">Status</dt>
            <dd>{detail.status}</dd>
            <dt className="text-muted">Elapsed</dt>
            <dd>{elapsed.toFixed(1)}s</dd>
            <dt className="text-muted">Epoch</dt>
            <dd>{history.length > 0 ? `${history[history.length - 1].epoch}` : "—"}</dd>
          </dl>
          <ol className="space-y-1">
            {STAGES.map((stage, i) => {
              const isCurrent = stage === detail.stage && !failed;
              const isDone = i < stageIndex || detail.status === "COMPLETED";
              return (
                <li
                  key={stage}
                  className={`flex items-center gap-2 px-1.5 py-1 text-xs ${
                    isCurrent ? "bg-accent/10 text-accent" : isDone ? "text-ink" : "text-muted"
                  }`}
                >
                  {isCurrent ? <Loader2 size={12} className="animate-spin" /> : <span className="inline-block h-2.5 w-2.5 rounded-full border border-current" />}
                  {stage}
                </li>
              );
            })}
          </ol>
          {failed && (
            <div className="mt-3 flex items-start gap-2 border border-danger/40 bg-danger/10 px-2 py-2 text-xs text-danger">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              <span>{detail.error}</span>
            </div>
          )}
        </div>

        <div className="col-span-2 border border-border border-t-2 border-t-accent bg-surface p-4">
          <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Loss</h2>
          {history.length === 0 ? (
            <p className="text-sm text-muted">Waiting for the first epoch to complete…</p>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3E7ED" />
                <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="train_loss" name="train loss" stroke="#315E9E" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="val_loss" name="val loss" stroke="#B5750A" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}

          <h2 className="mb-3 mt-5 text-[11px] font-semibold uppercase tracking-wide text-accent">Accuracy</h2>
          {history.length > 0 && (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3E7ED" />
                <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="train_accuracy" name="train acc" stroke="#1F8A5A" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="val_accuracy" name="val acc" stroke="#697386" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {detail.status === "COMPLETED" && detail.result && (
        <div className="mt-5 border border-border border-t-2 border-t-accent bg-surface p-4">
          <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Final metrics</h2>
          <dl className="mb-4 grid grid-cols-4 gap-y-1 font-mono-tech text-xs">
            <dt className="text-muted">Accuracy</dt>
            <dd>{detail.result.metrics.accuracy.toFixed(3)}</dd>
            <dt className="text-muted">F1</dt>
            <dd>{detail.result.metrics.f1.toFixed(3)}</dd>
            <dt className="text-muted">Params (classical/quantum)</dt>
            <dd>
              {detail.result.resource_summary.classical_parameters}/{detail.result.resource_summary.quantum_parameters}
            </dd>
            <dt className="text-muted">Train time</dt>
            <dd>{detail.result.training.total_seconds.toFixed(2)}s</dd>
          </dl>
          <div className="flex gap-3">
            <button onClick={() => navigate(`/predictions/${detail.id}`)} className="bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover">
              Inspect predictions
            </button>
            <Link to="/benchmark" className="border border-border px-3 py-1.5 text-sm text-ink hover:bg-canvas">
              Compare in Benchmark Workspace
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
