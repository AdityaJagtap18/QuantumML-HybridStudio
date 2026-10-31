import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../services/api";
import type { DatasetInfo } from "../types/dataset";

export default function Workspace() {
  const [datasets, setDatasets] = useState<DatasetInfo[] | null>(null);

  useEffect(() => {
    api.listDatasets().then(setDatasets);
  }, []);

  return (
    <div className="p-6">
      <div className="mb-6 border-b border-border pb-4">
        <h1 className="text-lg font-semibold tracking-tight text-ink">Workspace</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Research question: can a differentiable quantum layer contribute meaningfully to an
          end-to-end classical neural network? Hybrid Studio builds real hybrid architectures —
          image → classical feature extractor → projection → quantum layer (TorchConnector) →
          classification head — and trains them end to end in PyTorch.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 border border-border border-t-2 border-t-accent bg-surface p-4">
          <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Datasets</h2>
          {!datasets && <p className="text-sm text-muted">Loading…</p>}
          {datasets && (
            <ul className="space-y-1.5">
              {datasets.map((d) => (
                <li key={d.id} className="flex items-center justify-between text-sm">
                  <span className={d.available ? "text-ink" : "text-muted"}>{d.name}</span>
                  <span className="font-mono-tech text-xs text-muted">
                    {d.available ? `${d.image_height}x${d.image_width}` : "unavailable"}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <Link
            to="/images"
            className="mt-4 inline-block bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
          >
            Explore images
          </Link>
        </div>

        <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
          <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Architecture</h2>
          <p className="mb-2 text-xs text-ink">image → extractor → projection → quantum layer → head</p>
          <p className="text-xs text-muted">
            Classical: MLP / CNN. Quantum: EstimatorQNN or SamplerQNN via TorchConnector, with a
            configurable feature map and ansatz.
          </p>
          <Link to="/architecture" className="mt-4 inline-block border border-border px-3 py-1.5 text-sm text-ink hover:bg-canvas">
            Open Architecture Studio
          </Link>
        </div>
      </div>
    </div>
  );
}
