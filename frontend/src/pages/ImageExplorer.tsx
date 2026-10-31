import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PixelCanvas } from "../components/image-gallery/PixelCanvas";
import { api } from "../services/api";
import type { DatasetId, DatasetInfo, DatasetPreviewResponse, ImageSample } from "../types/dataset";

export default function ImageExplorer() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetId, setDatasetId] = useState<DatasetId>("digits");
  const [selectedClasses, setSelectedClasses] = useState<number[]>([0, 1]);
  const [maxSamples, setMaxSamples] = useState<number>(200);
  const [seed, setSeed] = useState(42);
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [classFilter, setClassFilter] = useState<number | "all">("all");
  const [selectedImage, setSelectedImage] = useState<ImageSample | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.listDatasets().then(setDatasets);
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .previewDataset({
        dataset_id: datasetId,
        selected_classes: selectedClasses.length > 0 ? selectedClasses : null,
        max_samples: maxSamples,
        test_size: 0.2,
        val_size: 0.2,
        random_seed: seed,
      })
      .then((p) => {
        setPreview(p);
        setSelectedImage(p.sample_images[0] ?? null);
      })
      .catch((err) => {
        setError(String(err));
        setPreview(null);
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, JSON.stringify(selectedClasses), maxSamples, seed]);

  const activeInfo = datasets.find((d) => d.id === datasetId);
  const filteredImages = useMemo(() => {
    if (!preview) return [];
    return classFilter === "all" ? preview.sample_images : preview.sample_images.filter((s) => s.label === classFilter);
  }, [preview, classFilter]);

  return (
    <div className="flex h-full">
      <aside className="w-64 shrink-0 overflow-y-auto border-r border-border bg-surface p-4">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-muted">Dataset</h2>
        <div className="mb-4 space-y-1.5">
          {datasets.map((d) => (
            <button
              key={d.id}
              disabled={!d.available}
              onClick={() => setDatasetId(d.id)}
              title={d.unavailable_reason ?? undefined}
              className={`block w-full border px-2.5 py-2 text-left text-xs transition-colors ${
                !d.available
                  ? "cursor-not-allowed border-border text-muted opacity-50"
                  : datasetId === d.id
                  ? "border-accent bg-accent/10 text-ink"
                  : "border-border text-ink hover:bg-canvas"
              }`}
            >
              <span className="block font-medium">{d.name}</span>
              <span className="mt-0.5 block text-[10px] text-muted">
                {d.available ? `${d.image_height}x${d.image_width} · ${d.n_classes} classes` : "unavailable"}
              </span>
            </button>
          ))}
        </div>

        {activeInfo && (
          <>
            <h2 className="mb-2 mt-5 text-[11px] font-semibold uppercase tracking-wide text-muted">Classes</h2>
            <div className="mb-4 flex flex-wrap gap-1.5">
              {activeInfo.class_names.map((name, idx) => {
                const checked = selectedClasses.includes(idx);
                return (
                  <button
                    key={idx}
                    onClick={() =>
                      setSelectedClasses((prev) =>
                        checked ? prev.filter((v) => v !== idx) : [...prev, idx]
                      )
                    }
                    className={`border px-2 py-1 text-[11px] font-mono-tech ${
                      checked ? "border-accent bg-accent/10 text-accent" : "border-border text-muted"
                    }`}
                  >
                    {name}
                  </button>
                );
              })}
            </div>

            <label className="mb-3 block text-[11px] font-medium uppercase tracking-wide text-muted">
              Max samples
              <input
                type="number"
                className="mt-1 w-full border border-border bg-canvas px-2 py-1.5 text-sm font-mono-tech"
                value={maxSamples}
                min={20}
                max={1797}
                step={20}
                onChange={(e) => setMaxSamples(Number(e.target.value))}
              />
            </label>
            <label className="mb-3 block text-[11px] font-medium uppercase tracking-wide text-muted">
              Random seed
              <input
                type="number"
                className="mt-1 w-full border border-border bg-canvas px-2 py-1.5 text-sm font-mono-tech"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
          </>
        )}
      </aside>

      <div className="flex-1 overflow-y-auto p-5">
        {loading && <p className="text-sm text-muted">Loading…</p>}
        {error && <p className="text-sm text-danger">{error}</p>}

        {preview && !loading && (
          <div className="space-y-5">
            <div className="grid grid-cols-4 gap-3">
              <Stat label="Total samples" value={preview.total_samples} />
              <Stat label="Train / val / test" value={`${preview.train_samples}/${preview.val_samples}/${preview.test_samples}`} />
              <Stat label="Pixel range" value={`${preview.pixel_min.toFixed(0)}–${preview.pixel_max.toFixed(0)}`} />
              <Stat label="Pixel mean" value={preview.pixel_mean.toFixed(2)} />
            </div>

            <div className="border border-border border-t-2 border-t-accent bg-surface p-4">
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-accent">Class distribution</h3>
              <ResponsiveContainer width="100%" height={140}>
                <BarChart data={preview.class_distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E3E7ED" />
                  <XAxis dataKey="class_name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#315E9E" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="flex gap-5">
              <div className="flex-1 border border-border border-t-2 border-t-accent bg-surface p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-[11px] font-semibold uppercase tracking-wide text-accent">
                    Image gallery ({filteredImages.length})
                  </h3>
                  <select
                    className="border border-border bg-canvas px-2 py-1 text-xs"
                    value={classFilter}
                    onChange={(e) => setClassFilter(e.target.value === "all" ? "all" : Number(e.target.value))}
                  >
                    <option value="all">All classes</option>
                    {preview.class_distribution.map((c) => (
                      <option key={c.label} value={c.label}>
                        {c.class_name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="grid grid-cols-6 gap-2">
                  {filteredImages.map((img) => (
                    <button key={img.index} onClick={() => setSelectedImage(img)} className="flex flex-col items-center gap-1">
                      <PixelCanvas
                        pixels={img.pixels}
                        pixelMax={preview.pixel_max}
                        size={48}
                        highlight={selectedImage?.index === img.index}
                      />
                      <span className="text-[10px] text-muted">{img.class_name}</span>
                    </button>
                  ))}
                </div>
              </div>

              {selectedImage && (
                <div className="w-64 shrink-0 border border-border border-t-2 border-t-accent bg-surface p-4">
                  <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-accent">Inspect image</h3>
                  <div className="mb-3 flex justify-center">
                    <PixelCanvas pixels={selectedImage.pixels} pixelMax={preview.pixel_max} size={160} />
                  </div>
                  <dl className="mb-3 grid grid-cols-2 gap-y-1 font-mono-tech text-xs">
                    <dt className="text-muted">Index</dt>
                    <dd>{selectedImage.index}</dd>
                    <dt className="text-muted">Label</dt>
                    <dd>{selectedImage.class_name}</dd>
                    <dt className="text-muted">Dimensions</dt>
                    <dd>
                      {selectedImage.pixels.length}x{selectedImage.pixels[0]?.length}
                    </dd>
                  </dl>
                  <p className="mb-1 text-[10px] uppercase tracking-wide text-muted">Pixel values</p>
                  <div className="overflow-x-auto font-mono-tech text-[9px] leading-tight text-ink">
                    {selectedImage.pixels.map((row, i) => (
                      <div key={i}>{row.map((v) => v.toFixed(0).padStart(2, " ")).join(" ")}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-border bg-surface p-3">
      <p className="text-[10px] uppercase tracking-wide text-muted">{label}</p>
      <p className="font-mono-tech text-sm text-ink">{value}</p>
    </div>
  );
}
