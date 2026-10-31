import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { Connector } from "../components/architecture-canvas/Connector";
import { NodeColumn } from "../components/architecture-canvas/NodeColumn";
import { QuantumNode } from "../components/architecture-canvas/QuantumNode";
import { SimpleNode } from "../components/architecture-canvas/SimpleNode";
import { PropertyInspector } from "../components/property-inspector/PropertyInspector";
import { api } from "../services/api";
import type { ArchitectureConfig, CircuitPreviewResponse, ModelKind } from "../types/architecture";
import type { DatasetConfigRequest, DatasetPreviewResponse } from "../types/dataset";
import type { ExperimentCreateRequest } from "../types/experiment";
import type { PreprocessingConfig } from "../types/architecture";
import type { TrainingConfig } from "../types/training";

type NodeKey = "input" | "extractor" | "projection" | "quantum" | "head" | "output" | null;

const DEFAULT_ARCHITECTURE: ArchitectureConfig = {
  model_kind: "hybrid_mlp_qnn",
  mlp_extractor: { hidden_dims: [16], activation: "relu", dropout: 0 },
  cnn_extractor: { channels: [4, 8], kernel_size: 3, pooling: true, fc_dim: 16, activation: "relu", dropout: 0 },
  projection: { activation: "tanh", normalize: true },
  quantum_layer: {
    n_qubits: 4,
    feature_map: "ZZFeatureMap",
    feature_map_reps: 1,
    ansatz: "RealAmplitudes",
    ansatz_reps: 2,
    entanglement: "linear",
    qnn_type: "estimator",
    execution_mode: "ideal",
    shots: 1024,
    noise: { enabled: false, depolarizing_prob: 0.02, readout_error_prob: 0.02 },
  },
  head: { n_outputs: 1 },
};

const MODEL_KINDS: { value: ModelKind; label: string }[] = [
  { value: "classical_mlp", label: "Classical MLP" },
  { value: "classical_cnn", label: "Classical CNN" },
  { value: "hybrid_mlp_qnn", label: "Hybrid MLP + QNN" },
  { value: "hybrid_cnn_qnn", label: "Hybrid CNN + QNN" },
];

export default function ArchitectureStudio() {
  const navigate = useNavigate();
  const [dataset, setDataset] = useState<DatasetConfigRequest>({
    dataset_id: "digits",
    selected_classes: [0, 1],
    max_samples: 200,
    test_size: 0.2,
    val_size: 0.2,
    random_seed: 42,
  });
  const [preprocessing, setPreprocessing] = useState<PreprocessingConfig>({ normalize: true, use_pca: false, n_components: null });
  const [architecture, setArchitecture] = useState<ArchitectureConfig>(DEFAULT_ARCHITECTURE);
  const [training, setTraining] = useState<TrainingConfig>({
    epochs: 10,
    batch_size: 16,
    learning_rate: 0.01,
    optimizer: "adam",
    random_seed: 42,
    early_stopping_patience: 5,
  });
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [selected, setSelected] = useState<NodeKey>("quantum");
  const [circuitPreview, setCircuitPreview] = useState<CircuitPreviewResponse | null>(null);
  const [circuitError, setCircuitError] = useState<string | null>(null);
  const [name, setName] = useState("my-first-hybrid-run");
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const isHybrid = architecture.model_kind === "hybrid_mlp_qnn" || architecture.model_kind === "hybrid_cnn_qnn";
  const isCnn = architecture.model_kind === "classical_cnn" || architecture.model_kind === "hybrid_cnn_qnn";

  useEffect(() => {
    api.previewDataset(dataset).then(setPreview).catch(() => setPreview(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(dataset)]);

  useEffect(() => {
    if (!isHybrid) {
      setCircuitPreview(null);
      return;
    }
    setCircuitError(null);
    api
      .previewCircuit(architecture.quantum_layer)
      .then(setCircuitPreview)
      .catch((err) => {
        setCircuitError(String(err));
        setCircuitPreview(null);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(architecture.quantum_layer), isHybrid]);

  async function handleRun() {
    setRunning(true);
    setRunError(null);
    try {
      const payload: ExperimentCreateRequest = { name, dataset, preprocessing, architecture, training };
      const detail = await api.createExperiment(payload);
      navigate(`/training/${detail.id}`);
    } catch (err) {
      setRunError(String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="flex h-full">
      <aside className="w-72 shrink-0 overflow-y-auto border-r border-border bg-surface p-4">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-muted">Dataset</h2>
        <div className="mb-4 grid grid-cols-2 gap-2 font-mono-tech text-xs">
          <label className="col-span-2 flex flex-col gap-1">
            <span className="text-muted">Max samples</span>
            <input
              type="number"
              className="border border-border bg-canvas px-2 py-1"
              value={dataset.max_samples ?? 200}
              onChange={(e) => setDataset({ ...dataset, max_samples: Number(e.target.value) })}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-muted">Test size</span>
            <input
              type="number"
              step={0.05}
              className="border border-border bg-canvas px-2 py-1"
              value={dataset.test_size}
              onChange={(e) => setDataset({ ...dataset, test_size: Number(e.target.value) })}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-muted">Val size</span>
            <input
              type="number"
              step={0.05}
              className="border border-border bg-canvas px-2 py-1"
              value={dataset.val_size}
              onChange={(e) => setDataset({ ...dataset, val_size: Number(e.target.value) })}
            />
          </label>
        </div>
        {preview && (
          <p className="mb-4 text-[11px] text-muted">
            {preview.total_samples} samples · train {preview.train_samples} / val {preview.val_samples} / test{" "}
            {preview.test_samples}
          </p>
        )}

        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-muted">Model kind</h2>
        <div className="mb-5 space-y-1.5">
          {MODEL_KINDS.map((m) => (
            <button
              key={m.value}
              onClick={() => setArchitecture({ ...architecture, model_kind: m.value })}
              className={`block w-full border px-2.5 py-1.5 text-left text-xs ${
                architecture.model_kind === m.value ? "border-accent bg-accent/10 text-accent" : "border-border text-ink hover:bg-canvas"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-muted">Training</h2>
        <div className="mb-5 grid grid-cols-2 gap-2 font-mono-tech text-xs">
          <label className="flex flex-col gap-1">
            <span className="text-muted">Epochs</span>
            <input type="number" className="border border-border bg-canvas px-2 py-1" value={training.epochs} onChange={(e) => setTraining({ ...training, epochs: Number(e.target.value) })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-muted">Batch size</span>
            <input type="number" className="border border-border bg-canvas px-2 py-1" value={training.batch_size} onChange={(e) => setTraining({ ...training, batch_size: Number(e.target.value) })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-muted">Learning rate</span>
            <input type="number" step={0.001} className="border border-border bg-canvas px-2 py-1" value={training.learning_rate} onChange={(e) => setTraining({ ...training, learning_rate: Number(e.target.value) })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-muted">Optimizer</span>
            <select className="border border-border bg-canvas px-2 py-1" value={training.optimizer} onChange={(e) => setTraining({ ...training, optimizer: e.target.value as any })}>
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
            </select>
          </label>
        </div>

        <h2 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted">Experiment name</h2>
        <input className="mb-4 w-full border border-border bg-canvas px-2 py-1.5 text-sm" value={name} onChange={(e) => setName(e.target.value)} />

        {runError && <p className="mb-2 text-xs text-danger">{runError}</p>}
        <button
          onClick={handleRun}
          disabled={running}
          className="w-full bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          {running ? "Starting…" : "Run Experiment"}
        </button>
      </aside>

      <div className="flex-1 overflow-x-auto p-8">
        <div className="flex items-start gap-0">
          <SimpleNode label="Image Input" sublabel={preview ? `${preview.dataset.image_height}x${preview.dataset.image_width}` : "…"} selected={selected === "input"} onSelect={() => setSelected("input")} />
          <Connector />
          <NodeColumn
            label={isCnn ? "CNN Extractor" : "MLP Extractor"}
            sublabel={isCnn ? architecture.cnn_extractor.channels.join(",") : architecture.mlp_extractor.hidden_dims.join(",")}
            nodeCount={isCnn ? architecture.cnn_extractor.channels.length * 3 : architecture.mlp_extractor.hidden_dims[0] ?? 1}
            selected={selected === "extractor"}
            onSelect={() => setSelected("extractor")}
            onAdd={
              isCnn
                ? () => setArchitecture({ ...architecture, cnn_extractor: { ...architecture.cnn_extractor, channels: [...architecture.cnn_extractor.channels, 8] } })
                : () => setArchitecture({ ...architecture, mlp_extractor: { ...architecture.mlp_extractor, hidden_dims: [...architecture.mlp_extractor.hidden_dims, 8] } })
            }
            onRemove={
              isCnn
                ? () => architecture.cnn_extractor.channels.length > 1 && setArchitecture({ ...architecture, cnn_extractor: { ...architecture.cnn_extractor, channels: architecture.cnn_extractor.channels.slice(0, -1) } })
                : () => architecture.mlp_extractor.hidden_dims.length > 1 && setArchitecture({ ...architecture, mlp_extractor: { ...architecture.mlp_extractor, hidden_dims: architecture.mlp_extractor.hidden_dims.slice(0, -1) } })
            }
          />
          {isHybrid && (
            <>
              <Connector />
              <SimpleNode label="Projection" sublabel={`-> ${architecture.quantum_layer.n_qubits}`} selected={selected === "projection"} onSelect={() => setSelected("projection")} />
              <Connector />
              <QuantumNode nQubits={architecture.quantum_layer.n_qubits} selected={selected === "quantum"} onSelect={() => setSelected("quantum")} />
            </>
          )}
          <Connector />
          <SimpleNode label="Head" sublabel={`${architecture.head.n_outputs} out`} selected={selected === "head"} onSelect={() => setSelected("head")} />
          <Connector />
          <SimpleNode label="Output" sublabel="class" selected={selected === "output"} onSelect={() => setSelected("output")} />
        </div>

        <div className="mt-8 border border-border border-t-2 border-t-accent bg-surface p-4">
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-accent">Architecture summary</h3>
          <p className="font-mono-tech text-xs text-ink">
            {preview ? `${preview.dataset.image_height}x${preview.dataset.image_width}` : "?"} input
            {isCnn ? " -> CNN" : " -> flatten -> MLP"}
            {isHybrid ? ` -> projection(${architecture.quantum_layer.n_qubits}) -> quantum layer` : ""} -&gt; head({architecture.head.n_outputs}) -&gt; prediction
          </p>
          {circuitPreview && (
            <p className="mt-1 font-mono-tech text-xs text-muted">
              Quantum circuit: {circuitPreview.n_encoding_parameters} encoding + {circuitPreview.n_trainable_parameters} trainable
              parameters, depth {circuitPreview.depth}
            </p>
          )}
        </div>
      </div>

      <aside className="w-72 shrink-0 overflow-y-auto border-l border-border bg-surface">
        <PropertyInspector
          selected={selected}
          architecture={architecture}
          onChange={setArchitecture}
          imageHeight={preview?.dataset.image_height ?? 8}
          imageWidth={preview?.dataset.image_width ?? 8}
          circuitPreview={circuitPreview}
          circuitError={circuitError}
        />
      </aside>
    </div>
  );
}
