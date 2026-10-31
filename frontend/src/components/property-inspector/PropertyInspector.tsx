import type { ReactNode } from "react";

import type {
  ArchitectureConfig,
  CircuitPreviewResponse,
} from "../../types/architecture";

type NodeKey = "input" | "extractor" | "projection" | "quantum" | "head" | "output" | null;

const ACTIVATIONS = ["relu", "tanh", "sigmoid"] as const;

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="text-[11px] font-medium uppercase tracking-wide text-muted">{label}</span>
      {children}
    </label>
  );
}

const inputCls = "border border-border bg-canvas px-2 py-1.5 text-sm font-mono-tech";

export function PropertyInspector({
  selected,
  architecture,
  onChange,
  imageHeight,
  imageWidth,
  circuitPreview,
  circuitError,
}: {
  selected: NodeKey;
  architecture: ArchitectureConfig;
  onChange: (next: ArchitectureConfig) => void;
  imageHeight: number;
  imageWidth: number;
  circuitPreview: CircuitPreviewResponse | null;
  circuitError: string | null;
}) {
  if (!selected) {
    return <p className="p-4 text-sm text-muted">Select a node on the canvas to inspect and configure it.</p>;
  }

  if (selected === "input") {
    return (
      <div className="space-y-3 p-4">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-accent">Image Input</h3>
        <p className="font-mono-tech text-sm text-ink">
          {imageHeight}x{imageWidth}, 1 channel
        </p>
        <p className="text-xs text-muted">Fixed by the selected dataset. Change it in the Image Explorer.</p>
      </div>
    );
  }

  if (selected === "extractor") {
    const isCnn = architecture.model_kind === "classical_cnn" || architecture.model_kind === "hybrid_cnn_qnn";
    return (
      <div className="space-y-3 p-4">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-accent">
          {isCnn ? "CNN Feature Extractor" : "MLP Feature Extractor"}
        </h3>
        {isCnn ? (
          <>
            <Field label="Kernel size">
              <input
                type="number"
                className={inputCls}
                value={architecture.cnn_extractor.kernel_size}
                min={2}
                max={5}
                onChange={(e) =>
                  onChange({
                    ...architecture,
                    cnn_extractor: { ...architecture.cnn_extractor, kernel_size: Number(e.target.value) },
                  })
                }
              />
            </Field>
            <Field label="Pooling">
              <select
                className={inputCls}
                value={architecture.cnn_extractor.pooling ? "yes" : "no"}
                onChange={(e) =>
                  onChange({
                    ...architecture,
                    cnn_extractor: { ...architecture.cnn_extractor, pooling: e.target.value === "yes" },
                  })
                }
              >
                <option value="yes">MaxPool2d after each conv</option>
                <option value="no">No pooling</option>
              </select>
            </Field>
            <Field label="FC projection dim">
              <input
                type="number"
                className={inputCls}
                value={architecture.cnn_extractor.fc_dim}
                min={4}
                max={128}
                onChange={(e) =>
                  onChange({
                    ...architecture,
                    cnn_extractor: { ...architecture.cnn_extractor, fc_dim: Number(e.target.value) },
                  })
                }
              />
            </Field>
          </>
        ) : (
          <p className="text-xs text-muted">
            Use the +/- controls on each column in the canvas to add or remove hidden layers.
            Click a column's number to resize it below.
          </p>
        )}
        <Field label="Activation">
          <select
            className={inputCls}
            value={isCnn ? architecture.cnn_extractor.activation : architecture.mlp_extractor.activation}
            onChange={(e) =>
              onChange(
                isCnn
                  ? { ...architecture, cnn_extractor: { ...architecture.cnn_extractor, activation: e.target.value as any } }
                  : { ...architecture, mlp_extractor: { ...architecture.mlp_extractor, activation: e.target.value as any } }
              )
            }
          >
            {ACTIVATIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Dropout">
          <input
            type="number"
            className={inputCls}
            step={0.05}
            min={0}
            max={0.9}
            value={isCnn ? architecture.cnn_extractor.dropout : architecture.mlp_extractor.dropout}
            onChange={(e) =>
              onChange(
                isCnn
                  ? { ...architecture, cnn_extractor: { ...architecture.cnn_extractor, dropout: Number(e.target.value) } }
                  : { ...architecture, mlp_extractor: { ...architecture.mlp_extractor, dropout: Number(e.target.value) } }
              )
            }
          />
        </Field>
        {!isCnn &&
          architecture.mlp_extractor.hidden_dims.map((dim, i) => (
            <Field key={i} label={`Layer ${i + 1} neurons`}>
              <input
                type="number"
                className={inputCls}
                min={2}
                max={128}
                value={dim}
                onChange={(e) => {
                  const next = [...architecture.mlp_extractor.hidden_dims];
                  next[i] = Number(e.target.value);
                  onChange({ ...architecture, mlp_extractor: { ...architecture.mlp_extractor, hidden_dims: next } });
                }}
              />
            </Field>
          ))}
      </div>
    );
  }

  if (selected === "projection") {
    return (
      <div className="space-y-3 p-4">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-accent">Feature Projection</h3>
        <p className="text-xs text-muted">
          Projects the extractor's output into the quantum layer's {architecture.quantum_layer.n_qubits}-dimensional
          input space.
        </p>
        <Field label="Activation">
          <select
            className={inputCls}
            value={architecture.projection.activation}
            onChange={(e) => onChange({ ...architecture, projection: { ...architecture.projection, activation: e.target.value as any } })}
          >
            {ACTIVATIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </Field>
        <label className="flex items-center gap-2 text-sm text-ink">
          <input
            type="checkbox"
            checked={architecture.projection.normalize}
            onChange={(e) => onChange({ ...architecture, projection: { ...architecture.projection, normalize: e.target.checked } })}
          />
          Scale toward [-pi, pi] for angle encoding
        </label>
      </div>
    );
  }

  if (selected === "quantum") {
    const q = architecture.quantum_layer;
    return (
      <div className="space-y-3 p-4">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-quantum">Quantum Layer</h3>
        <Field label="Qubits">
          <input
            type="number"
            className={inputCls}
            min={1}
            max={6}
            value={q.n_qubits}
            onChange={(e) => onChange({ ...architecture, quantum_layer: { ...q, n_qubits: Number(e.target.value) } })}
          />
        </Field>
        <Field label="Feature map">
          <select
            className={inputCls}
            value={q.feature_map}
            onChange={(e) => onChange({ ...architecture, quantum_layer: { ...q, feature_map: e.target.value as any } })}
          >
            <option value="ZFeatureMap">ZFeatureMap</option>
            <option value="ZZFeatureMap">ZZFeatureMap</option>
          </select>
        </Field>
        <Field label="Ansatz">
          <select
            className={inputCls}
            value={q.ansatz}
            onChange={(e) => onChange({ ...architecture, quantum_layer: { ...q, ansatz: e.target.value as any } })}
          >
            <option value="RealAmplitudes">RealAmplitudes</option>
            <option value="EfficientSU2">EfficientSU2</option>
          </select>
        </Field>
        <Field label="Ansatz depth (reps)">
          <input
            type="number"
            className={inputCls}
            min={1}
            max={5}
            value={q.ansatz_reps}
            onChange={(e) => onChange({ ...architecture, quantum_layer: { ...q, ansatz_reps: Number(e.target.value) } })}
          />
        </Field>
        <Field label="Entanglement">
          <select
            className={inputCls}
            value={q.entanglement}
            onChange={(e) => onChange({ ...architecture, quantum_layer: { ...q, entanglement: e.target.value as any } })}
          >
            <option value="linear">linear</option>
            <option value="circular">circular</option>
            <option value="full">full</option>
          </select>
        </Field>
        <Field label="QNN type">
          <select
            className={inputCls}
            value={q.qnn_type}
            onChange={(e) => onChange({ ...architecture, quantum_layer: { ...q, qnn_type: e.target.value as any } })}
          >
            <option value="estimator">EstimatorQNN</option>
            <option value="sampler">SamplerQNN</option>
          </select>
        </Field>

        {circuitError && <p className="text-xs text-danger">{circuitError}</p>}
        {circuitPreview && (
          <div className="border-t border-border pt-3">
            <dl className="mb-2 grid grid-cols-2 gap-y-1 font-mono-tech text-xs">
              <dt className="text-muted">Depth</dt>
              <dd>{circuitPreview.depth}</dd>
              <dt className="text-muted">Encoding params</dt>
              <dd>{circuitPreview.n_encoding_parameters}</dd>
              <dt className="text-muted">Trainable params</dt>
              <dd>{circuitPreview.n_trainable_parameters}</dd>
            </dl>
            <pre className="overflow-x-auto whitespace-pre bg-canvas p-2 text-[9px] leading-tight font-mono-tech text-ink">
              {circuitPreview.ascii_drawing}
            </pre>
          </div>
        )}
      </div>
    );
  }

  if (selected === "head") {
    return (
      <div className="space-y-3 p-4">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-accent">Classification Head</h3>
        <Field label="Outputs">
          <input
            type="number"
            className={inputCls}
            min={1}
            max={10}
            value={architecture.head.n_outputs}
            onChange={(e) => onChange({ ...architecture, head: { n_outputs: Number(e.target.value) } })}
          />
        </Field>
        <p className="text-xs text-muted">1 output = sigmoid/BCE binary classification; &gt;1 = softmax/cross-entropy.</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 p-4">
      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-accent">Output</h3>
      <p className="text-xs text-muted">Predicted class, derived from the head's logits.</p>
    </div>
  );
}
