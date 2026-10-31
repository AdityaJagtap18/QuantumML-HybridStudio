export default function Documentation() {
  return (
    <div className="p-6">
      <div className="mb-6 border-b border-border pb-4">
        <h1 className="text-lg font-semibold tracking-tight text-ink">Documentation</h1>
        <p className="mt-1 text-sm text-muted">What a hybrid quantum-classical network is, and how this app trains one.</p>
      </div>
      <div className="max-w-3xl space-y-4 border border-border border-t-2 border-t-accent bg-surface p-5 text-sm leading-relaxed text-ink">
        <p>
          <strong>The architecture.</strong> image → classical feature extractor (MLP or CNN) →
          feature projection into n_qubits dimensions → a quantum layer (EstimatorQNN or
          SamplerQNN, wrapped by Qiskit Machine Learning's TorchConnector) → a classical
          classification head.
        </p>
        <p>
          <strong>Why TorchConnector matters.</strong> It makes the quantum circuit an ordinary
          differentiable PyTorch module: one <code>loss.backward()</code> and one
          <code> optimizer.step()</code> update both the classical layers and the quantum
          circuit's trainable parameters together. This app does not train a VQC separately and
          append its output to a classical model — that would not be a genuine hybrid network.
        </p>
        <p>
          <strong>How we verify it's real.</strong> A dedicated test checks that the quantum
          layer's parameters receive non-zero gradients and actually change after an optimizer
          step. A successful forward pass is never treated as proof of trainability on its own.
        </p>
        <p>
          <strong>Research integrity.</strong> Preprocessing is fit on the training split only.
          Classical baselines and hybrid models always share the same split, preprocessing, and
          evaluation protocol. Synthetic noise is always labeled as synthetic, never as a
          hardware calibration. No metric, loss value, or circuit statistic is ever invented.
        </p>
      </div>
    </div>
  );
}
