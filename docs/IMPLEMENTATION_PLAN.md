# Hybrid Studio — Implementation Plan

## 0. Scope

Experiment 3 of the QuantumML research portfolio: a genuine hybrid
quantum-classical neural network, trained end to end in PyTorch with a
differentiable quantum layer (Qiskit Machine Learning's
`EstimatorQNN`/`SamplerQNN` wrapped by `TorchConnector`). Independent
repository (`QuantumML-HybridStudio`, github.com/AdityaJagtap18/QuantumML-HybridStudio),
no shared code with QuantumCloud, the Experiment 1 kernel workbench, or
the Experiment 2 VQC workbench.

## 1. Installed versions and verified API

- `torch` 2.15.0 (CPU), `qiskit` 2.6.1, `qiskit-machine-learning` 0.9.1
  (same qiskit stack version as Experiments 1 and 2).
- `EstimatorQNN(circuit, estimator, observables, input_params,
  weight_params, gradient, input_gradients=False, ...)` — **note the
  `input_gradients` flag defaults to `False`**. It must be set `True`
  for a quantum layer embedded mid-network, otherwise gradients cannot
  flow backward through it to earlier classical layers (confirmed by
  inspecting the signature directly, not assumed from tutorials — many
  older examples only use the quantum layer as the first/last layer,
  where this doesn't matter).
- `SamplerQNN(circuit, sampler, input_params, weight_params, sparse,
  interpret, output_shape, gradient, input_gradients=False, ...)` — same
  flag, same caveat.
- `TorchConnector(neural_network, initial_weights, sparse)` — wraps
  either QNN as an ordinary `nn.Module`.
- **Verified empirically** (not assumed): built a tiny
  `Linear(4,2) -> Tanh -> TorchConnector(EstimatorQNN) -> Linear(1,1)`
  network with `input_gradients=True`, ran one
  `loss.backward()`, and confirmed (a) the quantum layer's weight
  tensor receives a non-zero gradient, (b) the *preceding* classical
  `Linear` layer's weight also receives a non-zero gradient (proving
  gradients actually propagate backward through the quantum layer, not
  just locally), and (c) three `optimizer.step()` calls measurably
  change the quantum layer's weight values. This is the exact check
  `test_hybrid_model_service.py` automates.

## 2. What's different from Experiments 1 and 2

- **Data**: images (digits 8x8, not tabular features).
- **Model**: a full PyTorch `nn.Module` containing both classical layers
  and a quantum layer, trained with one optimizer over all parameters —
  not a classifier trained purely on a quantum circuit's own output.
- **Core verification burden**: proving the quantum layer's parameters
  actually receive gradients and update — forward-pass success alone is
  not accepted as evidence.
- **UI**: a different application shell entirely (top bar + icon-only
  project rail + central workspace, image gallery, node-based
  architecture view, training control room, prediction gallery,
  benchmark workspace) — no reused layouts, cards, or navigation
  structure from the other two apps.

## 3. Repository layout

```
backend/app/{main.py, api/routes/, core/, schemas/, models/, services/, workers/, tests/}
frontend/src/{components/{workspace,architecture-canvas,model-nodes,property-inspector,image-gallery,training-control,prediction-gallery,benchmarking,charts,quantum-circuit}/, pages/, services/, hooks/, types/, utils/}
experiments/{baseline_models, hybrid_models, research_runs}
data/, artifacts/{checkpoints,plots,reports}/
docs/
```

## 4. Experiment lifecycle

`QUEUED -> VALIDATING -> PREPROCESSING -> BUILDING_MODEL -> TRAINING ->
EVALUATING -> COMPLETED` (+ `FAILED`, `CANCELLED`). Background execution
via FastAPI `BackgroundTasks` running the PyTorch training loop in a
thread, status + live per-epoch history polled from a SQLite-backed
`ExperimentRecord` — same lifecycle/persistence pattern as Experiments 1
and 2, reimplemented fresh for this app's schema (architecture config,
training history with train/val loss+accuracy per epoch, checkpoint
path).

## 5. Phases

### Phase 1 — Foundation (in progress)
Independent FastAPI + React setup. `image_dataset_service` loads the
scikit-learn digits dataset (8x8, 10 classes, binary subsets like 0 vs 1
by default); MNIST/Fashion-MNIST are listed in the catalog as
not-yet-available rather than silently absent. `preprocessing_service`
normalizes and optionally flattens/PCA-reduces, fit on train only, shared
identically by classical and hybrid paths. Image Explorer UI: real digit
images rendered on canvas, class filter, pixel inspection, class
distribution, train/val/test counts — no placeholder imagery.

### Phase 2 — Classical models
`classical_model_service`: configurable MLP and small CNN feature
extractors + a linear classification head, plain PyTorch modules reused
by both classical baselines and the hybrid path. A `training_service`
runs the PyTorch train/validation loop (Adam/SGD, configurable epochs,
batch size, early stopping) and `checkpoint_service` saves/restores
`state_dict`s under `artifacts/checkpoints/`.

### Phase 3 — Quantum integration (critical path)
`quantum_layer_service` builds the feature map + ansatz circuit and wraps
an `EstimatorQNN`/`SamplerQNN` in `TorchConnector` as an ordinary
`nn.Module`. `hybrid_model_service` composes
`extractor -> projection -> quantum layer -> head` into one network.
**A dedicated test verifies the quantum layer's parameters receive
non-zero gradients and change after an optimizer step** — a successful
forward pass is never treated as proof of end-to-end trainability.

### Phase 4 — Architecture Studio
The node-based architecture view (inspired by a TensorFlow-Playground-style
connected-layer diagram and a circuit-composer-style property panel):
left component list, center node graph showing the actual configured
architecture (shapes, parameter counts), right inspector for the
selected node's configuration, with a live circuit preview for the
quantum layer node.

### Phase 5 — Training Control Room
Dedicated training monitor: architecture summary, live train/val loss and
accuracy charts from real per-epoch history, epoch timeline, runtime,
logs, cancellation. No invented progress percentages.

### Phase 6 — Prediction Analysis
Image-centric evaluation: thumbnail grid with actual vs. predicted label
and confidence, correct/incorrect filtering, click-through detail view.

### Phase 7 — Benchmark Workspace
Classical MLP/CNN vs. hybrid MLP+QNN/CNN+QNN: accuracy, F1, timing,
parameter count, training curves, confusion matrices, per-class
performance — all from actual completed runs, with an honest
research-interpretation summary (never an automatic claim of hybrid
superiority).

### Phase 8 — Research extensions
Qubit-count, ansatz-depth, and encoding-strategy studies; a synthetic
(explicitly non-hardware) Aer noise mode; repeated-seed statistics.

### Phase 9 — Quality
Full pytest suite (including the gradient-propagation test), responsive
refinement, export (JSON/CSV/training history/circuit diagram/Markdown
report), docs, README polish.

## 6. Research-integrity rules (enforced throughout)

No fabricated metrics, losses, circuit stats, or training progress.
Preprocessing fit on train only. Classical baselines and hybrid models
always share split/preprocessing/protocol. Gradient flow through the
quantum layer is tested directly, not assumed. Noise is always labeled
synthetic. No automatic claim of quantum advantage — the benchmark
reports whichever model actually won.

## 7. Status tracking

Each phase ends with: tests run (including, from Phase 3 onward, the
gradient-propagation test), frontend/backend integration verified live in
the browser, this document updated, and a short report of what's done vs.
outstanding.
