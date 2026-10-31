# Architecture

## System layout

```
React frontend (Architecture Studio, Training Control Room, Prediction
Gallery, Benchmark Workspace)
        │  REST + polling
        ▼
FastAPI backend
        │
        ├── image_dataset_service     (digits / MNIST / Fashion-MNIST)
        ├── preprocessing_service     (normalize, flatten, PCA — train-only fit)
        ├── classical_model_service   (MLP / CNN feature extractors + heads)
        ├── quantum_layer_service     (feature map + ansatz + EstimatorQNN/SamplerQNN)
        ├── hybrid_model_service      (composes classical + TorchConnector quantum layer)
        ├── training_service          (PyTorch training/validation loop, background thread)
        ├── evaluation_service        (metrics, confusion matrix, per-class report)
        ├── prediction_service        (per-image predictions, confidence, correctness)
        ├── circuit_analysis_service  (theoretical vs. actual resource summary)
        ├── checkpoint_service        (save/restore model state_dict)
        └── experiment_service        (orchestration + SQLite persistence)
```

## The hybrid model, concretely

```
image (8x8 or NxN)
   -> classical feature extractor (MLP or small CNN)
   -> feature projection (Linear -> activation -> normalize) into n_qubits dims
   -> quantum layer: EstimatorQNN(feature_map, ansatz) wrapped in TorchConnector,
      used as an ordinary nn.Module inside the network
   -> classification head (Linear -> sigmoid/softmax)
   -> prediction
```

`TorchConnector` makes the QNN's forward/backward passes participate in
the same `loss.backward()` / `optimizer.step()` call as every classical
layer — there is no separate quantum training stage. This is verified
directly by a test that checks the quantum layer's `weight.grad` is
non-zero and that its parameters change after an optimizer step (see
`backend/app/tests/test_hybrid_model_service.py`).

## Persistence

SQLite (`data/workbench.db`) stores the full experiment record: dataset
config, preprocessing config, architecture definition (classical + quantum
layer configs), training config, per-epoch training history, final
metrics, predictions, checkpoint references, random seeds, and software
versions — mirroring the same lifecycle-and-persistence pattern used in
Experiments 1 and 2, reimplemented fresh for this app's schema.
