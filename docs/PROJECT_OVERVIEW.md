# Project Overview

## What this is

Hybrid Studio is a research workbench for building and training genuine
**hybrid quantum-classical neural networks** — not a quantum circuit
playground, not a kernel-method app, not a VQC configuration dashboard.
The defining feature is a differentiable quantum layer (built with Qiskit
Machine Learning's `EstimatorQNN`/`SamplerQNN` wrapped by `TorchConnector`)
embedded inside an ordinary PyTorch `nn.Module`, trained end to end with
backpropagation flowing through both the classical and quantum parts in
one optimizer step.

## Why it's a separate project

It solves a different problem than Experiments 1 and 2:

| | Experiment 1 | Experiment 2 | Experiment 3 (this) |
|---|---|---|---|
| Method | Fidelity quantum kernel + SVM | Variational classifier (VQC) | Hybrid NN with a differentiable quantum layer |
| Training | Classical SVM on a precomputed kernel | A quantum circuit's own parameters, via a classical optimizer | A full PyTorch network (classical + quantum layers together) |
| Data | Tabular (2–30 features) | Tabular (2–30 features) | Images (8×8 digits, optionally MNIST/Fashion-MNIST) |
| Core question | Does a quantum kernel separate the data better than a classical one? | Can a parameterized quantum circuit alone learn to classify? | Does inserting a quantum layer improve an otherwise classical network? |

## Who it's for

Anyone who wants to run a real experiment — not a demo — testing whether a
small, simulated quantum layer changes end-to-end neural network behavior,
and to see that difference measured honestly against classical baselines
with a comparable parameter budget.
