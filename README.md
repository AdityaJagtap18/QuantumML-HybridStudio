# Hybrid Studio

**An interactive research workbench for hybrid quantum-classical neural
networks using Qiskit Machine Learning, PyTorch, and differentiable quantum
layers.**

Hybrid Studio is Experiment 3 of a QuantumML research portfolio. It is a
standalone project — independent of QuantumCloud, the Experiment 1 kernel
workbench, and the Experiment 2 VQC workbench, with no shared code, design
system, or Git history.

## Research question

> Can a differentiable quantum layer contribute meaningfully to an
> end-to-end classical neural network?

Hybrid Studio builds genuine hybrid architectures —
`image → classical feature extractor → feature projection → quantum neural
network layer (TorchConnector) → classification head` — and trains them
end to end in PyTorch, verifying that gradients actually flow through the
quantum layer rather than training a VQC separately and bolting it on.

## Status

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the
phase-by-phase build plan, verified package versions, and what's
implemented so far.

## Project layout

```
backend/        FastAPI + PyTorch + Qiskit Machine Learning
frontend/       React + TypeScript + Vite + Tailwind
experiments/    Saved baseline/hybrid model definitions and research runs
data/           Local dataset cache + experiment database (not versioned)
artifacts/      Checkpoints, plots, reports (not versioned)
docs/           Project overview, architecture, methodology, plan
```

## Running locally

Backend:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server (port 5175) proxies `/api` to `http://127.0.0.1:8002`.

## Tests

```bash
cd backend
source .venv/bin/activate
pytest
```
