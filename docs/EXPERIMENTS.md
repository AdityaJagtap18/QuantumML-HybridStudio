# Experiments Log

Tracks which of the research studies defined in
[RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md) are runnable in the
app, and any findings worth recording. Updated as phases land.

| Study | Runnable in UI | Notes |
|---|---|---|
| A — Classical vs. hybrid | Yes | Architecture Studio model-kind selector + Benchmark Workspace |
| B — Qubit count | Yes | Quantum Layer inspector; compare runs in Benchmark Workspace |
| C — Ansatz depth | Yes | "Ansatz depth (reps)" field in the Quantum Layer inspector |
| D — Encoding strategy | Yes | Feature map / entanglement fields in the Quantum Layer inspector |
| E — Quantum layer contribution | Yes | Benchmark Workspace — picks whichever model actually won, no automatic hybrid-superiority claim |
| F — Noise sensitivity | Yes | Quantum Layer inspector's Execution = Ideal/Noisy toggle, with configurable depolarizing/readout error and shot count. Verified live: a noisy hybrid run completed end to end with the real Aer noise model applied (`execution_mode: noisy` confirmed in the persisted config). |

Repeated-seed statistics (automated mean/std across seeds) are not yet
automated — `training.random_seed` can be changed manually between runs
and compared in the Benchmark Workspace today.

No findings recorded yet — these are capability checks, not research
conclusions. Run real experiments to populate this log.
