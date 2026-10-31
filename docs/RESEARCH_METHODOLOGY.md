# Research Methodology

## Core question

Can a differentiable quantum layer contribute meaningfully to an
end-to-end classical neural network?

## Rules this project holds itself to

1. **No fabricated numbers.** Every metric, loss value, circuit
   statistic, and prediction shown anywhere in the UI comes from an
   actual execution against the configured data and model.
2. **No leakage.** All preprocessing (normalization, PCA) is fit on the
   training split only, then applied to validation/test.
3. **Fair comparison.** Classical baselines (MLP, CNN) and hybrid models
   (MLP+QNN, CNN+QNN) always train on the identical split, preprocessing,
   and evaluation protocol. Where practical, a classical control with a
   comparable parameter budget is included so the quantum layer's
   contribution can be isolated from "the model just got bigger."
4. **Gradient flow is verified, not assumed.** A successful forward pass
   through the quantum layer is not treated as proof it's trainable. A
   dedicated test asserts the quantum layer's parameters receive non-zero
   gradients and actually change after an optimizer step.
5. **No fake progress.** Training progress is reported as real epoch/
   batch counts and real loss/accuracy values — never an invented
   percentage.
6. **Noise is never "hardware."** Any noisy-simulation mode uses an
   explicitly synthetic, user-configured Aer noise model and is labeled
   as such — never described as calibrated to real hardware.
7. **No automatic claims of quantum advantage.** The research
   interpretation shown after a benchmark run states whichever model
   actually won, including when a classical baseline beats every hybrid
   variant.

## Planned studies (docs/EXPERIMENTS.md tracks status)

- **A — Classical vs. hybrid**: MLP, CNN, MLP+QNN, CNN+QNN on the same data.
- **B — Qubit count**: same architecture, varying number of qubits.
- **C — Ansatz depth**: shallow vs. deep ansatz, same qubit count.
- **D — Encoding strategy**: compare supported feature-map encodings.
- **E — Quantum layer contribution**: hybrid vs. a parameter-matched classical control.
- **F — Noise sensitivity**: ideal vs. synthetic noisy simulation, train-time and test-time effects kept distinct.
