# Data

This directory holds locally cached datasets and the SQLite experiment
database. Nothing here is committed to Git except this file — datasets are
loaded at runtime via scikit-learn / torchvision and cached here, and the
experiment database (`workbench.db`) is a local run artifact.

- **Digits** (primary): scikit-learn's built-in 8×8 handwritten digit
  dataset — no download required.
- **MNIST subset** / **Fashion-MNIST subset** (secondary): fetched via
  `torchvision.datasets` on first use and cached here.
