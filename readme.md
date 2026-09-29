# ML from scratch

Functioning prototypes of machine learning algorithms, built in NumPy using OOP principles. The goal is to understand the internals (closed-form solutions, backpropagation, gradient descent) rather than wrap an existing library.

**Status: work in progress.** The table below reflects what currently runs.

## What's implemented

| Component | Location | Status |
|---|---|---|
| Linear regression, closed-form (SLR and MLR via one `OLS` class) | `classic_algorithms/ols.py` | Working |
| Neural network framework: `Layer` base class, `Dense` layer, `Sequential` container, `MSELoss`, with forward and backward passes | `nn/` | Working |
| Linear regression via gradient descent (a single `Dense` layer trained with MSE) | `demos/SLR_demo.py`, `demos/MLR_demo.py` | Working; matches the closed-form OLS result |
| Euclidean distance, train/test split | `utils/` | Working |
| K-Nearest Neighbors | `classic_algorithms/knn.py` | In progress; not yet runnable |

## Demos

The SLR and MLR demos fit the same synthetic data two ways, with gradient descent through the `nn` framework and with closed-form OLS, and print both sets of coefficients. They agree to about three decimal places. `SLR_demo.py` also plots both fitted lines.

Run from the repository root so the packages import correctly:

```
PYTHONPATH=. python demos/SLR_demo.py
PYTHONPATH=. python demos/MLR_demo.py
```

Requires `numpy` and `matplotlib`.

## Known issues

- **KNN is not yet implemented and does not run yet.**
- **`OLS` uses `np.linalg.inv`**, so it raises an error on perfectly collinear features. `lstsq` or `pinv` would handle this.
- **Weight updates in the demos are written by hand** inside the training loop. `nn/optim.py` is a placeholder for an optimizer module.

## Roadmap

Planned: KNN (fix and finish implementation), K-Means, PCA, logistic regression, activation functions, cross-entropy loss, optimizers, MLP, CNN.

Out of scope: RNNs, ResNets/U-Nets, Transformers.
