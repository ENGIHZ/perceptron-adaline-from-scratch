# Perceptron & ADALINE from Scratch | Machine Learning for Engineering

A reproducible educational comparison of two linear classifiers implemented **from scratch with NumPy**, using the Iris dataset as a **proxy for sensor measurements**. The project explores convergence, feature scaling, decision boundaries, and resilience to simulated measurement noise.

> **Scope:** This is an academic experiment, not a deployed industrial inspection system. The Iris measurements are not actual industrial sensor readings.

## Highlights

- Custom **Perceptron** and **ADALINE (batch gradient descent)** implementations, without prebuilt classifiers.
- Binary Setosa vs. Versicolor classification using sepal and petal lengths.
- Controlled comparisons of learning rates and standardized vs. raw features.
- Hold-out test evaluation, confusion matrices, and decision boundaries.
- Gaussian perturbations applied to standardized test inputs at σ = 0.10, 0.25, 0.50, with 30 repetitions per level and **no retraining**.
- Bonus three-class One-vs-Rest extension using the custom Perceptron.

## Results from the supplied report

| Experiment | Result |
|---|---:|
| Binary Perceptron test accuracy | 100% (30 test samples) |
| Binary ADALINE test accuracy | 100% (30 test samples) |
| Perceptron mean accuracy at σ = 0.50 | 95.56% |
| ADALINE mean accuracy at σ = 0.50 | 97.22% |
| Three-class OVR test accuracy | 82.22% (37/45) |
| Three-class OVR macro-F1 | 81.53% |

The clean binary task is small and relatively easy. Perfect performance on this split **does not** imply perfect real-world generalization. Differences in noise robustness are specific to these experiments.

## Results & Visualizations

### Decision Boundaries
![Decision Boundaries](figures/decision_boundaries.png)

### Training Curves
![Training Curves](figures/training_curves.png)

### Sensor Noise Robustness
![Sensor Noise Robustness](figures/noise_robustness.png)

### Confusion Matrices
![Confusion Matrices](figures/confusion_matrices.png)

## Repository structure

```text
perceptron-adaline-from-scratch/
├── README.md
├── perceptron_adaline_iris.ipynb
├── requirements.txt
├── report.pdf
├── figures/
│   ├── training_curves_perceptron.png
│   ├── training_curves_adaline.png
│   └── sensor_noise_robustness.png
└── .gitignore
```

## Run locally

1. Install Python 3 and create a virtual environment (recommended).
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open `perceptron_adaline_iris.ipynb` in Jupyter or VS Code (or upload to Google Colab).
4. Run all cells in order. The notebook uses random seed **42**.

The notebook attempts to download Iris from UCI; if unavailable, it uses scikit-learn's packaged copy and announces the fallback. To reproduce the official UCI download path, run with internet access.

## Engineering takeaways

Feature scaling affects convergence, especially for ADALINE. Real measurement systems may also face calibration drift, distribution shifts, and noisy inputs. The simulated noise experiment is a first robustness check, **not** a substitute for testing with real sensor data.

## Background

Created as a programming assignment for **AI4201 — Machine Learning in Engineering**. The notebook and figures are based on the original submitted work. Identifying student numbers have been removed from the public-facing copies.

**Dataset:** [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris).
