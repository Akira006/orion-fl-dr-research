# Orion Research: Diabetic Retinopathy

This project focuses on the application of Federated Learning (FL) for diabetic retinopathy image classification. The goal is to explore how different client sequencing strategies affect model performance, robustness, and generalization compared to centralized training.

---

## 📌 Objectives

The main objectives of this research are:

* To compare centralized learning and federated learning approaches in medical image classification.
* To investigate the impact of client sequencing strategies in federated learning.
* To evaluate robustness under different data distributions and client characteristics.
* To analyze whether ordering clients based on loss or dataset size influences convergence and performance.

---

## 🧪 Experiment Design

This project is structured into several controlled experiments:

1. **Baseline (Centralized Learning)**
   Training is performed on the entire dataset without any federated setup.

2. **Federated Learning – Loss Ascending**
   Clients are trained in order from lowest loss to highest loss (easy → hard).

3. **Federated Learning – Loss Descending**
   Clients are trained in order from highest loss to lowest loss (hard → easy).

4. **Federated Learning – Dataset Size Ascending**
   Clients are trained from smallest dataset to largest dataset.

5. **Federated Learning – Dataset Size Descending**
   Clients are trained from largest dataset to smallest dataset.

---

## 📁 Project Structure

```
orion-fl-dr-research/
│
├── notebooks/              # Experiment notebooks
│   ├── exp1_baseline.ipynb
│   ├── exp2_fl_loss_asc.ipynb
│   ├── exp3_fl_loss_desc.ipynb
│   ├── exp4_fl_size_asc.ipynb
│   └── exp5_fl_size_desc.ipynb
│
├── src/                    # Core implementation
│   ├── models/
│   ├── federated/
│   ├── training/
│   └── utils/
│
├── configs/                # Configuration files
├── results/                # Logs and outputs
│
└── README.md
```

---

## ⚙️ Workflow

The development workflow follows a hybrid setup:

* **VSCode** is used for development and code structuring.
* **Google Colab** is used for running experiments with GPU support.
* **GitHub** acts as the central synchronization and version control system.

Typical workflow:

```
VSCode → commit → GitHub → Colab → run → save → GitHub → pull → VSCode
```

---

## 🧠 Key Research Focus

This research explores the hypothesis that:

> The order of client participation in federated learning affects the learning dynamics and final model performance.

By systematically comparing different sequencing strategies, this project aims to provide insights into optimizing federated learning pipelines for medical imaging tasks.

---

## 🚀 Future Work

* Integration of noise-aware training
* Robust aggregation methods
* Domain adaptation (e.g., DANN)
* Comparison with additional CNN architectures

---

## 📌 Notes

* Large datasets and model weights are excluded from the repository.
* Experiments are tracked through notebooks and logs.
* Reproducibility is maintained via structured experiments and version control.

---
