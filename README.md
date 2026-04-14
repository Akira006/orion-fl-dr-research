# Orion Research: Diabetic Retinopathy

This project focuses on the application of Federated Learning (FL) for diabetic retinopathy image classification. The study investigates how different client sequencing strategies influence model performance, convergence behavior, robustness, and generalization compared to a centralized baseline.

---

## 📌 Objectives

The main objectives of this research are:

- To establish a centralized learning baseline for diabetic retinopathy image classification.
- To compare centralized learning and federated learning approaches in medical image classification.
- To investigate the impact of client sequencing strategies in federated learning.
- To evaluate whether ordering clients based on loss or dataset size affects convergence and overall performance.
- To build a structured experimental workflow that supports future extensions such as robust aggregation and domain adaptation.

---

## 🧪 Experiment Design

This project is organized into several controlled experiments on the **APTOS** dataset:

1. **EDA (Exploratory Data Analysis)**  
   Preliminary dataset inspection, label distribution analysis, sample visualization, and validation of the global 80:20 split.

2. **Baseline (Centralized Learning)**  
   Training is performed using a global 80:20 train-validation split on the full dataset, without federated learning.

3. **Federated Learning – Loss Ascending**  
   Clients are trained in order from lowest loss to highest loss *(easy → hard)*.

4. **Federated Learning – Loss Descending**  
   Clients are trained in order from highest loss to lowest loss *(hard → easy)*.

5. **Federated Learning – Dataset Size Ascending**  
   Clients are trained from the smallest dataset to the largest dataset.

6. **Federated Learning – Dataset Size Descending**  
   Clients are trained from the largest dataset to the smallest dataset.

---

## 📁 Project Structure

```text
orion-fl-dr-research/
│
├── notebooks/                          # Experiment and analysis notebooks
│   └── APTOS/
│       ├── eda_aptos.ipynb
│       ├── exp1_baseline.ipynb
│       ├── exp2_fl_loss_asc.ipynb
│       ├── exp3_fl_loss_desc.ipynb
│       ├── exp4_fl_size_asc.ipynb
│       └── exp5_fl_size_desc.ipynb
│
├── src/                                # Core implementation
│   ├── models/
│   ├── federated/
│   ├── training/
│   └── utils/
│
├── configs/                            # Configuration files
│
├── results/                            # Logs, figures, and experiment outputs
│   └── APTOS/
│       ├── eda_aptos/
│       │   ├── figures/
│       │   └── logs/
│       │
│       ├── exp1_baseline/
│       │   └── mobilenet/
│       │       ├── figures/
│       │       ├── logs/
│       │       └── models/
│       │
│       ├── exp2_fl_loss_asc/
│       │   └── mobilenet/
│       │       ├── figures/
│       │       ├── logs/
│       │       └── models/
│       │
│       ├── exp3_fl_loss_desc/
│       │   └── mobilenet/
│       │       ├── figures/
│       │       ├── logs/
│       │       └── models/
│       │
│       ├── exp4_fl_size_asc/
│       │   └── mobilenet/
│       │       ├── figures/
│       │       ├── logs/
│       │       └── models/
│       │
│       └── exp5_fl_size_desc/
│           └── mobilenet/
│               ├── figures/
│               ├── logs/
│               └── models/
│
└── README.md