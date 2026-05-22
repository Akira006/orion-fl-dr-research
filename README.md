# Experiment Report

**Project:** Client Sequencing Strategy in Federated Domain Incremental Learning for Diabetic Retinopathy Classification
**Repository:** [Akira006/orion-fl-dr-research](https://github.com/Akira006/orion-fl-dr-research)
**Researcher:** Akira
**Last Updated:** May 2026

---

## 1. Overview

This report documents the experimental setup, methods, parameters, and results of a federated learning (FL) research project focused on diabetic retinopathy (DR) image classification. The study investigates how **client sequencing strategies** — the order in which clients participate in each FL round — affect model convergence, generalization, and classification performance across multiple heterogeneous datasets.

The project further extends the FL framework with **Domain-Adversarial Neural Networks (DANN)** to address domain shift between datasets, and compares two backbone architectures: **MobileNetV2** and **EfficientNetB0**.

---

## 2. Research Objectives

1. Establish a **centralized pooled baseline** as an upper-bound reference for FL comparison
2. Evaluate the impact of **client sequencing strategies** (loss-based and size-based) on FL convergence and performance
3. Assess the effectiveness of **DANN-based domain adaptation** in a federated setting
4. Compare **MobileNetV2** vs **EfficientNetB0** as backbone architectures across all experiment variants
5. Analyze **per-client performance** and **domain shift** across three heterogeneous DR datasets

---

## 3. Datasets

Three publicly available diabetic retinopathy datasets are used, each treated as an independent FL client:

| Client | Dataset | Total Samples | Label Source | NPZ Key (Images) |
|--------|---------|--------------|--------------|-----------------|
| Client 0 | **DDR** | 12,522 | `DR_grading.csv` → `diagnosis` | `X` |
| Client 1 | **EyePACS** | 35,126 | `trainLabels.csv` → `level` | `images` |
| Client 2 | **APTOS 2019** | 3,662 | NPZ internal | `images` |
| — | **Total** | **51,310** | — | — |

### Label Schema (DR Severity)

All three datasets share a consistent 5-class label scheme:

| Class | Description |
|-------|-------------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

### Class Distribution (Approximate)

| Class | DDR | EyePACS | APTOS | Pooled |
|-------|-----|---------|-------|--------|
| 0 | ~50% | ~73% | ~49% | ~66% |
| 1 | ~7% | ~7% | ~8% | ~7% |
| 2 | ~21% | ~15% | ~24% | ~21% |
| 3 | ~3% | ~2% | ~5% | ~3% |
| 4 | ~4% | ~3% | ~5% | ~4% |

> **Note:** All datasets are heavily imbalanced toward Class 0 (No DR). Class weights (`balanced`) are applied per client during training to mitigate this.

### Data Split Strategy

- **Centralized (exp1):** Global pooled 70:15:15 stratified split across all datasets
- **Federated (exp2-9):** Per-client 70:15:15 stratified split — global val/test set = concatenation of all client val/test splits

```
Per-client approximate split:
  DDR     — train: 8,765  | val: 1,878 | test: 1,879
  EyePACS — train: 24,588 | val: 5,269 | test: 5,269
  APTOS   — train: 2,563  | val:   549 | test:   550
  ──────────────────────────────────────────────────
  Global val  : ~7,696 samples
  Global test : ~7,698 samples
```

---

## 4. Experiment Schema

A total of **18 experiments** are conducted, covering 2 backbone architectures × 9 experiment configurations:

### 4.1 MobileNetV2 Experiments

| ID | Experiment | Type | Sort Strategy | FL Rounds | DANN |
|----|-----------|------|--------------|-----------|------|
| exp1 | `exp1_centralized_multiclient` | Centralized Pooled | — | — | ❌ |
| exp2 | `exp2_fl_loss_asc_multiclient` | Federated Learning | Loss Ascending | 15 | ❌ |
| exp3 | `exp3_fl_loss_desc_multiclient` | Federated Learning | Loss Descending | 15 | ❌ |
| exp4 | `exp4_fl_size_asc_multiclient` | Federated Learning | Size Ascending | 15 | ❌ |
| exp5 | `exp5_fl_size_desc_multiclient` | Federated Learning | Size Descending | 15 | ❌ |
| exp6 | `exp6_fl_dann_loss_asc_multiclient` | FL + DANN | Loss Ascending | 15 | ✅ |
| exp7 | `exp7_fl_dann_loss_desc_multiclient` | FL + DANN | Loss Descending | 15 | ✅ |
| exp8 | `exp8_fl_dann_size_asc_multiclient` | FL + DANN | Size Ascending | 15 | ✅ |
| exp9 | `exp9_fl_dann_size_desc_multiclient` | FL + DANN | Size Descending | 15 | ✅ |

### 4.2 EfficientNetB0 Experiments

| ID | Experiment | Type | Sort Strategy | FL Rounds | DANN |
|----|-----------|------|--------------|-----------|------|
| exp1 | `exp1_centralized_multiclient_efficientnetb0` | Centralized Pooled | — | — | ❌ |
| exp2 | `exp2_fl_loss_asc_multiclient_efficientnetb0` | Federated Learning | Loss Ascending | 15 | ❌ |
| exp3 | `exp3_fl_loss_desc_multiclient_efficientnetb0` | Federated Learning | Loss Descending | 15 | ❌ |
| exp4 | `exp4_fl_size_asc_multiclient_efficientnetb0` | Federated Learning | Size Ascending | 15 | ❌ |
| exp5 | `exp5_fl_size_desc_multiclient_efficientnetb0` | Federated Learning | Size Descending | 15 | ❌ |
| exp6 | `exp6_fl_dann_loss_asc_multiclient_efficientnetb0` | FL + DANN | Loss Ascending | 15 | ✅ |
| exp7 | `exp7_fl_dann_loss_desc_multiclient_efficientnetb0` | FL + DANN | Loss Descending | 15 | ✅ |
| exp8 | `exp8_fl_dann_size_asc_multiclient_efficientnetb0` | FL + DANN | Size Ascending | 15 | ✅ |
| exp9 | `exp9_fl_dann_size_desc_multiclient_efficientnetb0` | FL + DANN | Size Descending | 15 | ✅ |

### 4.3 Client Sequencing Strategy Details

**Loss-based (dynamic per round):**
- Loss Ascending → client with lowest validation loss trains first (easy → hard)
- Loss Descending → client with highest validation loss trains first (hard → easy)
- Order re-evaluated each round based on current global model's label loss on each client's train data

**Size-based (fixed across all rounds):**
- Size Ascending → APTOS (2.5K) → DDR (8.7K) → EyePACS (24.6K)
- Size Descending → EyePACS (24.6K) → DDR (8.7K) → APTOS (2.5K)
- Order fixed at experiment start, does not change per round

---

## 5. Methods

### 5.1 Federated Learning — FedAvg

Standard **Federated Averaging (FedAvg)** is used for global model aggregation. After each round of local training, client model weights are averaged proportionally by dataset size:

$$w_{global} = \sum_{i=1}^{N} \frac{n_i}{n_{total}} \cdot w_i$$

Where $n_i$ is the number of training samples for client $i$ and $w_i$ is the locally trained model weights.

### 5.2 Two-Stage Local Training

Each client performs two-stage fine-tuning per FL round:

| Stage | Rounds | Backbone | Local Epochs | LR |
|-------|--------|----------|-------------|-----|
| Stage 1 | Round 1–7 | Frozen | 3 | 1e-3 |
| Stage 2 | Round 8–15 | Partially unfrozen (last 10 layers) | 3 | 1e-6 |

BatchNormalization layers remain frozen throughout both stages to prevent distribution shift.

### 5.3 Domain-Adversarial Neural Network (DANN)

For exp6-9, DANN is applied at the **server (global model) level**. The architecture consists of three components:

```
Input Image
    ↓
Feature Extractor (EfficientNetB0/MobileNetV2 backbone + shared 128-dim feature space)
    ↓              ↓
Label Classifier  Domain Classifier (via Gradient Reversal Layer)
    ↓              ↓
DR Class (0-4)   Dataset Origin (DDR=0 / EyePACS=1 / APTOS=2)
```

**Total loss:**
$$\mathcal{L}_{total} = \mathcal{L}_{label} + \lambda \cdot \mathcal{L}_{domain}$$

Where domain loss weight = **0.2** and $\lambda$ follows a **progressive schedule**:

$$\lambda(p) = \frac{2}{1 + e^{-\gamma p}} - 1, \quad p = \frac{\text{round}}{\text{total\_rounds}}, \quad \gamma = 10$$

**Progressive lambda schedule (15 rounds):**

| Round | p | λ |
|-------|---|---|
| 1 | 0.07 | 0.1311 |
| 3 | 0.20 | 0.4621 |
| 5 | 0.33 | 0.6900 |
| 8 | 0.53 | 0.8761 |
| 10 | 0.67 | 0.9354 |
| 12 | 0.80 | 0.9677 |
| 15 | 1.00 | 0.9933 |

### 5.4 Model Architectures

| Component | MobileNetV2 | EfficientNetB0 |
|-----------|------------|----------------|
| Backbone parameters | ~2.26M | ~4.05M |
| Input preprocessing | `preprocess_input()` (range -1 to 1) | `float32 0-255` (internal normalization) |
| Global pooling | GlobalAveragePooling2D | GlobalAveragePooling2D |
| Dropout | 0.4 | 0.4 |
| Head | Dense(5, softmax) | Dense(5, softmax) |
| L2 regularization | 1e-4 | 1e-4 |

**DANN-specific additions (exp6-9):**
- Shared feature space: Dense(128, relu) + BatchNormalization + Dropout(0.3)
- Label classifier: Dense(64, relu) → Dense(5, softmax)
- Domain classifier: GRL → Dense(64, relu) → Dense(3, softmax)

---

## 6. Parameters & Variables

### 6.1 General Config

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 × 3 |
| Batch size | 32 |
| Num classes | 5 |
| Num clients | 3 |
| Random seed | 42 |
| Framework | TensorFlow / Keras |
| Runtime | Google Colab A100 GPU (High RAM) |

### 6.2 Training Config

| Parameter | Centralized | Federated (non-DANN) | Federated (DANN) |
|-----------|------------|---------------------|-----------------|
| Stage 1 epochs/rounds | 15 epochs | 15 rounds × 3 local epoch | 15 rounds × 3 local epoch |
| Stage 2 epochs/rounds | 15 epochs | 15 rounds × 3 local epoch | 15 rounds × 3 local epoch |
| Stage 1 LR | 1e-3 | 1e-3 | 1e-3 |
| Stage 2 LR | 1e-6 | 1e-6 | 1e-6 |
| Unfreeze last N layers | 10 | 10 | 10 |
| Class weight | balanced | balanced (per client) | balanced (per client) |

### 6.3 DANN-Specific Config

| Parameter | Value |
|-----------|-------|
| Feature dim | 128 |
| Domain loss weight | 0.2 |
| Lambda gamma (γ) | 10.0 |
| Lambda schedule | Progressive (Ganin et al., 2016) |
| Num domains | 3 (DDR, EyePACS, APTOS) |

### 6.4 Data Augmentation

Applied during training only (not validation/test):

| Augmentation | Value |
|-------------|-------|
| RandomFlip | Horizontal |
| RandomRotation | 0.03 |
| RandomZoom | 0.05 |

### 6.5 Callbacks (Centralized only)

| Callback | Monitor | Config |
|----------|---------|--------|
| EarlyStopping | val_loss | patience=5 (S1), patience=3 (S2) |
| ReduceLROnPlateau | val_loss | factor=0.3, patience=2, min_lr=1e-7 |
| ModelCheckpoint | val_loss | save_best_only=True |

---

## 7. Evaluation Metrics

All experiments are evaluated on the **global test set** (concatenation of per-client test splits) with the following metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision (macro) | Macro-averaged precision across 5 classes |
| Recall (macro) | Macro-averaged recall across 5 classes |
| F1-score (macro) | Macro-averaged F1 across 5 classes |
| Precision (weighted) | Weighted by class support |
| F1-score (weighted) | Weighted by class support |
| AUC (macro OvR) | One-vs-Rest macro AUC |
| AUC (weighted OvR) | One-vs-Rest weighted AUC |

Additional analyses:
- **ROC curve** per class
- **Confusion matrix** (global test set)
- **Per-client performance** breakdown (DDR / EyePACS / APTOS separately)
- **FL training curve** — val accuracy & loss per round
- **Lambda schedule curve** — for DANN experiments

Saved outputs per experiment:
```
results/MultiClient/{exp_name}/{model_name}/
├── logs/
│   ├── baseline_summary_table.csv / .json
│   ├── classification_report_test.txt / .json
│   ├── per_class_metrics.csv / .json
│   ├── per_client_test_metrics.csv / .json
│   ├── fl_round_logs.json
│   ├── fl_round_summary.csv
│   ├── roc_auc_per_class.json
│   ├── confusion_matrix_table.csv
│   └── comparison_ready_baseline.json
└── figures/
    ├── fl_training_curve.png
    ├── roc_curve_baseline.png
    ├── confusion_matrix_heatmap_baseline.png
    └── dann_domain_analysis.png  (DANN only)
```

---

## 8. Results

> ⚠️ Results are partially available. Experiments completed so far: **exp1 (Centralized)** and **exp2 (FL Loss Ascending)** — both MobileNetV2. Remaining experiments will be updated as they complete.

---

### 8.1 Baseline Comparison Table

| Method | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | AUC (macro OvR) |
|--------|-------|----------|------------------|----------------|-----------|--------------|----------------|
| **Centralized Pooled** | MobileNetV2 | **0.4255** | 0.3066 | 0.4016 | 0.2647 | 0.4737 | 0.7289 |
| **FL Loss Ascending** | MobileNetV2 | **0.5029** | 0.2789 | 0.3239 | 0.2744 | 0.5053 | 0.6872 |
| Centralized Pooled | EfficientNetB0 | — | — | — | — | — | — |
| FL Loss Ascending | EfficientNetB0 | — | — | — | — | — | — |
| FL Loss Descending | MobileNetV2 | — | — | — | — | — | — |
| FL Loss Descending | EfficientNetB0 | — | — | — | — | — | — |
| FL Size Ascending | MobileNetV2 | — | — | — | — | — | — |
| FL Size Ascending | EfficientNetB0 | — | — | — | — | — | — |
| FL Size Descending | MobileNetV2 | — | — | — | — | — | — |
| FL Size Descending | EfficientNetB0 | — | — | — | — | — | — |
| FL+DANN Loss Ascending | MobileNetV2 | — | — | — | — | — | — |
| FL+DANN Loss Ascending | EfficientNetB0 | — | — | — | — | — | — |
| FL+DANN Loss Descending | MobileNetV2 | — | — | — | — | — | — |
| FL+DANN Loss Descending | EfficientNetB0 | — | — | — | — | — | — |
| FL+DANN Size Ascending | MobileNetV2 | — | — | — | — | — | — |
| FL+DANN Size Ascending | EfficientNetB0 | — | — | — | — | — | — |
| FL+DANN Size Descending | MobileNetV2 | — | — | — | — | — | — |
| FL+DANN Size Descending | EfficientNetB0 | — | — | — | — | — | — |

> 🔄 Rows marked `—` will be updated as experiments complete.

---

### 8.2 EXP1 — Centralized Pooled Baseline (MobileNetV2)

**Training config:** Global pooled split (70:15:15) | Stage 1: 10 epochs (EarlyStopping) | Stage 2: 4 epochs (EarlyStopping)

| Stage | Epochs Run | Best Val Accuracy | Best Val Loss |
|-------|-----------|------------------|--------------|
| Stage 1 (frozen backbone) | 10 | 0.5446 | 1.1526 |
| Stage 2 (fine-tune last 10 layers) | 4 | 0.4335 | 1.2430 |

**Final test set performance:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.4255 |
| Precision (macro) | 0.3066 |
| Recall (macro) | 0.4016 |
| F1-score (macro) | 0.2647 |
| F1-score (weighted) | 0.4737 |
| AUC (macro OvR) | 0.7289 |
| AUC (weighted OvR) | 0.7157 |

**Per-class metrics:**

| Class | Description | Precision | Recall | F1-score | Support |
|-------|-------------|-----------|--------|----------|---------|
| 0 | No DR | 0.8181 | 0.5362 | 0.6478 | 5,082 |
| 1 | Mild | 0.1102 | 0.3211 | 0.1641 | 517 |
| 2 | Moderate | 0.3355 | 0.0644 | 0.1081 | 1,615 |
| 3 | Severe | 0.1708 | 0.3538 | 0.2304 | 195 |
| 4 | Proliferative | 0.0983 | 0.7326 | 0.1734 | 288 |

**ROC AUC per class:**

| Class | AUC |
|-------|-----|
| Class 0 (No DR) | 0.7315 |
| Class 1 (Mild) | 0.6098 |
| Class 2 (Moderate) | 0.6674 |
| Class 3 (Severe) | 0.7846 |
| Class 4 (Proliferative) | 0.8513 |

**Per-dataset test performance:**

| Dataset | n | Accuracy | F1 (macro) | AUC (macro) |
|---------|---|----------|-----------|------------|
| APTOS | 558 | 0.6219 | 0.4563 | 0.8557 |
| DDR | 1,901 | **0.0726** | 0.0271 | **0.5367** |
| EyePACS | 5,238 | 0.5326 | 0.3202 | 0.7286 |

> ⚠️ **DDR anomaly detected:** DDR test accuracy of 7.26% and AUC of 0.537 (near-random) indicates a potential data loading or label alignment issue specific to the DDR dataset. APTOS and EyePACS perform at expected levels. Investigation is ongoing — see Section 8.5.

---

### 8.3 EXP2 — FL Loss Ascending (MobileNetV2)

**Training config:** Per-client split (70:15:15) | 10 FL rounds | 3 local epochs per round | FedAvg aggregation

**Final test set performance:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.5029 |
| Precision (macro) | 0.2789 |
| Recall (macro) | 0.3239 |
| F1-score (macro) | 0.2744 |
| F1-score (weighted) | 0.5053 |
| AUC (macro OvR) | 0.6872 |
| AUC (weighted OvR) | 0.6637 |

**Per-class metrics:**

| Class | Description | Precision | Recall | F1-score | Support |
|-------|-------------|-----------|--------|----------|---------|
| 0 | No DR | 0.7002 | 0.6770 | 0.6884 | 5,083 |
| 1 | Mild | 0.1098 | 0.2979 | 0.1605 | 517 |
| 2 | Moderate | 0.2819 | 0.0848 | 0.1304 | 1,616 |
| 3 | Severe | 0.1357 | 0.2359 | 0.1723 | 195 |
| 4 | Proliferative | 0.1670 | 0.3240 | 0.2204 | 287 |

**ROC AUC per class:**

| Class | AUC |
|-------|-----|
| Class 0 (No DR) | 0.6906 |
| Class 1 (Mild) | 0.6029 |
| Class 2 (Moderate) | 0.5599 |
| Class 3 (Severe) | 0.7763 |
| Class 4 (Proliferative) | 0.8066 |

**Per-client test performance:**

| Client | n | Accuracy | F1 (macro) | AUC (macro) |
|--------|---|----------|-----------|------------|
| DDR | 1,879 | 0.5003 | 0.1334 | **0.4931** |
| EyePACS | 5,269 | 0.5007 | 0.2922 | 0.7286 |
| APTOS | 550 | 0.5327 | 0.3771 | 0.7558 |

**FL round-by-round progress:**

| Round | Stage | Client Order | Val Loss | Val Accuracy |
|-------|-------|-------------|---------|-------------|
| 1 | 1 | DDR → EyePACS → APTOS | 1.1600 | 0.5526 |
| 2 | 1 | EyePACS → APTOS → DDR | 1.1650 | 0.5673 |
| 3 | 1 | EyePACS → APTOS → DDR | 1.1902 | 0.5715 |
| **4** | **1** | **EyePACS → APTOS → DDR** | **1.1412** | **0.6042** ★ |
| 5 | 1 | EyePACS → APTOS → DDR | 1.1618 | 0.5667 |
| 6 | 2 | APTOS → EyePACS → DDR | 1.2518 | 0.5032 |
| 7 | 2 | APTOS → EyePACS → DDR | 1.2725 | 0.4942 |
| 8 | 2 | APTOS → EyePACS → DDR | 1.2722 | 0.4938 |
| 9 | 2 | APTOS → EyePACS → DDR | 1.2601 | 0.5032 |
| 10 | 2 | APTOS → EyePACS → DDR | 1.2609 | 0.5047 |

> **Best val accuracy: 0.6042 at round 4 (Stage 1).** Stage 2 fine-tuning caused a significant performance drop from 60.4% to ~50%, suggesting instability during partial backbone unfreezing in the federated setting.

---

### 8.4 EXP1 vs EXP2 — Preliminary Comparison

| Metric | Centralized | FL Loss Asc | Δ |
|--------|-------------|------------|---|
| Accuracy | 0.4255 | **0.5029** | +0.0774 ↑ |
| F1 (macro) | 0.2647 | **0.2744** | +0.0097 ↑ |
| F1 (weighted) | 0.4737 | **0.5053** | +0.0316 ↑ |
| AUC (macro) | **0.7289** | 0.6872 | -0.0417 ↓ |
| AUC (weighted) | **0.7157** | 0.6637 | -0.0520 ↓ |

**Key observations:**
- FL Loss Ascending achieves **higher accuracy (+7.7%)** and **better F1** compared to the centralized baseline — a notable finding given that FL trains on isolated, non-shared data
- Centralized has **better AUC** — suggesting stronger probabilistic calibration despite lower accuracy, possibly due to the pooled training signal
- The sorting pattern in exp2 is consistent: in Stage 1, EyePACS always trains first (lowest loss = easiest domain), while DDR always trains last (highest loss = hardest domain)
- Stage 2 degradation in FL is a concern — both experiments show instability during fine-tuning

---

### 8.5 ⚠️ DDR Data Anomaly — Investigation Note

DDR test performance is anomalously low across both experiments:

| Experiment | DDR Accuracy | DDR AUC | Expected |
|-----------|-------------|---------|---------|
| EXP1 (Centralized) | 0.0726 | 0.5367 | ~0.50+ |
| EXP2 (FL Loss Asc) | 0.5003 | 0.4931 | ~0.50+ |

DDR accuracy in EXP2 (50%) aligns with random-majority-class prediction. AUC near 0.5 in both experiments indicates near-random discriminative power for DDR samples specifically, while APTOS and EyePACS perform normally.

**Possible causes under investigation:**
1. Label ordering mismatch between `DR_grading.csv` and `DDR_dataset_224.npz` — image-label pairs may not be aligned
2. DDR preprocessing pipeline may have produced images in a different order than the CSV
3. DDR domain distribution shift is significantly larger than APTOS/EyePACS

**Recommended verification step before running remaining experiments:**
```python
# Run this cell to verify DDR alignment
ddr_data = np.load(DDR_NPZ_PATH, allow_pickle=True)
ddr_df   = pd.read_csv(DDR_LABEL_PATH)
print("NPZ image count:", ddr_data["X"].shape[0])
print("CSV row count  :", len(ddr_df))
print("CSV columns    :", ddr_df.columns.tolist())
print(ddr_df.head(10))
```

---

### 8.6 Remaining Experiments

> 🔄 To be updated as experiments complete.

| Experiment | Status |
|-----------|--------|
| exp1 — Centralized MobileNetV2 | ✅ Complete |
| exp2 — FL Loss Asc MobileNetV2 | ✅ Complete |
| exp3 — FL Loss Desc MobileNetV2 | ⏳ Pending |
| exp4 — FL Size Asc MobileNetV2 | ⏳ Pending |
| exp5 — FL Size Desc MobileNetV2 | ⏳ Pending |
| exp6 — FL+DANN Loss Asc MobileNetV2 | ⏳ Pending |
| exp7 — FL+DANN Loss Desc MobileNetV2 | ⏳ Pending |
| exp8 — FL+DANN Size Asc MobileNetV2 | ⏳ Pending |
| exp9 — FL+DANN Size Desc MobileNetV2 | ⏳ Pending |
| exp1 — Centralized EfficientNetB0 | ⏳ Pending |
| exp2 — FL Loss Asc EfficientNetB0 | ⏳ Pending |
| exp3 — FL Loss Desc EfficientNetB0 | ⏳ Pending |
| exp4 — FL Size Asc EfficientNetB0 | ⏳ Pending |
| exp5 — FL Size Desc EfficientNetB0 | ⏳ Pending |
| exp6 — FL+DANN Loss Asc EfficientNetB0 | ⏳ Pending |
| exp7 — FL+DANN Loss Desc EfficientNetB0 | ⏳ Pending |
| exp8 — FL+DANN Size Asc EfficientNetB0 | ⏳ Pending |
| exp9 — FL+DANN Size Desc EfficientNetB0 | ⏳ Pending |

---

## 9. References

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). **Communication-efficient learning of deep networks from decentralized data.** *AISTATS 2017.*
2. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). **Domain-adversarial training of neural networks.** *JMLR, 17(1), 2096–2030.*
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). **MobileNetV2: Inverted residuals and linear bottlenecks.** *CVPR 2018.*
4. Raj, G. M., Morley, M. G., & Eslami, M. (2024). **Federated learning for diabetic retinopathy diagnosis: Enhancing accuracy and generalizability in under-resourced regions.**
5. Tan, M., & Le, Q. (2019). **EfficientNet: Rethinking model scaling for convolutional neural networks.** *ICML 2019.*
6. APTOS 2019 Blindness Detection — Kaggle Dataset.
7. DDR Dataset — Diabetic Retinopathy Grading.
8. EyePACS — Diabetic Retinopathy Detection, Kaggle Dataset.

---

*Report generated as part of Project Orion — Federated Learning for Diabetic Retinopathy Classification.*
