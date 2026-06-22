# Fraud Detection with PyOD AutoEncoder

## Overview

This project implements an **unsupervised deep learning approach to credit card fraud detection** using the **AutoEncoder** model from the PyOD (Python Outlier Detection) library.

The model learns the patterns of normal credit card transactions and identifies anomalous transactions based on reconstruction error. Transactions with high reconstruction errors are flagged as potential fraud without requiring fraud labels during training.

The project uses the Kaggle **Credit Card Fraud Detection** dataset and evaluates model performance using classification metrics, ROC-AUC, and Precision-Recall AUC.

---

## Dataset

**Source:** Kaggle Credit Card Fraud Detection Dataset

https://www.kaggle.com/datasets/whenamancodes/fraud-detection

### Dataset Characteristics

* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Fraud Rate: ~0.17%
* Features:

  * V1–V28 (PCA-transformed features)
  * Time
  * Amount
  * Class (0 = Legitimate, 1 = Fraud)

---

## Project Structure

```text
.
├── fraud_detection_autoencoder.py
├── requirements.txt
├── creditcard.csv
├── outputs/
│   ├── evaluation_report.txt
│   ├── roc_curve.png
│   └── anomaly_score_distribution.png
└── README.md
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Recommended requirements:

```txt
pyod>=2.0.0
torch>=2.0.0
numpy==1.26.4
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
numba<0.61
llvmlite<0.44
tqdm>=4.66.0
```

---

## Environment Setup

Create and activate a virtual environment:

### macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Place the downloaded `creditcard.csv` file in the project directory.

Run the AutoEncoder model:

```bash
python fraud_detection_autoencoder.py --epochs 30 --batch-size 64
```

Example with a custom dataset path:

```bash
python fraud_detection_autoencoder.py --data-path path/to/creditcard.csv
```

---

## Model Architecture

The AutoEncoder uses a symmetric encoder-decoder architecture:

```text
Input
  ↓
32 neurons
  ↓
16 neurons
  ↓
8 neurons
  ↓
16 neurons
  ↓
32 neurons
  ↓
Output
```

Activation Function:

```text
ReLU
```

Training Parameters:

* Epochs: 30
* Batch Size: 64
* Contamination: 0.0017
* Random State: 42

---

## Evaluation Metrics

The model is evaluated using:

* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC
* Average Precision (PR-AUC)

---

## Experimental Results

### Classification Metrics (Fraud Class)

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.2526 |
| Recall    | 0.2449 |
| F1-Score  | 0.2487 |

### Overall Metrics

| Metric  | Value  |
| ------- | ------ |
| ROC-AUC | 0.9585 |
| PR-AUC  | 0.1770 |

### Confusion Matrix

```text
[[56793   71]
 [   74   24]]
```

Where:

* True Positives (TP): 24
* False Positives (FP): 71
* False Negatives (FN): 74
* True Negatives (TN): 56,793

---

## Generated Outputs

The script automatically generates:

### Evaluation Report

```text
outputs/evaluation_report.txt
```

### ROC Curve

```text
outputs/roc_curve.png
```

### Anomaly Score Distribution

```text
outputs/anomaly_score_distribution.png
```

---

## Discussion

The AutoEncoder successfully learned patterns of normal transactions and achieved a strong ROC-AUC score of 0.9585, indicating effective separation of fraudulent and legitimate transactions through reconstruction error. However, because fraud cases represent only 0.17% of the dataset, precision and recall remained relatively low.

The PR-AUC score of 0.1770 provides a more realistic assessment of performance on this highly imbalanced dataset. Future improvements could include threshold tuning, deeper network architectures, additional training epochs, and comparisons with other PyOD detectors such as Isolation Forest and ECOD.

---

## Future Work

Potential enhancements include:

* Hyperparameter optimization
* Hidden layer architecture tuning
* Contamination threshold adjustment
* Ensemble anomaly detection methods
* Comparison with:

  * Isolation Forest
  * ECOD
  * COPOD
  * DeepSVDD
  * One-Class SVM

---

## Author

**Bala Krishna Konakanchi**

MSCS-633-A01: Advance Artificial Intelligence

University of the Cumberlands

Summer 2026

---

## License

This project is provided for educational and academic purposes.
