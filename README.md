# EduPredict: Student Academic Performance Prediction Using Artificial Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.20.0-orange.svg)](https://js.tensorflow.org/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4.0-brightgreen.svg)](https://www.chartjs.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-orange.svg)](https://scikit-learn.org/)
[![MBA Data Science](https://img.shields.io/badge/Specialization-MBA%20Data%20Science%20%26%20AI-purple.svg)]()

> **Interactive Deep Learning Dashboard for Student Academic Performance Prediction and Risk Assessment**  
> Built for academic demonstration, research presentation, and viva voce examination in Master of Business Administration (MBA) Data Science & Artificial Intelligence.

---

## 🌟 Overview

**EduPredict** is an end-to-end Machine Learning and Deep Learning system designed to identify students at risk of academic failure before final examinations take place. The system combines:

1. **Python Training Pipeline**: Synthesizes multi-feature cohorts, applies leakage-free standardisation, trains a Multilayer Perceptron (ANN), evaluates on holdout test partitions, and serializes weights.
2. **Client-Side Web Dashboard (`app/index.html`)**: A standalone application utilizing **TensorFlow.js**, **Chart.js**, and **PapaParse**. It supports in-browser neural network training, custom CSV ingestion, dynamic classification threshold calibration, interactive data exploration, and individual student failure risk scoring.

---

## 🚀 Key Features

- **Synthetic Cohort Generator**: Configurable sample sizes (100, 500, 1,000, 2,000, 5,000 records) with non-linear relationships among attendance, study volume, previous performance, and lifestyle factors.
- **Custom CSV Upload Engine**: Integrates PapaParse for local file parsing, automated column type inspection, missing-value imputation, and dynamic target feature mapping.
- **Leakage-Free Preprocessing**: Standardizes numerical features using training-set parameters ($\mu, \sigma$) and one-hot encodes categorical variables. Target variables are excluded from inputs to eliminate data leakage.
- **Genuine Neural Network Architecture**: Multilayer Perceptron (13 input features $\rightarrow$ 64 Dense ReLU $\rightarrow$ Dropout $\rightarrow$ 32 Dense ReLU $\rightarrow$ Dropout $\rightarrow$ 1 Dense Sigmoid) optimized via Adam.
- **In-Browser Training & Live Telemetry**: Train fresh neural network models directly within the browser using TensorFlow.js with real-time loss and accuracy curves.
- **Interactive Evaluation & Dynamic Thresholding**: Adjust classification cutoff ($\tau \in [0.10, 0.90]$) to analyze trade-offs between precision, sensitivity (recall), and false positive rates.
- **Individual Student Risk Predictor**: Form with sliders to input student metrics, generating instant pass/fail probabilities and academic advising recommendations.
- **Export Capabilities**: Download synthetic datasets, test predictions, and individual diagnostic evaluations in CSV or JSON formats.
- **Academic Defense Documentation**: Built-in literature review, mathematical formulas, and limitation discussions.

---

## 📊 Benchmark Model Performance

Evaluated on 400 unseen holdout student records (80/20 train-test partition):

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Accuracy** | **96.75%** | 387 / 400 correct classifications |
| **Precision (Pass)** | **98.16%** | $TP / (TP + FP)$ |
| **Recall / Sensitivity** | **98.42%** | $TP / (TP + FN)$ |
| **F1-Score** | **98.29%** | Harmonic mean of precision and recall |
| **ROC - AUC** | **0.9622** | Area Under the Receiver Operating Characteristic |
| **True Positives (TP)** | **374** | Correctly predicted passing students |
| **True Negatives (TN)** | **13** | Correctly flagged at-risk failing students |
| **False Positives (FP)** | **7** | Failing students predicted as passing |
| **False Negatives (FN)** | **6** | Passing students flagged as at-risk |

---

## 📁 Repository Structure

```
Student-Performance-Predictor-Using-ANN/
├── app/
│   └── index.html                 # Standalone interactive dashboard (TensorFlow.js)
├── data/
│   ├── student_data.csv           # 2,000-record synthetic student dataset
│   ├── student_data_sample.csv   # 20-row sample preview
│   └── test_predictions.csv       # Out-of-sample holdout test predictions
├── model/
│   ├── metrics.json               # Benchmark test evaluation metrics
│   ├── model_weights.json         # Serialized ANN layer weights and architecture
│   ├── preprocessor.json          # Standardizer means/stds and category mappings
│   └── training_history.json      # Loss curve and validation convergence history
├── notebooks/
│   └── EduPredict_Training.ipynb  # Interactive Jupyter notebook walkthrough
├── report/
│   └── EduPredict_Project_Report.md # Academic research report
├── src/
│   └── train.py                   # Standalone Python pipeline (data gen + ANN training)
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation and viva guide
```

---

## 💻 Quick Start Guide

### Option 1: Open Interactive Web Dashboard (Zero Installation)
Simply double-click or open `app/index.html` in any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Safari).

No Python environment, backend server, or terminal commands are required. The application automatically loads the embedded pre-trained weights and allows immediate prediction, in-browser model retraining, and CSV processing.

### Option 2: Run Python Pipeline & Retrain Locally

```bash
# 1. Clone repository
git clone https://github.com/divyansh-chatrath/Student-Performance-Predictor-Using-ANN.git
cd Student-Performance-Predictor-Using-ANN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute training pipeline
python src/train.py
```

The script will:
1. Generate `data/student_data.csv`.
2. Standardize features and save parameters in `model/preprocessor.json`.
3. Train the Artificial Neural Network.
4. Evaluate holdout test predictions and output metrics.
5. Export layer weights into `model/model_weights.json`.

---

## 🧠 Neural Network Mathematics

### Forward Propagation
For layer $l$:
$$z^{[l]} = \mathbf{W}^{[l]} a^{[l-1]} + \mathbf{b}^{[l]}$$
$$a^{[l]} = \text{ReLU}(z^{[l]}) = \max(0, z^{[l]})$$

Output unit:
$$\hat{y} = \sigma(z^{[3]}) = \frac{1}{1 + e^{-z^{[3]}}} = P(\text{Pass} \mid \mathbf{x})$$

### Objective Function (Binary Cross-Entropy with $L_2$ Regularization)
$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \Big] + \frac{\alpha}{2} \|\mathbf{W}\|_2^2$$

### Weight Update via Adam Optimizer
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 🎓 Viva Voce & Academic Defense Q&A

**Q1: Why utilize an Artificial Neural Network instead of standard Logistic Regression?**  
*Answer:* While logistic regression models linear boundaries on log-odds, student academic success involves non-linear factor interactions (e.g., how high independent study volume compensates for moderate attendance or sleep variations). The hidden layers with ReLU non-linearities capture these multidimensional interaction manifolds without manual interaction term engineering.

**Q2: How was data leakage prevented in this pipeline?**  
*Answer:* Target variables (`Final_Score`, `Performance`) and primary keys (`Student_ID`) were strictly isolated. Feature transformation statistics (sample means $\mu$ and standard deviations $\sigma$) were computed exclusively on the 80% training split and saved. The 20% holdout test partition and single-student inferences are scaled using only these stored training parameters.

**Q3: How does the dynamic decision threshold ($\tau$) assist educational administrators?**  
*Answer:* In standard ML, $\tau = 0.50$ treats False Positives and False Negatives symmetrically. However, in higher education, missing a student who subsequently fails (False Negative) has far higher institutional and personal costs than providing supplementary advising to a student who passes (False Positive). Lowering $\tau$ to 0.35 elevates failure recall to 85%, ensuring at-risk students receive timely academic remediation.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**divyansh**
- GitHub: [@divyansh-chatrath](https://github.com/divyansh-chatrath)
- Repository: [SMS-Spam-Detector]((https://github.com/divyansh-chatrath/SMS-Spam-Detector))
