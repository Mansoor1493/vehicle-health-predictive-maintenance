## 1. Project Overview
This project presents an intelligent predictive maintenance (PdM) and vehicle health monitoring framework. Utilizing multi-brand automotive service logs, machine learning classification pipelines correlate reported vehicular anomalies (`COMMON PROBLEM`), manufacturer signatures (`VEHICAL COMPANY`), and historical maintenance logs (`SERVICE HISTORY`) to prescribe automated, accurate workshop remediation protocols (`SOLUTION USED`).

---

## 2. Exploratory Data Analysis (EDA)
The experiment analyzed **508 service records** across **7 attributes** sourced from telemetry service logs:
- **Data Integrity:** Complete records with 0 null or missing values across all features.
- **Geographical Focus:** Ananthapuram region, Andhra Pradesh, capturing localized operational stress factors.
- **Manufacturer Representation:** 100 unique vehicle brands, led by **Hyundai** (42), **Ford** (37), **Hero MotoCorp** (33), **Volkswagen** (31), **Tata Motors** (31), and **Bajaj Auto** (29).
- **Prevalent Fault Categories:** Cloudy headlights (49), Brake noise (41), Exhaust noise (32), Tire wear (31), Fluid leak (27), Steering issues (20), Engine overheating (18).
- **Remedial Interventions:** Headlight restoration (101), Brake pad replacement (45), Exhaust system repair (38), Fluid replacement (27), Diagnostic scan (21).

---

## 3. Machine Learning Algorithms & Benchmark Evaluation
Input features were synthesized into composite semantic vectors and processed via **TF-IDF n-gram vectorization** (`ngram_range=(1, 2)`). Models were benchmarked on an 80/20 stratified train-test split:

| Model Architecture | Accuracy (%) | Weighted F1-Score | Remarks |
| :--- | :---: | :---: | :--- |
| **Logistic Regression (TF-IDF)** | **96.08%** | **0.9543** | Optimal generalization across high-dimensional sparse text vectors |
| **Gradient Boosting Classifier** | **94.12%** | **0.9355** | Robust multi-class decision boundaries across vehicle wear classes |
| **Random Forest Classifier** | **94.12%** | **0.9346** | Resilient ensemble baseline matching tree-based benchmarks |

---

## 4. Comparison with Literature Review (20 Research Papers)
The experimental results align closely with established benchmarks from the 20-paper literature review:
- **Tabular & NLP Diagnostics Alignment:** The **96.08%** accuracy directly mirrors benchmarks from Paper #4 (*IJIRT 2025*, 95.2% accuracy) and Paper #5 (*Trends in Machine Design 2026*, 97.3% fault-type classification).
- **Mitigating Class Imbalance:** Grouping long-tail infrequent maintenance categories aligns with methodology in Paper #3 (*MDPI Sensors 2025*), ensuring weighted F1-scores exceed 0.93 across all candidate architectures.
- **Edge AI Feasibility:** Lightweight linear and tree pipelines provide immediate inference under microsecond latencies, satisfying embedded hardware constraints outlined in Paper #6 (OBD-II microcontrollers) and Paper #13 (CAN-bus edge anomaly filtering).

---

## 5. Repository Directory Structure
```text
├── final dataset for kaggle.csv
├── vehicle_diagnostics_experiment.ipynb
├── Vehicle_Service_Dataset_Analysis_Final_Syed_Mansoor.docx
└── README.md
