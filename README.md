# Dissertation Project: AI-Driven Prediction and Staging of Alzheimer's Disease

This project presents an AI-driven framework for the **prediction** and **staging** of Alzheimer's Disease using **clinical data** and **MRI scans**, leveraging Machine Learning (ML) and Deep Learning (DL) techniques.

There are two code files in this repository:

* One for ML on clinical data.
* One for DL on MRI images.

---
## Datasets
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data
https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset


## 📄 Abstract

Alzheimer's disease is a progressive neurodegenerative disorder affecting millions globally. Early prediction and accurate staging are critical for effective intervention. Existing studies often focus on a single modality—clinical or imaging data. This project combines both in a **Step-wise, two-stage diagnostic system**:

* **Stage 1:** Supervised ML techniques (Random Forest, SVM, etc.) are applied to clinical data to diagnose the presence of Alzheimer’s.
* **Stage 2:** A CNN is trained on MRI scans to classify disease stage into: Non-Dementia, Very Mild Dementia, Mild Dementia, and Moderate Dementia.

**Explainable AI (LIME)** is used throughout to interpret model decisions, highlighting relevant features and brain regions.

---

## 🎯 Goal

To develop a two-stage diagnostic model that:

* Predicts Alzheimer's disease using **clinical features**.
* Stages disease severity using **MRI scans**.
* Supports clinicians through **explainable AI** insights.

### Example Workflow:

1. Doctor inputs clinical features (e.g., memory loss, MMSE score).
2. Model predicts presence of Alzheimer’s with explainable AI (LIME).
3. If positive, MRI scan is uploaded.
4. CNN model is used to predict the stage of the disease.
5. LIME highlights critical features/regions on MRI SCan.

This approach allows for early diagnosis and effective treatment planning.

---

## 🔍 Key Contributions

* **Step-wise detection and staging:** Combines clinical and imaging data in a two-stage system.
* **Two-stage diagnostic framework:** ML for diagnosis, CNN for staging.
* **Explainable AI (LIME):** Interpretable outputs for both modalities, enhancing clinician trust.

---

## 🧪 Methodology

### System Overview

![Methodology Flowchart](https://github.com/user-attachments/assets/1ceb31b3-a862-424a-ae13-6a83f2eacadc)

![Two-stage system](https://github.com/user-attachments/assets/9cc05d6f-8a7b-4e1b-80da-c6aebc2ab560)

---

## 📊 Results

### Clinical Data: ML Model Comparison

| Model               | Best Parameters                                      | Cross-Validation Accuracy (%) |
| ------------------- | ---------------------------------------------------- | ----------------------------- |
| Random Forest       | max\_depth=10, n\_estimators=200, ...                | 95.19                         |
| Logistic Regression | C=0.1, penalty='l2', solver='liblinear'              | 83.57                         |
| SVM                 | C=10, gamma='auto', kernel='rbf'                     | 92.67                         |
| Gradient Boosting   | learning\_rate=0.2, max\_depth=10, n\_estimators=200 | 94.92                         |

**Best Model: Random Forest**

* **Test Accuracy:** 95.12%
* **Train Accuracy:** 95.46%
* **AUC-ROC Score:** 94%
* **Cross-validation Accuracy:** 95%

**Learning Curve:**

![Learning Curve](https://github.com/user-attachments/assets/ac81bf65-c762-4317-95b8-7d986018b838)

### Clinical Data: Classification Report

| Class                | Precision (%) | Recall (%) | F1-Score (%) | Support (n) |
| -------------------- | ------------- | ---------- | ------------ | ----------- |
| No Alzheimer Disease | 96            | 96         | 96           | 278         |
| Has Alzheimer        | 93            | 93         | 93           | 152         |

**Confusion Matrix:**

![Confusion Matrix](https://github.com/user-attachments/assets/46b054bd-bba0-4cb6-8e9a-b71457b597e7)

---

### MRI Scan Data: CNN Results

**Confusion Matrix:**

![CNN Confusion Matrix](https://github.com/user-attachments/assets/d958e20b-5f77-4820-ada9-1e57f93bfe67)

**Performance Metrics:**

| Metric         | Value  |
| -------------- | ------ |
| Train Accuracy | 87.07% |
| Test Accuracy  | 86.32% |
| Training Loss  | 0.42   |
| Test Loss      | 0.42   |

**Classification Report:**

| Class              | Precision (%) | Recall (%) | F1-Score (%) | Support (n) |
| ------------------ | ------------- | ---------- | ------------ | ----------- |
| Mild Demented      | 80            | 96         | 87           | 2,693       |
| Moderate Demented  | 98            | 99         | 99           | 1,977       |
| Non Demented       | 86            | 84         | 85           | 2,811       |
| Very Mild Demented | 86            | 70         | 77           | 2,715       |

**Macro Average Accuracy:** 87%
**Weighted Accuracy:** 86%

---

## 🩺 Doctor-Assisted Diagnostic System

![Doctor System Step 1](https://github.com/user-attachments/assets/428cc679-5ab2-40b6-854c-be46fbc75723)

![Doctor System Step 2](https://github.com/user-attachments/assets/9b15116a-904a-47f1-9bcf-472a8b63d6d0)

---




