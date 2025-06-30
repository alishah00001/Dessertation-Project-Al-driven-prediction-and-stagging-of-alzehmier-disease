<img width="215" alt="image" src="https://github.com/user-attachments/assets/f72ba5d7-dedd-4a4c-b9b1-5732e623a2ad" /># Dessertation-Project-Al-driven-prediction-and-stagging-of-alzehmier-disease
This was my dessertation project in which I design AI-driven prediction and stagging of alzehmier disease using clinical and MRI data set using ML and DL approaches.
There are 2 code files as we can see in above and their file name also describe about it.

Abstract

Alzheimer disease is a progressive neurodegenerative disorder that significantly affects a lot of people in United Kingdom and many other countries. Various research studies have aimed to understand and manage this condition, with early prediction, and accurate staging being important for timely intervention and care planning. However, studies are often done in isolation, for example image analysis through deep learning (DL) being done without complementary machine learning (ML) through classification of clinical data features, a bi-modal approach used in this paper. Experiments used two data sets, MRI scans data and Clinical data to investigate both early prediction and stage classification of Alzheimer’s disease. Firstly, supervised machine learning techniques were applied to the clinical data set to diagnose whether a patient has Alzheimer disease or not.  The best model was Random Forest, with an accuracy of 95% on the test set. Explainable AI was also used in this part to highlight which features are influencing the predictive model's decision. The analysis reveals that certain features like behavioral problems and memory complaints have a greater impact for predicting the presence of the disease. Secondly, a convolutional neural network (CNN) model was implemented and then trained on MRI scans to further classify the patients into 4 categories i.e. Non-Dementia, Very Mild Dementia, Mild Dementia, and Moderate Dementia. This model achieved an accuracy of 86% on test set. Explainable AI (LIME) was also used in this part to highlight the brain region influencing the CNN model’s decision. The approach would assist medical practitioners to make informed decisions for their patients regarding disease presence and disease severity using ML and DL techniques respectively. 

Goal
The goal of this project is to develop a model that can aid the doctor in making a diagnosis by integrating data-driven modelling with explainable AI approaches such as LIME. This project adhered to a traditional methodology.
 In this section, we describe the methodology that would allow this Alzheimer's detection system to operate in a real-world clinical context. Imagine a real-life situation where a patient goes to the doctor and describes their symptoms, such as memory loss, changes in behaviour, MMSE scores, age, etc. The doctor then inputs all of the patient's symptoms into our system and runs the model. If the model predicts a positive result, meaning the patient has Alzheimer's, the doctor will then suggest an MRI brain scan to further investigate. The process of uploading the MRI scan to the system begins after its acquisition. Next, a convolutional neural network (CNN) model evaluates the scan and assigns it to one of four dementia severity levels: mild, moderate, very 12 mild, or non-existent. The decision-making process is further aided by an explainable AI (LIME) that reveals where region of the brain is involved. This two-pronged strategy aids in both early diagnosis and the appropriate staging of diseases using medical imaging. It helps doctors and nurses make quick, well-informed judgements about patient diagnosis and treatment.

Key Novelty
The existing literature demonstrates robust advancements in AD diagnosis using both unimodal and multimodal ML and DL approaches. However, a significant gap remains in studies that was addressed in this research. This research aims to bridge existing gaps by developing a  explainable, two-stage diagnostic framework that leverages both clinical and imaging data for improved Alzheimer’s disease detection and staging. The main contributions are as follows:
A step-wise detection and stagging of Alzehmier disease: While many multimodal studies combine different imaging modalities (e.g., MRI and PET), they often overlook the inclusion of rich clinical features. Our work addresses this gap by incorporating comprehensive clinical data for detection of disease alongside using MRI imaging for predicting stage of disease.
A two-stage diagnostic framework: We propose a sequential approach where a machine learning model analyzes clinical data to first predict the presence of Alzheimer's disease, followed by a deep learning model (CNN) applied to MRI scans for disease staging.
Explainable AI (XAI) for both clinical and imaging modalities: Our framework provides interpretable insights into model decisions across both data types, enhancing clinical trust and facilitating adoption.

Methodology
<img width="365" alt="image" src="https://github.com/user-attachments/assets/1ceb31b3-a862-424a-ae13-6a83f2eacadc" />

<img width="350" alt="image" src="https://github.com/user-attachments/assets/9cc05d6f-8a7b-4e1b-80da-c6aebc2ab560" />

Results:-
 Experimental Results for Clinical Data
 Table  Hyperparameter Tuning Results and Cross-Validation Accuracy
Model	Best Parameters	Cross-Validation Accuracy (%)
Random Forest	max_depth=10, max_features='sqrt', min_samples_leaf=5, min_samples_split=10, n_estimators=200	95.19
Logistic Regression	C=0.1, penalty='l2', solver='liblinear'	83.57
Support Vector Machine (SVM)	C=10, gamma='auto', kernel='rbf'	92.67
Gradient Boosting	learning_rate=0.2, max_depth=10, min_samples_split=10, n_estimators=200	94.92

Random forest emerges as best model
Table 6 Performance Metrics of the Random Forest model
Test accuracy	95.12%
Train accuracy	95.46%
precision-recall curve 	92%,
AUC-ROC score	94%
cross-validation accuracy	95%
out-of-fold AUC-ROC	97%
confidence of interval 	95%

Learning Curves to diagnose overfitting of Random Forest using Clinical dataset
<img width="346" alt="image" src="https://github.com/user-attachments/assets/ac81bf65-c762-4317-95b8-7d986018b838" />

Classification results of Random Forest with Clinical dataset:
Table 7 Classification Results for Random Forest using clinical dataset

Class	Precision (%)	Recall (%)	F1-Score (%)	Support (n)
No Alzheimer Disease	96.0	96.0	96.0	278
Has Alzheimer Disease	93.0	93.0	93.0	152
				
Accuracy			95.0	430
Macro Average	95.0	95.0	95.0	430
Weighted Average	95.0	95.0	95.0	430

Confusion matrix for Random Forest using Clinical Dataset:
<img width="215" alt="image" src="https://github.com/user-attachments/assets/46b054bd-bba0-4cb6-8e9a-b71457b597e7" />

Experimental Results for MRI Scans
Confusion matrix of CNN Model using MRI Scans:
<img width="272" alt="image" src="https://github.com/user-attachments/assets/d958e20b-5f77-4820-ada9-1e57f93bfe67" />

CNN Model Performance Metrics Across Training and Testing Phases
Table  CNN Model Performance Metrics Across Training and Testing Phases with using MRI scan data Set.
Test name	Accuracy
Train Accuracy	87.07%
Training loss	0.42
Test accuracy	86.32%
Test loss	0.42

Classification results of CNN Model using MRI Scan dataset
Table 10 Classification results of CNN model using MRI Scans
Class	Precision (%)	Recall (%)	F1-Score (%)	Support (n)
Mild Demented	80.0	96.0	87.0	2,693
Moderate Demented	98.0	99.0	99.0	1,977
Non Demented	86.0	84.0	85.0	2,811
Very Mild Demented	86.0	70.0	77.0	2,715
Metric	Score (%)
Accuracy	86.0
Macro Average	87.0
Weighted Average	86.0


Doctor Assisted system
<img width="338" alt="image" src="https://github.com/user-attachments/assets/428cc679-5ab2-40b6-854c-be46fbc75723" />

<img width="450" alt="image" src="https://github.com/user-attachments/assets/9b15116a-904a-47f1-9bcf-472a8b63d6d0" />

 
