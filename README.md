# Fraud Detection Classifier

## Overview
Credit card fraud is a critical problem in the financial industry. 
This project builds a machine learning model to detect fraudulent 
transactions using a real-world dataset of 284,807 transactions, 
of which only 492 (0.17%) are fraud.

## Problem
The dataset is highly imbalanced — 99.83% legitimate, 0.17% fraud.
A naive model predicting "not fraud" always would score 99.83% 
accuracy but catch zero fraudsters. This project addresses that 
by using proper evaluation metrics.

## Dataset
- Source: Kaggle Credit Card Fraud Detection
- 284,807 transactions, 31 features
- Features V1–V28 are PCA-transformed for privacy
- Original features: Time, Amount, Class

## Approach
1. Exploratory Data Analysis — class distribution visualization
2. Preprocessing — dropped Time column, log transformed Amount, 
   StandardScaler normalization
3. Train/Test Split — 80/20 with stratification to preserve fraud ratio
4. Model — Logistic Regression
5. Evaluation — AUC-ROC, Precision, Recall, Confusion Matrix

## Results
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.9599 |
| Precision (Fraud) | 0.83 |
| Recall (Fraud) | 0.65 |
| F1 Score (Fraud) | 0.73 |

## Key Insight
Accuracy is a misleading metric for imbalanced datasets.
AUC-ROC and Recall are the metrics that actually matter for 
fraud detection — missing real fraud is far more costly than 
a false alarm.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## How To Run
1. Download creditcard.csv from Kaggle
2. Place it in the project folder
3. Run fraud_detection.py