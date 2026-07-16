import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('creditcard.csv')

# 2. Check the shape (rows, columns)
print("Dataset Shape:", df.shape)

# 3. Inspect columns, data types, and missing values
print("\n--- Dataset Info ---")
print(df.info())

# 4. Check the class distribution (0 = Normal, 1 = Fraud)
print("\n--- Class Distribution ---")
print(df['Class'].value_counts())

import matplotlib.pyplot as plt

# # Visualize class distribution
class_counts = df['Class'].value_counts()
# plt.bar(['Not Fraud (0)', 'Fraud (1)'], class_counts.values, 
#         color=['steelblue', 'crimson'])
# plt.title('Class Distribution — Fraud vs Not Fraud')
# plt.ylabel('Number of Transactions')
# plt.tight_layout()
# plt.savefig('class_distribution.png')
# print("Chart saved!")
# Better visualization - log scale to see both bars
plt.figure(figsize=(6,4))
plt.bar(['Not Fraud (0)', 'Fraud (1)'], class_counts.values,
        color=['steelblue', 'crimson'])
plt.title('Class Distribution — Fraud vs Not Fraud')
plt.ylabel('Number of Transactions (Log Scale)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('class_distribution.png')
print("Chart saved!")

import numpy as np
from sklearn.preprocessing import StandardScaler


# 2. Log transform the Amount column
df['Amount'] = np.log1p(df['Amount'])

# 3. Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing done!")
print("X shape:", X_scaled.shape)
print("Fraud cases in y:", y.sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
       X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained!")
print("Training samples:", X_train.shape[0])
print("Testing samples:",X_test.shape[0])

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(y_test,y_pred))

auc = roc_auc_score(y_test,y_prob)
print("AUC-ROC Score:", round(auc,4))

cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix: ")
print(cm)
