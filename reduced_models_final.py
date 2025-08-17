import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt

np.random.seed(42)

# ==============================
# Logistic Model 1 (Age 24 only)
# ==============================
df1 = pd.read_csv("cleaned_data.csv")
df1_24 = df1[df1["Age"] == 24].copy()

features_model1 = ['Labor_Income', 'Education', 'Gender']
X1 = pd.get_dummies(df1_24[features_model1], columns=['Education', 'Gender'], drop_first=True)
Y1 = df1_24['Employment']

# Add interaction terms
X1['Education_2:Gender_1'] = X1.get('Education_2', 0) * X1.get('Gender_1', 0)
X1['Education_3:Gender_1'] = X1.get('Education_3', 0) * X1.get('Gender_1', 0)

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=42, stratify=Y1)
X1_resampled, Y1_resampled = SMOTETomek(random_state=42).fit_resample(X1_train, Y1_train)

model1 = LogisticRegression(solver='liblinear', max_iter=1000)
model1.fit(X1_resampled, Y1_resampled)

# Coefficients
model1_coefficients = dict(zip(X1.columns, model1.coef_[0]))
model1_coefficients['Intercept'] = model1.intercept_[0]

print("=== Model 1 Coefficients (Age 24) ===")
for k, v in model1_coefficients.items():
    print(f"{k:<25} : {v:.4f}")
print("\n")

# Predictions
y1_pred = model1.predict(X1_test)
y1_probs = model1.predict_proba(X1_test)[:, 1]

# Metrics
print("=== Model 1 Evaluation Metrics ===")
print(f"Accuracy  : {accuracy_score(Y1_test, y1_pred):.3f}")
print(f"Precision : {precision_score(Y1_test, y1_pred):.3f}")
print(f"Recall    : {recall_score(Y1_test, y1_pred):.3f}")
print(f"F1 Score  : {f1_score(Y1_test, y1_pred):.3f}")
print(f"AUC       : {roc_auc_score(Y1_test, y1_probs):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(Y1_test, y1_pred))
print("\nClassification Report:")
print(classification_report(Y1_test, y1_pred))

# ROC Curve
fpr1, tpr1, _ = roc_curve(Y1_test, y1_probs)
auc1 = roc_auc_score(Y1_test, y1_probs)

plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(fpr1, tpr1, label=f'AUC = {auc1:.2f}', color='darkgreen', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve for Logistic Regression Model 1 (Age 24)', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=17, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=17, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==============================
# Logistic Model 2 (Reduced)
# ==============================
df2 = pd.read_csv("cleaned_data_model2.csv")
df2["Age_Squared"] = df2["Age"] ** 2
# Save df2
df2.to_csv("df2_saved.csv", index=False)

features_model2 = ['Age', 'Age_Squared', 'Labor_Income', 'Education', 'Gender', 'employed_lag']
X2 = pd.get_dummies(df2[features_model2], columns=['Education', 'Gender', 'employed_lag'], drop_first=True)
Y2 = df2['Employment']

# Add interactions
X2['Education_2:Gender_1'] = X2.get('Education_2', 0) * X2.get('Gender_1', 0)
X2['Education_3:Gender_1'] = X2.get('Education_3', 0) * X2.get('Gender_1', 0)
X2['Age:Labor_Income'] = X2['Age'] * X2['Labor_Income']

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=42, stratify=Y2)

model2 = LogisticRegression(solver='liblinear', max_iter=1000)
model2.fit(X2_train, Y2_train)

# Coefficients
model2_coefficients = dict(zip(X2_train.columns, model2.coef_[0]))
model2_coefficients['Intercept'] = model2.intercept_[0]

print("=== Model 2 Coefficients ===")
for k, v in model2_coefficients.items():
    print(f"{k:<25} : {v:.4f}")
print("\n")

# Predictions
y2_pred = model2.predict(X2_test)
y2_probs = model2.predict_proba(X2_test)[:, 1]

# Metrics
print("=== Model 2 Evaluation Metrics ===")
print(f"Accuracy  : {accuracy_score(Y2_test, y2_pred):.3f}")
print(f"Precision : {precision_score(Y2_test, y2_pred):.3f}")
print(f"Recall    : {recall_score(Y2_test, y2_pred):.3f}")
print(f"F1 Score  : {f1_score(Y2_test, y2_pred):.3f}")
print(f"AUC       : {roc_auc_score(Y2_test, y2_probs):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(Y2_test, y2_pred))
print("\nClassification Report:")
print(classification_report(Y2_test, y2_pred))

# ROC Curve
fpr2, tpr2, _ = roc_curve(Y2_test, y2_probs)
auc2 = roc_auc_score(Y2_test, y2_probs)

plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(fpr2, tpr2, label=f'AUC = {auc2:.2f}', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve for Logistic Regression Model 2', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=17, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=17, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


