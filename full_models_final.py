import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

######################## Logistic Model 1 ################################
df = pd.read_csv("cleaned_data.csv")

df["Age_Squared"] = df["Age"] ** 2

# Logistic Model 1 - all years
features = [
    'Age',
    'Age_Squared',
    'Labor_Income',
    'Insurance',
    'Education',
    'Gender',
    'Last_Marital',
    'Year'
]
X = df[features]
Y = df['Employment']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Insurance', 'Education', 'Gender',
                               'Last_Marital', 'Year'], drop_first=True)

# (1) Education x Gender
X['Education_2:Gender_1'] = X.get('Education_2', 0) * X.get('Gender_1', 0)
X['Education_3:Gender_1'] = X.get('Education_3', 0) * X.get('Gender_1', 0)

# (2) Age x Labor_Income
X['Age:Labor_Income'] = X['Age'] * X['Labor_Income']
X['Age:Labor_Income'] = X['Age'] * X['Labor_Income']

# (3) Income x Year
X['Labor_Income:Year_2021'] = X['Labor_Income'] * X.get('Year_2021', 0)
X['Labor_Income:Year_2019'] = X['Labor_Income'] * X.get('Year_2019', 0)
X['Labor_Income:Year_2017'] = X['Labor_Income'] * X.get('Year_2019', 0)
X['Labor_Income:Year_2015'] = X['Labor_Income'] * X.get('Year_2019', 0)

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y)

# Oversampler
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_train_res, Y_train_res = smt.fit_resample(X_train, Y_train)

# Combine into DataFrame for statsmodels
train_df = pd.DataFrame(X_train_res, columns=X_train.columns)
train_df['Employment'] = Y_train_res

# Create formula string dynamically for statsmodels
feature_cols = X_train.columns.tolist()
formula = 'Employment ~ ' + ' + '.join(feature_cols)

# Fit logistic regression using statsmodels
model1 = smf.logit(formula, data=train_df).fit()
print(model1.summary())

# Predict on test set
pred_probs = model1.predict(X_test)
y_pred = (pred_probs > 0.5).astype(int)

# Evaluate
cm = confusion_matrix(Y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

model1_coefficients = model1.params.to_dict()

model1_pred_probs = model1.predict(X_test)
model1_preds = (model1_pred_probs > 0.5).astype(int)

# Compute ROC curve and AUC for Model 1
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
fpr1, tpr1, _ = roc_curve(Y_test, model1_pred_probs)
auc1 = roc_auc_score(Y_test, model1_pred_probs)
# F1 score
f1_model1 = f1_score(Y_test, model1_preds)
print("F1 Score for Logistic Model 1:", f1_model1)
f1_macro_model1 = f1_score(Y_test, model1_preds, average='macro')
f1_weighted_model1 = f1_score(Y_test, model1_preds, average='weighted')

# Plot ROC Curve for Model 1
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='AUC = {:.2f}'.format(auc1))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve for Logistic Regression Model 1')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
# plt.show()

######################## Logistic Model 2 ################################
df = pd.read_csv("cleaned_data_model2.csv")

df["Age_Squared"] = df["Age"] ** 2

# Logistic Model 1 - all years
features = [
    'Age',
    'Age_Squared',
    'Labor_Income',
    'Insurance',
    'Education',
    'Gender',
    'Last_Marital',
    'Year',
    'employed_lag'
]
X = df[features]
Y = df['Employment']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Insurance', 'Education', 'Gender',
                               'Last_Marital', 'Year', 'employed_lag'], drop_first=True)

# (1) Education x Gender
X['Education_2:Gender_1'] = X.get('Education_2', 0) * X.get('Gender_1', 0)
X['Education_3:Gender_1'] = X.get('Education_3', 0) * X.get('Gender_1', 0)

# (2) Age_Group x Labor_Income
X['Age:Labor_Income'] = X['Age'] * X['Labor_Income']
X['Age:Labor_Income'] = X['Age'] * X['Labor_Income']

# (3) Income x Year
X['Labor_Income:Year_2021'] = X['Labor_Income'] * X.get('Year_2021', 0)
X['Labor_Income:Year_2019'] = X['Labor_Income'] * X.get('Year_2019', 0)
X['Labor_Income:Year_2017'] = X['Labor_Income'] * X.get('Year_2019', 0)
X['Labor_Income:Year_2015'] = X['Labor_Income'] * X.get('Year_2019', 0)

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y)

# Oversampler
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_train_res, Y_train_res = smt.fit_resample(X_train, Y_train)

# Combine into DataFrame for statsmodels
train_df = pd.DataFrame(X_train_res, columns=X_train.columns)
train_df['Employment'] = Y_train_res

# Create formula string dynamically for statsmodels
feature_cols = X_train.columns.tolist()
formula = 'Employment ~ ' + ' + '.join(feature_cols)

# Fit logistic regression using statsmodels
model2 = smf.logit(formula, data=train_df).fit()
print(model2.summary())

# Predict on test set
pred_probs = model2.predict(X_test)
y_pred = (pred_probs > 0.5).astype(int)

# Evaluate
cm = confusion_matrix(Y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

model2_coefficients = model2.params.to_dict()

model2_pred_probs = model2.predict(X_test)
model2_preds = (model2_pred_probs > 0.5).astype(int)

fpr2, tpr2, _ = roc_curve(Y_test, model2_pred_probs)
auc2 = roc_auc_score(Y_test, model2_pred_probs)
# F1 score
f1_model2 = f1_score(Y_test, model2_preds)
print("F1 Score for Logistic Model 2:", f1_model2)

# Plot ROC Curve for Model 2
plt.figure(figsize=(8, 6))
plt.plot(fpr2, tpr2, color='darkorange', lw=2, label='AUC = {:.2f}'.format(auc2))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve for Logistic Regression Model 2')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
# plt.show()

# Macro average
f1_macro_model2 = f1_score(Y_test, model2_preds, average='macro')

# Weighted average
f1_weighted_model2 = f1_score(Y_test, model2_preds, average='weighted')
print(f"F1 Macro (Model 1):     {f1_macro_model1:.3f}")
print(f"F1 Weighted (Model 1):  {f1_weighted_model1:.3f}")
print(f"F1 Macro (Model 2):     {f1_macro_model2:.3f}")
print(f"F1 Weighted (Model 2):  {f1_weighted_model2:.3f}")
