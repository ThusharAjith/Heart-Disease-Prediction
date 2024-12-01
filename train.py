import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('framingham.csv')

# Data Preprocessing
df.dropna(inplace=True)  # Remove any rows with NaNs
X = df.drop('TenYearCHD', axis=1)  # Features
y = df['TenYearCHD']  # Target

# Check for NaNs in features and target
print("Initial NaNs in X:", np.isnan(X).sum().sum())
print("Initial NaNs in y:", np.isnan(y).sum())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check for NaNs and Infs after SMOTE
print("NaNs after SMOTE:", np.isnan(X_train_resampled).sum())
print("Infs after SMOTE:", np.isinf(X_train_resampled).sum())

# Hyperparameter tuning with XGBoost using Stratified Cross-Validation
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42),
                           param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model from grid search
best_xgb_model = grid_search.best_estimator_

# Evaluate on the test set
y_pred = best_xgb_model.predict(X_test)
y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]

# Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Save the model
joblib.dump(best_xgb_model, 'models/xgboost_model.pkl')
print("XGBoost model trained and saved successfully!")
