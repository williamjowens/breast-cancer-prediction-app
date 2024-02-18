# Imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from ucimlrepo import fetch_ucirepo, list_available_datasets
from joblib import dump

# Load the dataset
breast_cancer_ds = fetch_ucirepo(id=15)

# Define features and labels
X = breast_cancer_ds.data.features
y = breast_cancer_ds.data.targets

# Modify feature names
X.columns = X.columns.str.lower()

# Redefine labels
y = y.replace({2: 0, 4: 1})
y = y.values.ravel()
y.shape

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Create pipeline workflow
pipeline = Pipeline(steps=[
    ("imputer", IterativeImputer(random_state=42)),
    ("scaler", StandardScaler()),
    ("classifier", XGBClassifier(random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print (f"Accuracy: {accuracy}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Extract values
tn, fp, fn, tp = conf_matrix.ravel()

# Create table
conf_matrix_table = pd.DataFrame({
    "Metric": ["True Negative", "False Positive", "False Negative", "True Positive"],
    "Count": [tn, fp, fn, tp]
})
print(conf_matrix_table)

# Save the model
model_filename = "/Users/williamjowens/Desktop/ANA 680/Week 2/best_model_xgb.pkl"
dump(pipeline, model_filename)

model_filename