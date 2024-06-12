import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

# Load data from CSV files
data_A = pd.read_csv("data/category_decision_tree.csv")
data_B = pd.read_csv("data/keywords_match.csv")
data_C = pd.read_csv("data/scibert.csv")

# Merge data from A, B, and C
data = pd.merge(data_A, data_B, on="sample_id")
data = pd.merge(data, data_C, on="sample_id")

# Load real labels from JSONL file
real_labels = {}
with open("data/manual_labelled.jsonl", "r") as f:
    for line in f:
        item = eval(line)
        real_labels[item['id']] = int(item['label'])

# Add real labels to the merged data
data['real_label'] = data['sample_id'].map(real_labels)

# Separate features and labels
X = data[['prediction_probability_x', 'prediction_probability_y', 'prediction_probability']].values
y = data['real_label'].values

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')

# Define cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform 5-fold cross-validation
predicted_labels = cross_val_predict(svm_classifier, X, y, cv=skf)

# Evaluate performance
accuracy = accuracy_score(y, predicted_labels)
precision = precision_score(y, predicted_labels)
recall = recall_score(y, predicted_labels)
f1 = f1_score(y, predicted_labels)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)