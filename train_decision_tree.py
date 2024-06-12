
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

def traverse_tree(tree, node, sample, feature_names, path=[]):
    # Base condition: If it's a leaf node
    if tree.children_left[node] == tree.children_right[node]:
        return path

    # Decision condition: Go left or right
    feature_index = tree.feature[node]
    threshold = tree.threshold[node]
    feature_name = feature_names[feature_index]
    feature_value = sample[feature_index]

    if feature_value <= threshold:
        path.append(f"{feature_name} <= {threshold}")
        return traverse_tree(tree, tree.children_left[node], sample, feature_names, path)
    else:
        path.append(f"{feature_name} > {threshold}")
        return traverse_tree(tree, tree.children_right[node], sample, feature_names, path)


# Load data
with open('data/manual_labelled.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Process categories: tokenize and keep the top 20 high frequency categories
all_categories = [item for sublist in [sample['category'].split(';') for sample in data] for item in sublist]
top_20_categories = [cat[0] for cat in Counter(all_categories).most_common(20)]
mlb = MultiLabelBinarizer(classes=top_20_categories)

# Extract features and labels
X = mlb.fit_transform([sample['category'].split('; ') for sample in data])
y = [int(sample['label']) for sample in data]
ids = [sample['id'] for sample in data]

# Initialize a decision tree classifier
clf = DecisionTreeClassifier(max_depth=2)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
output = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    ids_test = np.array(ids)[test_index]

    # Train decision tree
    clf.fit(X_train, y_train)

    # Predict probabilities
    probs = clf.predict_proba(X_test)[:, 1]

    # Find threshold that yields a precision higher than 0.9
    thresholds = np.linspace(0, 1, 200)
    best_threshold = 0.5  # default
    best_precision = 0
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        precision = precision_score(y_test, preds)
        if precision > best_precision:
            best_precision = precision
            best_threshold = threshold

        if best_precision > 0.9:
            break

    # Use the best threshold to get predictions
    preds = (probs > best_threshold).astype(int)

    # For each positive prediction, print the decision path
    for i, pred in enumerate(preds):
        if pred == 1:
            path = traverse_tree(clf.tree_, 0, X_test[i], top_20_categories)
            print(f"Decision path for sample {ids_test[i]}:")
            for step in path:
                print(step)
            print("\n")

    output.extend(list(zip(ids_test, y_test, preds)))

# Write predictions to output file
with open('data/category_decision_tree.csv', 'w') as f:
    f.write('sample_id,prediction_probability\n')
    for id_, label, pred in output:
        f.write(f"{id_},{pred}\n")


# new_data = []
# with open("../../../../final_data/202310/data/random_50000.csv", 'r') as f:
#     new_data = [json.loads(line) for line in f]
#
# # Extract features from the new data
# X_new = mlb.transform([sample['category'].split('; ') for sample in new_data])
#
# # Make predictions using the loaded decision tree model
# new_preds = clf.predict(X_new)
#
# # Write predictions to an output file
# with open('category_decision_tree.csv', 'w') as f:
#     f.write('sample_id,prediction\n')
#     for id_, pred in zip([sample['id'] for sample in new_data], new_preds):
#         f.write(f"{id_},{pred}\n")