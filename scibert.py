import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and preprocess the data
data = pd.read_json("data/manual_labelled.jsonl", lines=True)

# Split data into features and labels
X = data["abstract"].values
y = data["label"].values

# Initialize SciBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=2)

# Define cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Define evaluation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Perform cross-validation
fold = 1
with open("data/scibert.csv", "w") as file:
    file.write("sample_id,prediction_probability\n")
    for train_index, val_index in skf.split(X, y):
        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Tokenize inputs
        val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)

        # Prepare validation dataset
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_encodings['input_ids']),
                                                      torch.tensor(val_encodings['attention_mask']),
                                                      torch.tensor(y_val))

        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Evaluate the model
        result = trainer.predict(val_dataset)

        # Write prediction probabilities to file
        for sample_id, prob in zip(data.iloc[val_index]["id"].values, result.predictions):
            file.write(f"{sample_id},{prob[1]}\n")

        fold += 1
