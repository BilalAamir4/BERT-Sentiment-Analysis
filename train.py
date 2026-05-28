# Dataset: IMDB Movie Reviews (HuggingFace datasets)
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import sys

# 0. Check if model is already trained
if os.path.exists("./saved_model"):
    print("Model already trained. Delete ./saved_model to retrain.")
    sys.exit(0)

# 1. Load the IMDB dataset from HuggingFace datasets
print("Loading the IMDB dataset...")
dataset = load_dataset("imdb")

# 2. Load bert-base-uncased tokenizer and model
print("Loading bert-base-uncased model and tokenizer...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenize the dataset
def tokenize_function(examples):
    # Tokenize with truncation and padding to max_length of 128
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("Tokenizing the dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# 4. Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy and F1 score (weighted) using sklearn
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": acc, "f1": f1}

# 5. Set up TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=100,
)

# 6. Initialize Trainer
print("Initializing the Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. Call trainer.train() then trainer.evaluate()
print("Starting training...")
trainer.train()

print("Evaluating the model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 8. Save the final model and tokenizer to ./saved_model using save_pretrained()
print("Saving the final model and tokenizer to ./saved_model...")
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# 9. Generate and print a full classification report and plot confusion matrix
print("Generating evaluation metrics on test set...")
predictions = trainer.predict(eval_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Print classification report
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["negative", "positive"]))

# Plot and save a confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["negative", "positive"])
plt.yticks(tick_marks, ["negative", "positive"])

plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add values inside the confusion matrix plot
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix as confusion_matrix.png")

# 10. Print Training Summary
print("\n--- Training Summary ---")
print(f"Best Eval Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
print(f"Best Eval F1: {eval_results.get('eval_f1', 'N/A'):.4f}")
print("Model saved to: ./saved_model")
print("------------------------\n")
