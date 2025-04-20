import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report
import numpy as np
import torch
import os

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "modules", "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")
MODELS_DIR = os.path.join(PROJECT_ROOT, "modules", "models")

# ✅ Load your CSV datasets
train_df = pd.read_csv(os.path.join(DATA_DIR,"intent_train_final.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR,"intent_test_final.csv"))

# ✅ Encode labels
labels = sorted(train_df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
train_df['label'] = train_df['label'].map(label2id)
test_df['label'] = test_df['label'].map(label2id)

# ✅ Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True)

train_dataset = Dataset.from_pandas(train_df).map(tokenize)
test_dataset = Dataset.from_pandas(test_df).map(tokenize)

# ✅ Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ✅ Training config
training_args = TrainingArguments(
    output_dir=os.path.join(MODELS_DIR,"./intent_model"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir=os.path.join(MODELS_DIR,"./logs"),
    report_to="none"
)

# ✅ Evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels_true = eval_pred
    print(predictions)
    preds = np.argmax(predictions, axis=1)
    report = classification_report(labels_true, preds, target_names=labels, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ✅ Train!
trainer.train()

# ✅ Save model + tokenizer
trainer.save_model("intent_model")
tokenizer.save_pretrained("intent_model")

print("🎯 Fine-tuning complete! Model saved to ./intent_model")
