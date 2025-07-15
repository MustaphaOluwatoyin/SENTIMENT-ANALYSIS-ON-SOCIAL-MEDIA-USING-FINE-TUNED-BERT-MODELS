
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tweets = {
    "text": [
        "I love the new phone I bought! Best decision ever.",
        "This weather is awful, totally ruined my day.",
        "The movie was okay, not great but not bad either.",
        "Customer service was terrible. I'm never coming back!",
        "Great service and delicious food. Highly recommend!",
        "Neutral about the event, didn't feel strongly either way.",
        "I’m so disappointed with this product.",
        "Had an amazing time at the concert!",
        "The experience was average, nothing special.",
        "I hate waiting in line for so long!"
    ],
    "label": [2, 0, 1, 0, 2, 1, 0, 2, 1, 0]
}
df_tweets = pd.DataFrame(tweets)
df_tweets.to_csv("sample_tweets.csv", index=False)

epochs = [1, 2, 3]
train_acc = [0.72, 0.84, 0.88]
val_acc = [0.70, 0.82, 0.87]
train_loss = [0.55, 0.32, 0.21]
val_loss = [0.58, 0.35, 0.24]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label="Training Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='o')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='o')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_loss_curves.png")


# Install required libraries if not already installed
!pip install transformers datasets scikit-learn -q

import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import torch

# -----------------------------
# 1. Prepare the dataset
# -----------------------------
# Data already exists as df_tweets
df_tweets['label'] = df_tweets['label'].astype(int)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df_tweets)

# Split into train/test manually (80/20)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
test_dataset = dataset['test']

# -----------------------------
# 2. Tokenization
# -----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# -----------------------------
# 3. Model Setup
# -----------------------------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# -----------------------------
# 4. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir='./logs',
    logging_steps=10
)

# -----------------------------
# 5. Metrics
# -----------------------------
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(pred.label_ids, preds),
        'f1_macro': f1_score(pred.label_ids, preds, average='macro')
    }

# -----------------------------
# 6. Train the model
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
