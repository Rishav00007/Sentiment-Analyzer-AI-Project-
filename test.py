import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.model import build_transformer
from utils.dataset import AmazonReviewDataset
from config import *  # This should have all your model parameters
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
test_dataset = AmazonReviewDataset(TEST_CSV, tokenizer, SRC_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
# Set up device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model (keep it as binary classification since that's how it was trained)
model = build_transformer(
    SRC_VOCAB_SIZE,
    3,  # Binary classification
    SRC_SEQ_LEN,
    TGT_SEQ_LEN,
    D_MODEL,
    N_LAYERS,
    H,
    DROPOUT,
    D_FF
)
model = model.to(device)  # Move model to the device

class_counts = torch.tensor([7000, 15000, 8000], dtype=torch.float)  # e.g., [neg, pos, neu]
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum() * 3  # Normalize so avg weight = 1
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Set to evaluation mode

# Load the tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Modify your dataset to ensure binary labels
# test_dataset = AmazonReviewDataset(
#     "data/test.csv",
#     tokenizer, 
#     src_seq_len=256
# )


# After creating the dataset
print("Checking dataset labels:")
for i in range(len(test_dataset)):
    _, label = test_dataset[i]
    #print(f"Sample {i} label:", label.item())

# After creating the dataset, add this debug code
print("\nChecking first few samples from CSV:")
for i in range(len(test_dataset)):
    tokens, label = test_dataset[i]
    # Get the original text from tokenizer
    text = tokenizer.decode(tokens[tokens != 0])  # Remove padding tokens
    #print(f"Review: {text[:50]}...")  # Print first 50 chars
    #print(f"Label: {label.item()}\n")

# Add this after creating your test_dataset
print("\nVerifying data from CSV:")
df = pd.read_csv("data/test.csv")
print("CSV contents:")
print(df['label'].value_counts())

# After loading the model
print("\nModel parameters:")
for name, param in model.named_parameters():
    if 'proj' in name:
        print(f"{name}: {param.shape}")

# After creating the dataset
print("\nDataset samples:")
for i in range(len(test_dataset)):
    _, label = test_dataset[i]
    #print(f"Sample {i} label:", label.item())

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_mask = None
            # Forward pass
    #         encoder_output = model.encode(inputs, None)
    #         # Take the first token's output ([CLS] token)
    #         cls_output = encoder_output[:, 0, :]
    #         logits = model.project(cls_output)  # Should give shape [batch_size, 2]
    #         _, predicted = torch.max(logits, dim=1)  # Get predictions
            
    #         # Store predictions and actual labels
    #         predictions.extend(predicted.cpu().numpy())
    #         actuals.extend(labels.cpu().numpy())
    
    # return np.array(predictions), np.array(actuals)

            encoder_output = model.encode(inputs, inputs_mask)
            cls_token = encoder_output[:, 0, :]  # Only use [CLS] token output
            output = model.project(cls_token)

            
            #loss = criterion(output, labels)
            #total_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            # logits = model.project(cls_token)    # [batch_size, 3]

            # # Get predicted class index
            # #prediction = torch.softmax(logits, dim=1)
            # #prediction = torch.argmax(logits, dim=1).item()
            # preds = torch.argmax(logits, dim=1)  # [batch_size]

            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(actuals)

# Run evaluation
predictions, actuals = evaluate(model, test_loader, criterion, device)

# Print results
# print("\nTest Results:")
# print(f"Number of test samples: {len(predictions)}")
# print(f"Accuracy: {(predictions == actuals).mean():.4f}")
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Show test results
print("\nTest Results:")
print(f"Total Test Samples: {len(predictions)}")
print(f"Accuracy: {(predictions == actuals).mean():.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(actuals, predictions, target_names=['Negative', 'Positive', 'Neutral'], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(actuals, predictions)
print("\nConfusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive", "Neutral"], yticklabels=["Negative", "Positive", "Neutral"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

