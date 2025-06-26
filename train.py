# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.model import build_transformer
from utils.dataset import AmazonReviewDataset
from config import *

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load datasets
train_dataset = AmazonReviewDataset(TRAIN_CSV, tokenizer, SRC_SEQ_LEN)
test_dataset = AmazonReviewDataset(TEST_CSV, tokenizer, SRC_SEQ_LEN)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define model
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, D_MODEL, N_LAYERS, H, DROPOUT, D_FF)
model.to(device)

# Loss & optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Class weights for imbalance (example values, adjust after checking class distribution)
class_counts = torch.tensor([7000, 15000, 8000], dtype=torch.float)  # e.g., [neg, pos, neu]
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum() * 3  # Normalize so avg weight = 1
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),
    label_smoothing=0.1  # You can tune this between 0.05 to 0.2
)


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early Stopping
best_val_loss = float('inf')
patience_counter = 0
PATIENCE = 3

# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)

        src_mask = None  # No mask for now

        # Forward pass
        encoder_output = model.encode(src, src_mask)  # [batch_size, seq_len, d_model]
        cls_token = encoder_output[:, 0, :]  # [batch_size, d_model]
        output = model.project(cls_token)    # [batch_size, 3]

        # Loss
        loss = criterion(output, tgt)
        total_loss += loss.item()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            src_mask = None

            encoder_output = model.encode(src, src_mask)
            cls_token = encoder_output[:, 0, :]
            output = model.project(cls_token)

            loss = criterion(output, tgt)
            total_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy

# GPU info
print("Number of GPU: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name: ", torch.cuda.get_device_name())
print('Using device:', device)

# Training loop
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

# Save model
# torch.save(model.state_dict(), MODEL_PATH)
# print(f"Model saved at {MODEL_PATH}")
    if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print(f"Model improved. Saved at {MODEL_PATH}")
    else:
        patience_counter += 1
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model improved. Saved at {MODEL_PATH}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break


