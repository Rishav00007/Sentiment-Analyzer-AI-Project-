
from transformers import BertTokenizer
from models.model import build_transformer
from config import *
import torch

def predict_sentiment(text, model, tokenizer, device, max_len=128):
    # Tokenize
    encoded = tokenizer.encode(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(encoded, None)
        cls_output = encoder_output[:, 0, :]
        logits = model.project(cls_output)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1).squeeze(0)  # Now it's shape [3], 1D tensor

        prediction = torch.argmax(probs).item()  # Get the index of the max probability

        # Debug: Print probabilities directly as a 1D tensor
        print(f"Probabilities: Negative={probs[0]:.3f}, Positive={probs[1]:.3f}, Neutral={probs[2]:.3f}")

        # Convert to label
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        
        return sentiment, probs

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = build_transformer(
    SRC_VOCAB_SIZE,
    3,  # 3-class classification
    SRC_SEQ_LEN,
    TGT_SEQ_LEN,
    D_MODEL,
    N_LAYERS,
    H,
    DROPOUT,
    D_FF
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
