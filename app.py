"""
Intent Classification Chatbot — FastAPI Backend
Deploy on HuggingFace Spaces (Docker SDK)
Author: Nilotpal Dhar
"""

import os
import re
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title       = "Intent Chatbot API",
    description = "LSTM Intent Classification Chatbot",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Config ───────────────────────────────────────────────────
DEVICE   = torch.device("cpu")   # HF Spaces free CPU only
CONFIDENCE_THRESHOLD = 0.30

# ── Model Definition (must match training exactly) ───────────
class LSTMIntentClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim,
                 n_layers, dropout, num_classes,
                 pad_idx=0, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.n_directions  = 2 if bidirectional else 1

        self.embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size    = emb_dim,
            hidden_size   = hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0,
            bidirectional = bidirectional
        )
        lstm_output_dim     = hidden_dim * self.n_directions
        self.layer_norm     = nn.LayerNorm(lstm_output_dim)
        self.dropout        = nn.Dropout(dropout)
        self.classifier     = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths):
        embedded     = self.dropout(self.embedding(x))
        packed       = pack_padded_sequence(
            embedded, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        output, _     = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        mask     = (x != 0).unsqueeze(-1).float()
        output   = output * mask
        sum_out  = output.sum(dim=1)
        count    = mask.sum(dim=1).clamp(min=1)
        mean_out = sum_out / count
        normed   = self.layer_norm(mean_out)
        return self.classifier(normed)


# ── Global State ─────────────────────────────────────────────
model       = None
vocab       = None
labels_data = None


# ── Helpers ──────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def encode(sentence: str, max_len: int = 20):
    unk    = vocab["token2idx"].get("<unk>", 1)
    tokens = preprocess(sentence).split()[:max_len]
    return [vocab["token2idx"].get(t, unk) for t in tokens]


def predict(text: str):
    tokens = encode(text)
    if not tokens:
        return "unknown", 0.0, {}

    x       = torch.tensor([tokens], dtype=torch.long)
    lengths = torch.tensor([len(tokens)])

    with torch.no_grad():
        logits = model(x, lengths)
        probs  = F.softmax(logits, dim=1)[0]

    idx2intent  = labels_data["idx2intent"]
    best_idx    = probs.argmax().item()
    intent      = idx2intent[str(best_idx)]
    confidence  = probs[best_idx].item()

    all_probs = {
        idx2intent[str(i)]: round(probs[i].item(), 4)
        for i in range(len(probs))
    }

    return intent, confidence, all_probs


# ── Startup — Load All Artifacts LOCALLY ─────────────────────
@app.on_event("startup")
async def load_artifacts():
    global model, vocab, labels_data

    print("Loading artifacts locally from Docker container...")

    # Load vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load labels
    with open("labels.pkl", "rb") as f:
        labels_data = pickle.load(f)

    # Fix idx2intent keys → must be strings for JSON compat
    labels_data["idx2intent"] = {
        str(k): v for k, v in labels_data["idx2intent"].items()
    }

    # Load model
    checkpoint = torch.load("model.pt", map_location=DEVICE)
    cfg        = checkpoint["model_config"]

    model = LSTMIntentClassifier(
        vocab_size    = cfg["vocab_size"],
        emb_dim       = cfg["emb_dim"],
        hidden_dim    = cfg["hidden_dim"],
        n_layers      = cfg["n_layers"],
        dropout       = cfg["dropout"],
        num_classes   = cfg["num_classes"],
        bidirectional = True
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model loaded. Classes: {cfg['num_classes']}")
    print("Ready to serve requests.")


# ── Endpoints ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response   : str
    intent     : str
    confidence : float
    all_probs  : dict


@app.get("/")
def root():
    return {
        "message" : "Intent Chatbot API is running",
        "docs"    : "/docs",
        "health"  : "/ping"
    }


@app.get("/ping")
def ping():
    return {"status": "alive", "model": "lstm-intent-classifier"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        return ChatResponse(
            response   = "Please say something!",
            intent     = "unknown",
            confidence = 0.0,
            all_probs  = {}
        )

    intent, confidence, all_probs = predict(req.message)

    # Low confidence fallback
    if confidence < CONFIDENCE_THRESHOLD:
        return ChatResponse(
            response   = "I'm not sure I understand. Could you rephrase that?",
            intent     = intent,
            confidence = confidence,
            all_probs  = all_probs
        )

    # Get random response for detected intent
    responses      = labels_data.get("responses", {})
    intent_resps   = responses.get(intent, [])
    response_text  = (
        random.choice(intent_resps)
        if intent_resps
        else "I don't have a response for that yet."
    )

    return ChatResponse(
        response   = response_text,
        intent     = intent,
        confidence = round(confidence, 4),
        all_probs  = all_probs
    )


@app.get("/intents")
def get_intents():
    """Return all available intents."""
    return {
        "intents"     : labels_data.get("all_intents", []),
        "total"       : labels_data.get("num_classes", 0)
    }
