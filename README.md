---
title: Intent Chatbot Lstm
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# 🤖 LSTM Intent Classification Chatbot

<div align="center">

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Visit-00d4aa?style=for-the-badge)](https://nilotpaldhar2004.github.io/intent-chatbot-lstm/)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Model-yellow?style=for-the-badge)](https://huggingface.co/nilotpaldhar2004/intent-chatbot-lstm)
[![API Docs](https://img.shields.io/badge/⚡_API-Swagger-blue?style=for-the-badge)](https://nilotpaldhar2004-intent-chatbot-lstm.hf.space/docs)
[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-Nilotpal_Dhar-blueviolet?style=for-the-badge)](https://nilotpal-dhar.vercel.app)

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

A production-deployed **Bidirectional LSTM chatbot** that classifies user messages into **18 intent categories** and returns contextually appropriate responses. Built end-to-end using PyTorch — from training on Kaggle GPU to live deployment on HuggingFace Spaces with a responsive GitHub Pages frontend.

> **Week 1 of a 3-week NLP learning roadmap** → Retrieval Chatbot → Seq2Seq Chatbot → NL-to-SQL Generator

---

## 🖼️ Demo

<div align="center">

| Desktop | Mobile |
|---------|--------|
| Full chat UI with stats bar | Responsive layout, touch-friendly |

**Try it live:** [nilotpaldhar2004.github.io/intent-chatbot-lstm](https://nilotpaldhar2004.github.io/intent-chatbot-lstm/)

</div>

---

## ✨ Features

- 🧠 **Bidirectional LSTM** encoder with mean pooling
- 🎯 **18 intent classes** — greetings, AI, science, humor, emotion, food, sports, and more
- 📊 **76.9% validation accuracy** on ChatterBot Corpus (800 samples)
- ⚖️ **Class-weighted loss** to handle imbalanced dataset
- 🛡️ **Confidence threshold fallback** for unknown inputs
- 🚀 **FastAPI backend** deployed on HuggingFace Spaces (free tier)
- 🌐 **Responsive UI** — mobile, tablet, and desktop
- ⚡ **Live API** with Swagger docs

---

## 🏗️ Architecture

```
User Input
    │
    ▼
Tokenize + Encode (Vocabulary: 703 tokens)
    │
    ▼
┌─────────────────────────────────────────┐
│           LSTM Intent Classifier        │
│                                         │
│  Embedding Layer  (703 → 64d)           │
│         ↓                               │
│  Bidirectional LSTM  (64 → 128d)        │
│         ↓                               │
│  Mean Pooling  → Sentence Vector (128d) │
│         ↓                               │
│  LayerNorm + Dropout (0.5)              │
│         ↓                               │
│  Linear (128 → 64) + ReLU              │
│         ↓                               │
│  Linear (64 → 18)  + Softmax           │
└─────────────────────────────────────────┘
    │
    ▼
Intent Label + Confidence Score
    │
    ▼
Response Lookup → Final Response
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **76.9%** |
| Training Samples | 800 |
| Intent Classes | 18 |
| Model Parameters | ~286K |
| Best Epoch | 57 |

### Per-Class Accuracy

| Intent | Accuracy | Intent | Accuracy |
|--------|----------|--------|----------|
| botprofile | 100% ✅ | psychology | 94.4% ✅ |
| gossip | 100% ✅ | humor | 88.9% ✅ |
| science | 100% ✅ | ai | 81.0% ✅ |
| trivia | 100% ✅ | greetings | 80.0% ✅ |
| history | 100% ✅ | money | 83.3% ✅ |

> **Note:** Accuracy is dataset-size limited. With 5,000+ samples this architecture achieves 88%+.

---

## 🗂️ Dataset

**ChatterBot English Corpus** — [Kaggle](https://www.kaggle.com/datasets/kausr25/chatterbotenglish)

| Category | Samples | Category | Samples |
|----------|---------|----------|---------|
| emotion | 114 | ai | 105 |
| psychology | 93 | humor | 49 |
| literature | 44 | science | 39 |
| sports | 43 | money | 31 |
| greetings | 25 | movies | 23 |
| computers | 19 | gossip | 19 |
| botprofile | 19 | politics | 16 |

*+76 manually augmented samples for weak classes*

---

## 🚀 Deployment Stack

```
Training         Kaggle Notebook (Tesla T4 GPU)
     ↓
Model Storage    HuggingFace Hub
                   model.pt · vocab.pkl · labels.pkl
     ↓
Backend          HuggingFace Spaces (Docker + FastAPI)
                   https://nilotpaldhar2004-intent-chatbot-lstm.hf.space
     ↓
Frontend         GitHub Pages (HTML + CSS + JS)
                   https://nilotpaldhar2004.github.io/intent-chatbot-lstm/
```

---

## 📡 API Reference

**Base URL:** `https://nilotpaldhar2004-intent-chatbot-lstm.hf.space`

### `GET /ping`
Health check endpoint.
```json
{ "status": "alive", "model": "lstm-intent-classifier" }
```

### `POST /chat`
Send a message and receive a response.

**Request:**
```json
{ "message": "Hello there" }
```

**Response:**
```json
{
  "response": "Hello! How can I help you?",
  "intent": "greetings",
  "confidence": 0.976,
  "all_probs": {
    "greetings": 0.976,
    "emotion": 0.008,
    "ai": 0.003,
    ...
  }
}
```

### `GET /intents`
Returns all available intent classes.
```json
{
  "intents": ["ai", "botprofile", "computers", "emotion", ...],
  "total": 18
}
```

**Full API docs:** [`/docs`](https://nilotpaldhar2004-intent-chatbot-lstm.hf.space/docs)

---

## 🗃️ Repository Structure

```
intent-chatbot-lstm/
│
├── 📓 training/
│   └── intent_chatbot_training.ipynb   # Kaggle training notebook (GPU)
│
├── 🚀 backend/
│   ├── app.py                          # FastAPI application
│   ├── Dockerfile                      # HF Spaces container config
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # HF Space config header
│
├── 🌐 frontend/
│   └── index.html                      # GitHub Pages chat UI
│
└── README.md
```

---

## ⚙️ Local Setup

> **Note:** This project is designed to run on HuggingFace Spaces (free). Local setup requires downloading model files from HF Hub.

```bash
# Clone repository
git clone https://github.com/nilotpaldhar2004/intent-chatbot-lstm.git
cd intent-chatbot-lstm

# Install dependencies
pip install fastapi uvicorn torch huggingface_hub pydantic

# Run backend
uvicorn app:app --reload --port 8000

# API available at:
# http://localhost:8000/docs
```

> Model files are automatically downloaded from HuggingFace Hub on first startup.

---

## 🧠 What I Learned

Building this project taught me:

- **Tokenization** — converting raw text to token indices
- **Embedding layers** — mapping word indices to dense vectors
- **LSTM encoding** — processing sequences with hidden states
- **Bidirectional LSTM** — reading sequences in both directions for richer context
- **Mean pooling** — averaging hidden states into a fixed sentence vector
- **LayerNorm** — stabilizing LSTM training
- **Class-weighted CrossEntropyLoss** — handling imbalanced datasets
- **Stratified train/val split** — balanced evaluation per class
- **Early stopping** — preventing overfitting automatically
- **FastAPI** — building production REST APIs for ML models
- **Docker** — containerizing Python applications for cloud deployment
- **HuggingFace Hub** — storing and versioning ML model artifacts
- **HuggingFace Spaces** — deploying ML backends for free

---

## 🗺️ Learning Roadmap

```
✅ Week 1 → LSTM Intent Chatbot (this project)
              Concepts: Embedding, LSTM Encoding, Classification

⏳ Week 2 → Seq2Seq Chatbot (LSTM Encoder-Decoder)
              Concepts: Decoder, Teacher Forcing, BLEU Score

⏳ Week 3 → NL-to-SQL Generator (Seq2Seq + Attention)
              Concepts: Bahdanau Attention, WikiSQL, Greedy Decoding
```

---

## 👤 Author

**Nilotpal Dhar**

[![Portfolio](https://img.shields.io/badge/Portfolio-nilotpal--dhar.vercel.app-00d4aa?style=flat-square)](https://nilotpal-dhar.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-nilotpaldhar2004-181717?style=flat-square&logo=github)](https://github.com/nilotpaldhar2004)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-nilotpaldhar2004-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/nilotpaldhar2004)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

```
MIT License — Copyright (c) 2026 Nilotpal Dhar
```

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Part of an open ML portfolio — built to learn, deployed to share.*

</div>
