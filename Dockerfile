FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000 for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy EVERYTHING (app.py, model.pt, vocab.pkl)
COPY --chown=user . /app

# HuggingFace Spaces uses port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
