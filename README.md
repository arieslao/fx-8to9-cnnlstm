# FX CNN-LSTM (1h) — Predicting the 8→9 a.m. London Move

This repo trains a CNN-LSTM on **1-hour Forex bars** to predict whether the **9:00** London close will be above the **8:00** London close.

### Quickstart

1) Clone and set up:
```bash
git clone <your_repo_url> fx-8to9-cnnlstm
cd fx-8to9-cnnlstm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set FX_TICKERS (e.g., EURUSD=X,GBPUSD=X)
