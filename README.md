# RxNet-LSTM

An LSTM neural network that learns and predicts pharmaceutical brand names. This extends the original [RxNet](https://github.com/galactixx/rxnet) MLP implementation (based on Bengio et al. 2003) with a 2-layer LSTM architecture.

## Architecture

- **Embedding**: `Embedding(vocab_size, 64)`
- **LSTM**: 2-layer LSTM with hidden size 256 and dropout (0.20)
- **Output**: `Linear(256, vocab_size)`

Uses teacher forcing, EMA evaluation, and mixed precision training. Results are roughly comparable to the original MLP, with marginal improvements.

## Quick Start

```bash
pip install torch pandas scikit-learn tqdm torch-ema
python train.py
```

## Data

Uses `rxnorm-names.csv` with RxNorm pharmaceutical brand names. Character-level modeling with `<BOS>`, `<EOS>`, and `<PAD>` tokens.
