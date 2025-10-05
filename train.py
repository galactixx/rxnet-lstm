"""
Training script for the RxNet LSTM model.

Loads RxNorm data, trains the character-level LSTM model with teacher forcing,
and implements early stopping with exponential moving average evaluation.
"""

from typing import List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from constants import PAD, SEED
from processing import load_rxnorm_data
from rxlstm import LSTMModule, RXNetLSTM
from utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: RXNetLSTM,
    loader: DataLoader,
    criterion: CrossEntropyLoss,
    ema: ExponentialMovingAverage,
    pad_id: int,
) -> Tuple[float, float]:
    """Evaluate with EMA-averaged weights on a data loader.

    Temporarily swaps model parameters with their EMA averages for evaluation
    and restores them afterward.
    """
    model.eval()
    ema.store()
    ema.copy_to()

    total = correct = running_loss = 0

    with torch.no_grad():
        for seqs, seqs_lens in tqdm(loader):
            seqs = seqs.to(device)

            seqs_in = seqs[:, :-1]
            seqs_out = seqs[:, 1:]

            logits = model(seqs_in, seqs_lens - 1, 1)
            loss = criterion(logits.transpose(1, 2), seqs_out)
            running_loss += loss.item() * seqs.size(0)

            pred = logits.argmax(dim=2)
            mask = seqs_out != pad_id
            correct += ((seqs_out == pred) & mask).sum().item()
            total += mask.sum().item()

    ema.restore()
    return running_loss / len(loader.dataset), correct / total


if __name__ == "__main__":
    seed_everything(seed=SEED)

    config = load_rxnorm_data()
    pairs = [config.encode(context=name) for name in config.names]

    train_names, test_names = train_test_split(pairs, test_size=0.2, random_state=SEED)

    PAD_ID = config.char_to_id[PAD]

    def get_padded_seq(seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
        return pad_sequence(seqs, batch_first=True, padding_value=pad_id)

    def get_lengths(seqs: List[torch.Tensor]) -> torch.Tensor:
        return torch.tensor([len(seq) for seq in seqs], dtype=torch.long)

    def collate_fn(names: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = get_padded_seq(seqs=names, pad_id=PAD_ID)
        padded_lens = get_lengths(seqs=names)
        return padded, padded_lens

    class RxNames(Dataset):
        def __init__(self, names: List[List[int]]) -> None:
            super().__init__()
            self.names = names

        def __len__(self) -> int:
            """Return dataset size."""
            return len(self.names)

        def __getitem__(self, idx: int) -> torch.Tensor:
            """Get one sample."""
            name = self.names[idx]
            return torch.tensor(name, device=device)

    train_dataset = RxNames(names=train_names)
    test_dataset = RxNames(names=test_names)

    g = torch.Generator()
    g.manual_seed(SEED)

    # Data loaders: shuffle for training, deterministic for evaluation.
    trainloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, generator=g, batch_size=64
    )
    testloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=32
    )

    EPOCHS = 100
    PATIENCE = 7

    no_improve, best_loss = 0, float("inf")

    # Hidden layer widths scale with context size.

    module = LSTMModule(embed=64, hidden=256, layers=2, dropout=0.20)
    model = RXNetLSTM(vocab=len(config.vocab), module=module, pad_idx=PAD_ID)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
    criterion = CrossEntropyLoss(ignore_index=PAD_ID)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    scaler = GradScaler()

    step = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for seqs, seqs_lens in tqdm(trainloader):
            seqs = seqs.to(device)

            seqs_in = seqs[:, :-1]
            seqs_out = seqs[:, 1:]

            p = max(0.5, 1.0 - step / 2_000)
            optimizer.zero_grad()

            with autocast():
                logits = model(seqs_in, seqs_lens - 1, p)
                loss = criterion(logits.transpose(1, 2), seqs_out)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            running_loss += loss.item() * seqs.size(0)

            step += 1

        train_loss = running_loss / len(trainloader.dataset)
        val_loss, val_acc = evaluate(model, testloader, criterion, ema, PAD_ID)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS}.. "
            f"Probability: {p:.3f}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Val token loss: {val_loss:.3f}.. "
            f"Val token accuracy: {val_acc:.3f}.."
        )

        if val_loss < best_loss:
            no_improve = 0
            best_loss = val_loss
        else:
            no_improve += 1
            if no_improve > PATIENCE:
                break
