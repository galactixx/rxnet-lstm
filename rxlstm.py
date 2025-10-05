"""
RxNet LSTM model implementation.

Defines the LSTM architecture for character-level language modeling with
teacher forcing and dropout support.
"""

from dataclasses import dataclass
from typing import List

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class LSTMModule:
    embed: int
    hidden: int
    layers: int
    dropout: float


class RXNetLSTM(torch.nn.Module):
    def __init__(self, vocab: int, module: LSTMModule, pad_idx: int) -> None:
        super().__init__()
        self.embeds = torch.nn.Embedding(vocab, module.embed, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(
            module.embed,
            module.hidden,
            module.layers,
            dropout=module.dropout,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(module.hidden, vocab)

    def forward(
        self, seq: torch.Tensor, seq_lens: torch.Tensor, teacher_p: float
    ) -> torch.Tensor:
        batch_size = seq.size(0)

        h = torch.zeros(
            self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device
        )
        c = torch.zeros_like(h)

        logits_t = seq[:, 0]
        outputs: List[torch.Tensor] = []
        for t in range(seq.size(1)):
            if not t:
                seq_in_t = seq[:, t]
            else:
                use_teacher = torch.rand(batch_size, device=device) < teacher_p
                seq_in_t = torch.where(use_teacher, seq[:, t], logits_t.argmax(dim=1))
            seq_in_t = self.embeds(seq_in_t)

            seq_out_t, (h, c) = self.lstm(seq_in_t.unsqueeze(1), (h, c))
            logits_t = self.fc(seq_out_t).squeeze(1)
            outputs.append(logits_t.unsqueeze(1))

        logits = torch.cat(outputs, dim=1)
        return logits
