"""
Data processing utilities for RxNorm names.

Loads and processes RxNorm drug names, creates character vocabulary, and provides
encoding functionality for the LSTM model.
"""

import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, List

from constants import BOS, EOS, PAD

warnings.filterwarnings("ignore")

import pandas as pd


@dataclass(frozen=True)
class RxNormDataConfig:
    names: List[str]
    vocab: List[str]
    char_to_id: Dict[str, int]
    id_to_char: Dict[int, str]

    def encode(self, context: str) -> List[int]:
        bos_id = self.char_to_id[BOS]
        eos_id = self.char_to_id[EOS]
        return [bos_id] + [self.char_to_id[char] for char in context] + [eos_id]


def load_rxnorm_data() -> RxNormDataConfig:
    # Read CSV with expected RxNorm string column `STR` and normalize to NFC.
    data = pd.read_csv("rxnorm-names.csv")
    names = data.STR.apply(lambda s: unicodedata.normalize("NFC", s.strip())).tolist()

    # Collect all unique characters present across names (excluding specials).
    chars = sorted(set(char for name in names for char in name))

    # Special tokens: start-of-sequence and end-of-sequence.
    specials = [BOS, EOS, PAD]

    vocab = specials + chars
    char_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_char = {i: c for c, i in char_to_id.items()}
    return RxNormDataConfig(
        names=names,
        vocab=vocab,
        char_to_id=char_to_id,
        id_to_char=id_to_char,
    )
