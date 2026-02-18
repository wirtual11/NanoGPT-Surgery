from typing import Protocol
import torch
from torch.utils.data import Dataset


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

class GPTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, text: str, seq_length: int, tokenizer: Tokenizer) -> None:
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx:int):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y