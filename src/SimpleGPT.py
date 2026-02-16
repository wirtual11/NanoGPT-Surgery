
import torch
import torch.nn as nn

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        
        # Custom blocks injected.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(n_embd, n_head, batch_first=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        b, t = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for layer in self.layers: x = layer(x)
        return self.lm_head(self.ln_f(x))