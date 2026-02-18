from CharacterTokenizer import CharacterTokenizer
from LoRALinear import LoRALinear
from GPTDataset import GPTDataset
from SimpleGPT import SimpleGPT
from torch.utils.data import DataLoader

import torch.nn as nn
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model: SimpleGPT, tokenizer: CharacterTokenizer,
              prompt: str, device: str, max_new_tokens: int = 50, 
              temp: float = 1.0, top_k: int = 5) -> str:
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -256:]  # Crop to block_size
        logits: torch.Tensor = model(idx_cond)[:, -1, :] / temp
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    tokens: list[int] = idx[0].tolist()  # type: ignore[assignment]
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_text: str = "This is a test of the emergency broadcast system. This is only a test."

    # 1. Setup
    tokenizer: CharacterTokenizer = CharacterTokenizer(sample_text)
    dataset: GPTDataset = GPTDataset(sample_text, seq_length=16, tokenizer=tokenizer)
    model: SimpleGPT = SimpleGPT(tokenizer.vocab_size).to(device)

    # 2. Inject LoRA into all Linear layers
    name: str
    module: nn.Module
    for name, module in model.named_modules():  # type: ignore[assignment]
        if isinstance(module, nn.Linear):
            # Example: wrapping the head
            setattr(model, 'lm_head', LoRALinear(model.lm_head, rank=4))
            break

    # 3. Training Loop (Subset)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loader = DataLoader(dataset, batch_size=2)

    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()  # type: ignore[no-untyped-call]
        print(f"Loss: {loss.item():.4f}")
        break  # Just a sample step

    # 4. Generate
    print("\nGenerated Output:", generate(model, tokenizer, "This is", device, max_new_tokens=10))