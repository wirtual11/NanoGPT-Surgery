import torch.nn as nn
import torch

from CharacterTokenizer import CharacterTokenizer
from GPTDataset import GPTDataset
from SimpleGPT import SimpleGPT
from LoRALinear import LoRALinear
from torch.utils.data import DataLoader
import torch.nn.functional as F
import GPTDataset
import SimpleGPT

if __name__ == "__main__":
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_text:str = "This is a test of the emergency broadcast system. This is only a test."

    # 1. Setup
    tokenizer: CharacterTokenizer = CharacterTokenizer(sample_text) 
    dataset: GPTDataset.GPTDataset = GPTDataset.GPTDataset(sample_text, seq_length=16, tokenizer=tokenizer)
    model: SimpleGPT.SimpleGPT = SimpleGPT.SimpleGPT(tokenizer.vocab_size).to(device)

    # 2. Inject LoRA into all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Example: wrapping the head
            model.lm_head = LoRALinear(model.lm_head, rank=4)
            break 

    # 3. Training Loop (Subset)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loader = DataLoader(dataset, batch_size=2)
    
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Loss: {loss.item():.4f}")
        break # Just a sample step

    # 4. Generate
    print("\nGenerated Output:", generate(model, tokenizer, "This is", max_new_tokens=10))

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temp=1.0, top_k=5):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -256:] # Crop to block_size
        logits = model(idx_cond)[:, -1, :] / temp
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        
    return tokenizer.decode(idx[0].tolist())