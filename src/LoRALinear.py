import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    
    def __init__(self, original_layer: nn.Linear, rank: int =8, alpha: int = 160):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
    
        # 1. Reference the origynal GTP-2 layer.
        self.original_layer = original_layer

        # 2. Freeze the original weights - here LoRA happens!
        for param in self.original_layer.parameters():
            # sStopping gradients to flow into the massive GPT-2 backbone.
            param.requires_grad = False

        # 3. Deinining a Low-Rank matrices A and B.
        # In_features and out_features come from your existing GPT-2 block.
        out_features:int = self.original_layer.weight.shape[0]
        in_features: int = self.original_layer.weight.shape[1]

        # Matrix A: Initialized with Kaiming uniform (standard for neural nets)
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))

        # Matrix B: Initialized to zero so the "delta" starts at zero
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A should be random, B should be zero so the initial model behavior is unchanged
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard GPT-2 path
        result = self.original_layer(x)
        
        # LoRA path: (x @ A.T @ B.T) * scaling
        # We use .T (transpose) because of how linear algebra works in PyTorch
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        
        return result + lora_out




     