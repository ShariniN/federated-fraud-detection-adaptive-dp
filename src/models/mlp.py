"""
Model architecture for fraud classification.

Designed for DP-SGD compatibility:
- Batch Normalization is excluded as it is incompatible with Opacus
  (Opacus requires per-sample gradient computation, which is violated by
  BatchNorm sharing activations across samples).
- LayerNorm/GroupNorm or Dropout are suitable alternatives.
- The forward pass returns raw logits. Conformal prediction and probability
  estimation methods apply Softmax on top of these logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudMLP(nn.Module):
    """
    Compact MLP for fraud detection on the 72 PCA components.

    Architecture: 72 -> 32 -> 16 -> 2 (two hidden layers, 2,898 trainable
    parameters). Deliberately small: under DP-SGD the injected noise norm grows
    with sqrt(parameter count), so at a fixed privacy budget a compact model
    keeps more of its learning signal. An earlier 72-256-128-64-2 variant
    (59,970 parameters) was no better than logistic regression under the same
    budget and was replaced by this one on 4 Sep 2026.

    - GELU activation to avoid dead units under gradient clipping.
    - Kaiming initialisation to keep early-epoch gradient magnitudes stable.
    - Low dropout (0.1) because DP-SGD noise already regularises strongly.
    """
    def __init__(self, input_dim=72):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 2)  # Raw logits: [benign, fraud]
        )
        
        # Apply He/Kaiming Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Kaiming normal init for ReLU/GELU networks
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass returning raw logits.
        """
        return self.network(x)

    def predict_proba(self, x):
        """
        Return Softmax probabilities [prob_benign, prob_fraud].
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
