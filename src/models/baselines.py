"""
Baseline model architectures for comparative evaluation under DP-SGD.

All models omit BatchNorm (incompatible with Opacus per-sample gradient
clipping) and use Dropout as the sole regularisation mechanism.

Models
------
LogisticRegressionModel : linear baseline (parameter floor)
TabularResNet           : deep residual baseline (parameter ceiling)

See baselines section of Chapter 5 for architectural rationale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionModel(nn.Module):
    """Single Linear layer — the linear baseline.

    Purpose: Establishes a performance floor. If FraudMLP does not
    significantly exceed this, DP noise has swamped gradient learning.
    """

    def __init__(self, input_dim: int = 72):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


class ResidualBlock(nn.Module):
    """One residual block for tabular data.

    Structure: Linear -> ReLU -> Dropout -> Linear -> Dropout + skip -> ReLU.
    No BatchNorm: violates per-sample independence required by Opacus.
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class TabularResNet(nn.Module):
    """Deep residual network adapted for 1-D tabular fraud data.

    Hypothesis: The ~200K parameter count incurs a higher DP-SGD gradient
    noise overhead than FraudMLP (~50K params), negating any representational
    benefit from residual connections at the same epsilon budget.
    (Gorishniy et al., 2021; Anil et al., 2021)

    Architecture
    ------------
    Input proj [input_dim -> hidden_dim]
    -> N x ResidualBlock [hidden_dim -> hidden_dim]
    -> Head [hidden_dim -> 2]
    """

    def __init__(
        self,
        input_dim: int = 72,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.res_blocks(self.input_proj(x)))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_type: str, input_dim: int) -> nn.Module:
    """Factory: return a model instance by name.

    Parameters
    ----------
    model_type : str
        One of: "fraud_mlp" | "logistic_regression" | "tabular_resnet"
    input_dim : int
        Number of input features.
    """
    t = model_type.lower().strip()
    if t == "fraud_mlp":
        from .mlp import FraudMLP
        return FraudMLP(input_dim=input_dim)
    elif t in ("logistic_regression", "lr"):
        return LogisticRegressionModel(input_dim=input_dim)
    elif t in ("tabular_resnet", "resnet"):
        return TabularResNet(input_dim=input_dim)
    else:
        raise ValueError(
            f"Unknown model_type={t!r}. "
            "Choose from: 'fraud_mlp', 'logistic_regression', 'tabular_resnet'."
        )
