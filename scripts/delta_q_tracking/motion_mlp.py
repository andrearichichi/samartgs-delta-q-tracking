from __future__ import annotations

import torch


class MotionMLP(torch.nn.Module):
    """Small temporal MLP mapping normalized time t in [0, 1] to scalar q(t)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        time_encoding: str = "raw",
        fourier_frequencies: int = 4,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if time_encoding not in {"raw", "fourier"}:
            raise ValueError(f"Unsupported time_encoding={time_encoding!r}")
        if fourier_frequencies < 0:
            raise ValueError("fourier_frequencies must be non-negative")

        self.time_encoding = time_encoding
        self.fourier_frequencies = int(fourier_frequencies)
        input_dim = 1 if time_encoding == "raw" else 1 + 2 * self.fourier_frequencies
        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(hidden_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.time_encoding == "raw":
            return t
        features = [t]
        for idx in range(self.fourier_frequencies):
            frequency = float(2 ** idx)
            angle = (2.0 * torch.pi * frequency) * t
            features.extend([torch.sin(angle), torch.cos(angle)])
        return torch.cat(features, dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError(f"Expected t with shape [N, 1], got {tuple(t.shape)}")
        return self.net(self.encode_time(t))
