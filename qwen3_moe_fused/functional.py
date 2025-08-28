import torch

from .grouped_gemm.interface import grouped_gemm

def moe_fused_linear(input: torch.Tensor, weight: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    return grouped_gemm(input, weight, m_sizes)
