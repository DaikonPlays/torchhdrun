import torch
from torch import Tensor
import torch.nn.functional as F

from torchhd.base import VSA_Model


class MAP(VSA_Model):
    @classmethod
    def random_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        sparsity=0.5,
        generator=None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAP":
        if dtype is None:
            dtype = torch.get_default_dtype()

        size = (num_vectors, dimensions)
        select = torch.empty(size, dtype=torch.bool, device=device)
        select.bernoulli_(1.0 - sparsity, generator=generator)

        result = torch.where(select, -1, +1).to(dtype=dtype, device=device)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "MAP") -> "MAP":
        return self.add(other)

    def multibundle(self) -> "MAP":
        return self.sum(dim=-2)

    def bind(self, other: "MAP") -> "MAP":
        return self.mul(other)

    def multibind(self) -> "MAP":
        return self.prod(dim=-2)

    def inverse(self) -> "MAP":
        return self.clone()

    def permute(self, n: int = 1) -> "MAP":
        return self.roll(shifts=n, dim=-1)

    def dot_similarity(self, others: "MAP") -> Tensor:
        return F.linear(self, others)

    def cos_similarity(self, others: "MAP", *, eps=1e-08) -> Tensor:
        dtype = torch.get_default_dtype()
        
        self_dot = torch.sum(self * self, dim=-1, dtype=dtype)
        self_mag = self_dot.sqrt()

        others_dot = torch.sum(others * others, dim=-1, dtype=dtype)
        others_mag = others_dot.sqrt()

        if self.dim() > 1:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
        else:
            magnitude = self_mag * others_mag

        magnitude = magnitude.clamp(min=eps)
        return self.dot_similarity(others).to(dtype) / magnitude