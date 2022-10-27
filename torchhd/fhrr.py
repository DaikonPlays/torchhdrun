import math
import torch
from torch import Tensor
import torch.nn.functional as F

from torchhd.base import VSA_Model

type_conversion = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


class FHRR(VSA_Model):
    @classmethod
    def random_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        generator=None,
        dtype=torch.complex64,
        device=None,
        requires_grad=False,
    ) -> "FHRR":

        dtype = type_conversion[dtype]

        size = (num_vectors, dimensions)
        angle = torch.empty(size, dtype=dtype, device=device)
        angle.uniform_(-math.pi, math.pi, generator=generator)
        magnitude = torch.ones_like(angle)

        result = torch.polar(magnitude, angle)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: 'FHRR') -> 'FHRR':
        return self.add(other)

    def multibundle(self) -> 'FHRR':
        return self.sum(dim=-2)

    def bind(self, other: 'FHRR') -> 'FHRR':
        return self.mul(other)

    def multibind(self) -> 'FHRR':
        return self.prod(dim=-2)

    def inverse(self) -> 'FHRR':
        return self.conj()

    def permute(self, n: int = 1) -> 'FHRR':
        return self.roll(shifts=n, dim=-1)

    def dot_similarity(self, others: "FHRR") -> Tensor:
        return F.linear(self, others.conj()).real

    def cos_similarity(self, others: "FHRR", *, eps=1e-08) -> Tensor:      
        self_dot = torch.real(self * self.conj()).sum(dim=-1)
        self_mag = self_dot.sqrt()

        others_dot = torch.real(others * others.conj()).sum(dim=-1)
        others_mag = others_dot.sqrt()

        if self.dim() > 1:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
        else:
            magnitude = self_mag * others_mag

        magnitude = magnitude.clamp(min=eps)
        return self.dot_similarity(others) / magnitude