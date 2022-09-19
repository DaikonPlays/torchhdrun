from typing import List
import torch
from torch import Tensor


def biggest_power_two(n):
    """Returns the biggest power of two <= n"""
    # if n is a power of two simply return it
    if not (n & (n - 1)):
        return n

    # else set only the most significant bit
    return int("1" + (len(bin(n)) - 3) * "0", 2)



class HyperTensor(Tensor):
    def bundle(self, other: 'HyperTensor') -> 'HyperTensor':
        raise NotImplementedError

    def multibundle(self) -> 'HyperTensor':
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(f"{class_name} needs to have at least two dimensions for multibundle, got size: {tuple(self.shape)}")

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[HyperTensor] = torch.unbind(self, dim=-2)
        print(type(tensors[0]))

        output = tensors[0].bundle(tensors[1])
        for i in range(2, n):
            output = output.bundle(tensors[i])

        return output 

    def difference(self, other: 'HyperTensor') -> 'HyperTensor':
        raise NotImplementedError

    def diff(self, other: 'HyperTensor') -> 'HyperTensor':
        return self.difference(other)

    def bind(self, other: 'HyperTensor') -> 'HyperTensor':
        raise NotImplementedError

    def multibind(self) -> 'HyperTensor':
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}")

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[HyperTensor] = torch.unbind(self, dim=-2)

        output = tensors[0].bind(tensors[1])
        for i in range(2, n):
            output = output.bind(tensors[i])

        return output 

    def unbind(self, other: 'HyperTensor') -> 'HyperTensor':
        raise NotImplementedError

    def permute(self, n: int = 1) -> 'HyperTensor':
        raise NotImplementedError
    

class MAPTensor(HyperTensor):
    def bundle(self, other: 'MAPTensor') -> 'MAPTensor':
        return self + other

    def multibundle(self) -> 'MAPTensor':
        return self.sum(dim=-2)

    def difference(self, other: 'MAPTensor') -> 'MAPTensor':
        return self - other

    def bind(self, other: 'MAPTensor') -> 'MAPTensor':
        return self * other

    def multibind(self) -> 'MAPTensor':
        return self.prod(dim=-2)

    def unbind(self, other: 'MAPTensor') -> 'MAPTensor':
        return self * other

    def permute(self, n: int = 1) -> 'MAPTensor':
        return self.roll(shifts=n, dim=-1)


class BSCTensor(HyperTensor):
    def bundle(self, other: Tensor, *, generator: torch.Generator = None) -> Tensor:
        tiebreaker = torch.empty_like(input)
        tiebreaker.bernoulli_(0.5, generator=generator)

        is_majority = self == other
        return self.where(is_majority, tiebreaker)

    def multibundle(self, *, generator:torch.Generator = None) -> Tensor:
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}")

        n = self.size(-2)

        count = self.sum(dim=-2, dtype=torch.long)
        
        # add a tiebreaker when there are an even number of hvs
        if n % 2 == 0:
            tiebreaker = torch.empty_like(count)
            tiebreaker.bernoulli_(0.5, generator=generator)
            count += tiebreaker
            n += 1
        
        threshold = n // 2
        return count > threshold

    def difference(self, other: Tensor) -> Tensor:
        return self.logical_and(~other)

    def bind(self, other: Tensor) -> Tensor:
        return self.logical_xor(other)

    def multibind(self) -> Tensor:
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}")

        n = self.size(-2)
        n_ = biggest_power_two(n)
        output = self[..., :n_, :]

        # parallelize many XORs in a hierarchical manner
        # for larger batches this is significantly faster
        while output.size(-2) > 1:
            output = torch.logical_xor(output[..., 0::2, :], output[..., 1::2, :])

        output = output.squeeze(-2)

        # TODO: as an optimization we could also perform the hierarchical XOR
        # on the leftovers in a recursive fashion
        leftovers = torch.unbind(self[..., n_:, :], -2)
        for i in range(n - n_):
            output = torch.logical_xor(output, leftovers[i])

        return output

    def unbind(self, other: Tensor) -> Tensor:
        return self.logical_xor(other)

    def permute(self, n: int = 1) -> Tensor:
        return self.roll(shifts=n, dim=-1)


if __name__ == "__main__":
    x = HyperTensor(torch.randn(2, 6))
    print(x)
    print(torch.add(x, 1))
    x.multibundle()