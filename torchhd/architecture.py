import torch
from torch import Tensor

class ArchModel:
    def bundle(self, input: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError

    def multibundle(self, input: Tensor) -> Tensor:
        n = input.size(-2)

        if n == 1:
            return input.clone()

        hvs = torch.unbind(input, dim=-2)

        output = self.bundle(hvs[0], hvs[1])
        for i in range(2, n):
            output = self.bundle(output, hvs[i])

        return output 

    def difference(self, input: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError

    def diff(self, input: Tensor, other: Tensor) -> Tensor:
        return self.difference(input, other)

    def bind(self, input: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError

    def multibind(self, input: Tensor) -> Tensor:
        n = input.size(-2)

        if n == 1:
            return input.clone()

        hvs = torch.unbind(input, dim=-2)

        output = self.bind(hvs[0], hvs[1])
        for i in range(2, n):
            output = self.bind(output, hvs[i])

        return output

    def unbind(self, input: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError

    def permute(self, input: Tensor, n: int = 1) -> Tensor:
        raise NotImplementedError


class MAP(ArchModel):
    @staticmethod
    def bundle(input: Tensor, other: Tensor) -> Tensor:
        return input.add(other)

    @staticmethod
    def multibundle(input: Tensor) -> Tensor:
        return input.sum(dim=-2)

    @staticmethod
    def difference(input: Tensor, other: Tensor) -> Tensor:
        return input.sub(other)

    @staticmethod
    def bind(input: Tensor, other: Tensor) -> Tensor:
        return input.mul(other)

    @staticmethod
    def multibind(input: Tensor) -> Tensor:
        return input.prod(dim=-2)

    @staticmethod
    def unbind(input: Tensor, other: Tensor) -> Tensor:
        return input.mul(other)

    @staticmethod
    def permute(input: Tensor, n: int = 1) -> Tensor:
        return input.roll(shifts=n, dim=-1)


def biggest_power_two(n):
    """Returns the biggest power of two <= n"""
    # if n is a power of two simply return it
    if not (n & (n - 1)):
        return n

    # else set only the most significant bit
    return int("1" + (len(bin(n)) - 3) * "0", 2)


class BSC(ArchModel):
    @staticmethod
    def bundle(input: Tensor, other: Tensor, *, generator: torch.Generator = None) -> Tensor:
        tiebreaker = torch.empty_like(input)
        tiebreaker.bernoulli_(0.5, generator=generator)
        return input.where(input == other, tiebreaker)

    @staticmethod
    def multibundle(input: Tensor, *, generator:torch.Generator = None) -> Tensor:
        n = input.size(-2)

        count = torch.sum(input, dim=-2, dtype=torch.long)
        
        # add a tiebreaker when there are an even number of hvs
        if n % 2 == 0:
            tiebreaker = torch.empty_like(count)
            tiebreaker.bernoulli_(0.5, generator=generator)
            count += tiebreaker
            n += 1
        
        threshold = n // 2
        return torch.greater(count, threshold)

    @staticmethod
    def difference(input: Tensor, other: Tensor) -> Tensor:
        return input.logical_and(~other)

    @staticmethod
    def bind(input: Tensor, other: Tensor) -> Tensor:
        return input.logical_xor(other)

    @staticmethod
    def multibind(input: Tensor) -> Tensor:
        n = input.size(-2)
        n_ = biggest_power_two(n)
        output = input[..., :n_, :]

        # parallelize many XORs in a hierarchical manner
        # for larger batches this is significantly faster
        while output.size(-2) > 1:
            output = torch.logical_xor(output[..., 0::2, :], output[..., 1::2, :])

        output = output.squeeze(-2)

        # TODO: as an optimization we could also perform the hierarchical XOR
        # on the leftovers in a recursive fashion
        leftovers = torch.unbind(input[..., n_:, :], -2)
        for i in range(n - n_):
            output = torch.logical_xor(output, leftovers[i])

        return output

    @staticmethod
    def unbind(input: Tensor, other: Tensor) -> Tensor:
        return input.logical_xor(other)

    @staticmethod
    def permute(input: Tensor, n: int = 1) -> Tensor:
        return input.roll(shifts=n, dim=-1)