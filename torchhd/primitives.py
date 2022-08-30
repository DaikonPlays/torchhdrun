from typing import Literal, Callable, overload
import torch
from torch import Tensor, FloatTensor

__all__ = [
    "mul",
    "bmul",
    "add",
    "badd",
    "randsel",
    "brandsel",
    "shift",
    "bind",
    "multibind",
    "release",
    "bundle",
    "multibundle",
    "scratch",
]


def mul(input: Tensor, other: Tensor) -> Tensor:
    r"""Binds two hypervectors which produces a hypervector dissimilar to both.

    Binding is used to associate information, for instance, to assign values to variables.

    .. math::

        \otimes: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.bind``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1., -1., -1.],
                [ 1.,  1.,  1.]])
        >>> functional.bind(x[0], x[1])
        tensor([ 1., -1., -1.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        return torch.logical_xor(input, other)

    return torch.mul(input, other)


def biggest_power_two(n):
    """Returns the biggest power of two <= n"""
    # if n is a power of two simply return it
    if not (n & (n - 1)):
        return n

    # else set only the most significant bit
    return int("1" + (len(bin(n)) - 3) * "0", 2)


def bmul(input: Tensor) -> Tensor:
    r"""Binding of multiple hypervectors.

    Binds all the input hypervectors together.

    .. math::

        \bigotimes_{i=0}^{n-1} V_i

    Args:
        input (Tensor): input hypervector tensor.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    .. note::

        This method is not supported for ``torch.float16`` and ``torch.bfloat16`` input data types on a CPU device.

    Examples::

        >>> x = functional.random_hv(3, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.multibind(x)
        tensor([ 1.,  1., -1.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
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

    return torch.prod(input, dim=-2, dtype=dtype)


def add(input: Tensor, other: Tensor, *, generator: torch.Generator = None) -> Tensor:
    r"""Bundles two hypervectors which produces a hypervector maximally similar to both.

    The bundling operation is used to aggregate information into a single hypervector.

    .. math::

        \oplus: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.bundle``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector
        tie (BoolTensor, optional): specifies how to break a tie while bundling boolean hypervectors. Default: only set bit if both ``input`` and ``other`` are ``True``.

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.bundle(x[0], x[1])
        tensor([0., 2., 0.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        tiebreaker = torch.empty_like(input)
        tiebreaker.bernoulli_(0.5, generator=generator)
        return input.where(input == other, tiebreaker)

    return torch.add(input, other)


def badd(input: Tensor) -> Tensor:
    r"""Multiset of input hypervectors.

    Bundles all the input hypervectors together.

    Aliased as ``torchhd.functional.multibundle``.

    .. math::

        \bigoplus_{i=0}^{n-1} V_i

    Args:
        input (Tensor): input hypervector tensor

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = functional.random_hv(3, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.multiset(x)
        tensor([-1.,  3.,  1.])

    """
    dim = -2
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        count = torch.sum(input, dim=dim, dtype=torch.long)
        threshold = input.size(dim) // 2
        return torch.greater(count, threshold)

    return torch.sum(input, dim=dim, dtype=dtype)


def sub(input: Tensor, other: Tensor, *, generator: torch.Generator = None) -> Tensor:
    r"""Bundles two hypervectors which produces a hypervector maximally similar to both.

    The bundling operation is used to aggregate information into a single hypervector.

    .. math::

        \oplus: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.bundle``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector
        tie (BoolTensor, optional): specifies how to break a tie while bundling boolean hypervectors. Default: only set bit if both ``input`` and ``other`` are ``True``.

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.bundle(x[0], x[1])
        tensor([0., 2., 0.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        tiebreaker = torch.empty_like(input)
        tiebreaker.bernoulli_(0.5, generator=generator)
        return input.where(input == other, tiebreaker)

    return torch.sub(input, other)


def randsel(
    input: Tensor, other: Tensor, *, p: float = 0.5, generator: torch.Generator = None
) -> Tensor:
    select = torch.empty_like(input, dtype=torch.bool)
    select.bernoulli_(p, generator=generator)
    return input.where(select, other)


def brandsel(
    input: Tensor, *, p: FloatTensor = None, generator: torch.Generator = None
) -> Tensor:
    d = input.size(-1)
    device = input.device

    if p is None:
        p = torch.ones(input.shape[:-1], dtype=torch.float, device=device)

    select = torch.multinomial(p, d, replacement=True, generator=generator)
    select.unsqueeze_(-2)
    return input.gather(-2, select).squeeze(-2)


def shift(input: Tensor, *, n=1) -> Tensor:
    r"""Permutes hypervector by specified number of shifts.

    The permutation operator is used to assign an order to hypervectors.

    .. math::

        \Pi: \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.permute``.

    Args:
        input (Tensor): input hypervector
        n (int or tuple of ints, optional): The number of places by which the elements of the tensor are shifted. If shifts is a tuple, dims must be a tuple of the same size, and each dimension will be rolled by the corresponding value.

    Shapes:
        - Input: :math:`(*, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = functional.random_hv(1, 3)
        >>> x
        tensor([ 1.,  -1.,  -1.])
        >>> functional.permute(x)
        tensor([ -1.,  1.,  -1.])

    """
    return torch.roll(input, shifts=n, dims=-1)


_bind = mul
_multibind = bmul
_bundle = add
_multibundle = badd
_permute = shift


def bind(input: Tensor, other: Tensor) -> Tensor:
    return _bind(input, other)


def multibind(input: Tensor) -> Tensor:
    return _multibind(input)


def release(input: Tensor, other: Tensor) -> Tensor:
    if torch.is_complex(other):
        other = other.conj()

    return bind(input, other)


def bundle(input: Tensor, other: Tensor) -> Tensor:
    return _bundle(input, other)


def multibundle(input: Tensor) -> Tensor:
    return _multibundle(input)


def scratch(input: Tensor, other: Tensor) -> Tensor:
    return sub(input, other)


def permute(input: Tensor, n: int) -> Tensor:
    return _permute(input, n)


@overload
def set_bind_method(name: Literal["multiply"]):
    ...


@overload
def set_bind_method(
    single: Callable[[Tensor, Tensor], Tensor], multi: Callable[[Tensor], Tensor] = None
):
    ...


def set_bind_method(name_or_single: str, multi=None):
    global _bind
    global _multibind

    # handle case when a build-in name is provided
    if isinstance(name_or_single, str):
        name = name_or_single
        supported = {"multiply"}
        if name not in supported:
            raise ValueError(
                f"bind method {name} is not supported, use one of: {supported}"
            )

        if name == "multiply":
            _bind = mul
            _multibind = bmul

    # handle case when a custom function is provided
    elif callable(name_or_single):
        single = name_or_single
        _bind = single

        if multi is not None:
            if not callable(multi):
                raise ValueError("The provided multibind method is not callable")

            _multibind = multi

        else:
            # When no efficient multibundle implementation is provide,
            # fallback to for-loop using single.
            _multibind = multi_from_single(single)

    else:
        raise ValueError("Must pass a method name or a custom function")


@overload
def set_bundle_method(name: Literal["add", "randsel"]):
    ...


@overload
def set_bundle_method(
    single: Callable[[Tensor, Tensor], Tensor], multi: Callable[[Tensor], Tensor] = None
):
    ...


def set_bundle_method(name_or_single: str, multi=None):
    global _bundle
    global _multibundle

    # handle case when a build-in name is provided
    if isinstance(name_or_single, str):
        name = name_or_single
        supported = {"add", "randsel"}
        if name not in supported:
            raise ValueError(
                f"bundle method {name} is not supported, use one of: {supported}"
            )

        if name == "add":
            _bundle = add
            _multibundle = badd

        if name == "randsel":
            _bundle = randsel
            _multibundle = brandsel

    # handle case when a custom function is provided
    elif callable(name_or_single):
        single = name_or_single
        _bundle = single

        if multi is not None:
            if not callable(multi):
                raise ValueError("The provided multibundle method is not callable")

            _multibundle = multi

        else:
            # When no efficient multibundle implementation is provide,
            # fallback to for-loop using single.
            _multibundle = multi_from_single(single)

    else:
        raise ValueError("Must pass a method name or a custom function")


@overload
def set_permute_method(name: Literal["shift"]):
    ...


@overload
def set_permute_method(func: Callable[[Tensor, Tensor], Tensor]):
    ...


def set_permute_method(name_or_func: str):
    global _permute

    # handle case when a build-in name is provided
    if isinstance(name_or_func, str):
        name = name_or_func
        supported = {"shift"}
        if name not in supported:
            raise ValueError(
                f"bundle method {name} is not supported, use one of: {supported}"
            )

        if name == "shift":
            _permute = shift

    # handle case when a custom function is provided
    elif callable(name_or_func):
        func = name_or_func
        _permute = func

    else:
        raise ValueError("Must pass a method name or a custom function")


def multi_from_single(
    single: Callable[[Tensor, Tensor], Tensor]
) -> Callable[[Tensor], Tensor]:
    def fallback_multi(input: Tensor) -> Tensor:
        n = input.size(-2)

        if n == 1:
            return input.clone()

        output = single(input[..., 0, :], input[..., 1, :])
        for i in range(2, n):
            output = single(output, input[..., i, :])

        return output

    return fallback_multi
