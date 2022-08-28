from typing import Literal, Callable, overload
import torch
from torch import Tensor, BoolTensor

__all__ = ["mul", "prod", "add", "sum", "bind", "multibind", "bundle", "multibundle"]


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


def prod(input: Tensor) -> Tensor:
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
    dim = -2

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        hvs = torch.unbind(input, dim)
        result = hvs[0]

        for i in range(1, len(hvs)):
            result = torch.logical_xor(result, hvs[i])

        return result

    return torch.prod(input, dim=dim, dtype=dtype)


def add(input: Tensor, other: Tensor, *, tie: BoolTensor = None) -> Tensor:
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
        if tie is not None:
            return torch.where(input == other, input, tie)
        else:
            return torch.logical_and(input, other)

    return torch.add(input, other)


def sum(input: Tensor) -> Tensor:
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


def shift(input: Tensor, *, shifts=1, dims=-1) -> Tensor:
    r"""Permutes hypervector by specified number of shifts.

    The permutation operator is used to assign an order to hypervectors.

    .. math::

        \Pi: \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.permute``.

    Args:
        input (Tensor): input hypervector
        shifts (int or tuple of ints, optional): The number of places by which the elements of the tensor are shifted. If shifts is a tuple, dims must be a tuple of the same size, and each dimension will be rolled by the corresponding value.
        dims (int or tuple of ints, optional): axis along which to permute the hypervector. Default: ``-1``.

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(1, 3)
        >>> x
        tensor([ 1.,  -1.,  -1.])
        >>> functional.permute(x)
        tensor([ -1.,  1.,  -1.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    return torch.roll(input, shifts=shifts, dims=dims)


_bind = mul
_multibind = prod
_bundle = add
_multibundle = sum
_permute = shift


def bind(input: Tensor, other: Tensor) -> Tensor:
    return _bind(input, other)


def multibind(input: Tensor) -> Tensor:
    return _multibind(input)


def bundle(input: Tensor, other: Tensor) -> Tensor:
    return _bundle(input, other)


def multibundle(input: Tensor) -> Tensor:
    return _multibundle(input)


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
            _multibind = prod

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
def set_bundle_method(name: Literal["add"]):
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
        supported = {"add"}
        if name not in supported:
            raise ValueError(
                f"bundle method {name} is not supported, use one of: {supported}"
            )

        if name == "add":
            _bundle = add
            _multibundle = sum

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
