"""Utilities.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, cast

from jax import dtypes
from jaxtyping import ArrayLike
from quax import quaxify

import quaxed.lax as qlax

if TYPE_CHECKING:
    from typing import Self


_singleton_insts: dict[type, object] = {}


class SingletonMixin:
    """Singleton class.

    This class is a mixin that can be used to create singletons.

    Examples
    --------
    >>> class MySingleton(SingletonMixin):
    ...     pass

    >>> a = MySingleton()
    >>> b = MySingleton()
    >>> a is b
    True

    """

    def __new__(cls, /, *_: Any, **__: Any) -> "Self":
        # Check if instance already exists
        if cls in _singleton_insts:
            return cast("Self", _singleton_insts[cls])
        # Create new instance and cache it
        self = object.__new__(cls)
        _singleton_insts[cls] = self
        return self


@quaxify  # type: ignore[misc]
def promote_dtypes(*arrays: ArrayLike) -> tuple[ArrayLike, ...]:
    """Promotes all input arrays to a common dtype.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> x1 = jnp.array([1, 2, 3], dtype=jnp.int32)
    >>> x2 = jnp.array([4, 5, 6], dtype=jnp.float32)

    >>> x1, x2 = promote_dtypes(x1, x2)
    >>> x1.dtype, x2.dtype
    (dtype('float32'), dtype('float32'))

    >>> q1 = u.Quantity.from_([1, 2, 3], "m", dtype=int)
    >>> q2 = u.Quantity([4.0, 5, 6], unit="km")
    >>> q1, q2 = promote_dtypes(q1, q2)
    >>> q1.dtype, q2.dtype
    (dtype('float32'), dtype('float32'))

    """
    common_dtype = dtypes.result_type(*arrays)
    return tuple(qlax.convert_element_type(arr, common_dtype) for arr in arrays)
