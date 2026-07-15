"""Registry mapping NumPy ufuncs to unit-aware handlers.

NumPy ufuncs (``np.add``, ``np.multiply``, ``np.sqrt``, ...) are dispatched to
quantities through :meth:`AbstractQuantity.__array_ufunc__`. Because every numpy
ufunc shares the single type ``numpy.ufunc``, the per-ufunc unit rule cannot be
selected by type-based (plum) dispatch; it must be keyed on the ufunc *object*.
This module holds that registry.

For the built-in numpy ufuncs the default behaviour is to delegate to the
identically named function in `quaxed.numpy`, which is a ``quaxify``-wrapped
`jax.numpy` and therefore routes through unxt's ``quax.register`` primitive
handlers (see ``register_primitives``) that propagate units correctly.

Custom (user-defined) ufuncs carry no unit semantics, so a handler must be
registered explicitly with :func:`register_ufunc`; an unregistered custom ufunc
raises a loud ``TypeError`` rather than silently dropping units.
"""

__all__ = ("register_ufunc",)

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

import quaxed.numpy as qnp

AnyCallable: TypeAlias = Callable[..., Any]

# Registry keyed by the ufunc object.
_UFUNC_REGISTRY: dict[np.ufunc, Callable[..., Any]] = {}

# ``ufunc.reduce`` / ``ufunc.accumulate`` map to these `quaxed.numpy` functions.
_REDUCE_MAP: dict[str, str] = {
    "add": "sum",
    "multiply": "prod",
    "maximum": "max",
    "minimum": "min",
    "logical_and": "all",
    "logical_or": "any",
}
_ACCUMULATE_MAP: dict[str, str] = {"add": "cumsum", "multiply": "cumprod"}


def register_ufunc(ufunc: np.ufunc, /) -> Callable[[AnyCallable], AnyCallable]:
    """Register a unit-aware handler for a NumPy ufunc.

    The decorated function is called as ``handler(ufunc, method, *inputs,
    **kwargs)`` and must return a unit-carrying result. It may itself be
    `plum`-dispatched on the input types.

    Examples
    --------
    >>> import numpy as np
    >>> import unxt as u

    >>> doubler = np.frompyfunc(lambda x: 2 * x, 1, 1)

    >>> @u.quantity.register_ufunc(doubler)
    ... def _(ufunc, method, x, /, **kw):
    ...     return u.Q(2 * x.value, x.unit)

    >>> doubler(u.Q(3.0, "m"))
    Quantity(Array(6., dtype=float32), unit='m')

    """

    def decorator(func: AnyCallable) -> AnyCallable:
        _UFUNC_REGISTRY[ufunc] = func
        return func

    return decorator


def apply_ufunc(
    ufunc: np.ufunc,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Apply a NumPy ufunc to quantities, propagating units.

    Resolution order:

    1. an explicitly registered handler (custom ufuncs, or built-in overrides),
    2. delegation to a `quaxed.numpy` function for the ``__call__``, ``reduce``,
       and ``accumulate`` methods of built-in ufuncs,
    3. otherwise ``NotImplemented`` so NumPy raises a loud ``TypeError`` -- units
       are never silently dropped.
    """
    handler = _UFUNC_REGISTRY.get(ufunc)
    if handler is not None:
        return handler(ufunc, method, *inputs, **kwargs)

    # Only the genuine built-in numpy ufuncs are delegated to quaxed.numpy.
    # Key on identity, not __name__: a custom ufunc (e.g. from numba) whose name
    # collides with a built-in must NOT be silently routed -- it requires an
    # explicit handler via ``register_ufunc`` or it errors loudly below.
    if getattr(np, getattr(ufunc, "__name__", ""), None) is not ufunc:
        return NotImplemented

    if method == "__call__":
        fn = getattr(qnp, ufunc.__name__, None)
        if fn is not None:
            return fn(*inputs, **kwargs)

    elif method in ("reduce", "accumulate"):
        mapping = _REDUCE_MAP if method == "reduce" else _ACCUMULATE_MAP
        fn = getattr(qnp, mapping.get(ufunc.__name__, ""), None)
        if fn is not None:
            # NumPy's reduce/accumulate default to axis 0, unlike jax.numpy's
            # sum/cumsum (all-axes / flattened); preserve numpy semantics.
            kwargs = {"axis": 0, **kwargs}
            return fn(*inputs, **kwargs)

    return NotImplemented


# ---------------------------------------------------------------------------
# Built-in overrides
#
# ``deg2rad``/``rad2deg``/``radians``/``degrees`` are angle *conversions*, but
# JAX lowers them to ``x * (pi/180)`` (a bare ``mul``). The default delegation
# to ``quaxed.numpy`` therefore scales the value while leaving the unit label
# unchanged -- a silently wrong result (``deg2rad(180 deg) -> 3.14159 deg``).
# Register explicit handlers that convert an angle quantity to the target unit
# (and raise on a non-angle quantity, matching astropy).

_ANGLE_CONVERSIONS: dict[np.ufunc, str] = {
    np.deg2rad: "rad",
    np.radians: "rad",
    np.rad2deg: "deg",
    np.degrees: "deg",
}


def _make_angle_handler(to_unit: str) -> AnyCallable:
    def handler(_ufunc: np.ufunc, method: str, x: Any, /, **_kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented
        return x.uconvert(to_unit)

    return handler


for _ufunc, _to_unit in _ANGLE_CONVERSIONS.items():
    _UFUNC_REGISTRY[_ufunc] = _make_angle_handler(_to_unit)
