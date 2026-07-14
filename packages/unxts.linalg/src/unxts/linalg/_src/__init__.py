"""Heterogeneous unit containers for vectors and matrices.

This package provides two closely related building blocks:

- `UnitsMatrix`, an immutable nested tuple of units with indexing support
- `QuantityMatrix`, a quantity-like wrapper around one array plus a matching
    static `UnitsMatrix`

The numeric payload is a single JAX array of shape ``(..., *shape)`` where the
trailing dimensions are the logical vector or matrix dimensions and any leading
dimensions are batch dimensions. Units are stored separately as a static nested
tuple structure with the same logical shape, allowing every element to carry
its own physical unit.

Currently the public surface supports only 1-D and 2-D structures:

- 1-D: ``(..., N)`` with units ``(u0, u1, ..., uN-1)``
- 2-D: ``(..., N, M)`` with units ``((u00, u01, ...), (u10, u11, ...), ...)``

Quax primitive dispatches (``add_p``, ``dot_general_p``) perform the
necessary per-element unit conversions via `unxt.uconvert_value` — which
correctly handles affine conversions (e.g. °F → °C), not just
multiplicative scale factors.
"""

__all__ = (
    "QM",
    "QuantityMatrix",
    "UnitsMatrix",
    "cdict_units",
    "det",
    "det_p",
    "inv",
    "inv_p",
)

from . import _register_primitives  # noqa: F401
from ._det import det, det_p
from ._inv import inv, inv_p
from ._quantity_matrix import (  # noqa: F401
    QM,
    QuantityMatrix,
    _convert_value_matrix,
    _convert_value_vector,
)
from ._units_matrix import UnitsMatrix
from ._utils import cdict_units
