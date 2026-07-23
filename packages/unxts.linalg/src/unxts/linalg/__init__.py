"""Heterogeneous-unit matrices and vectors for unxt (canonical: unxts.linalg).

`QuantityMatrix` (alias `QM`) is a quantity container whose elements may each
carry a different unit, backed by a single JAX array plus a static
`UnitsMatrix`. It supports 1-D (vector) and 2-D (matrix) structures, with
Quax-registered arithmetic (add/sub, mul/div, matmul, transpose, diag,
reduce-sum) plus the ``matmul``/``matvec``/``vecmat``/``vecdot`` products and
``det``/``inv`` primitives that track per-element units through the linear
algebra.
"""

from ._version import version as __version__

__all__ = (
    "__version__",
    "QM",
    "QuantityMatrix",
    "UnitsMatrix",
    "cdict_units",
    "det",
    "det_p",
    "inv",
    "inv_p",
    "matmul",
    "matvec",
    "vecdot",
    "vecmat",
)

# Importing from ``._src`` triggers the Quax primitive registrations and the
# plum conversion/dispatch side effects (see ``._src.__init__``).
from ._src import (
    QM,
    QuantityMatrix,
    UnitsMatrix,
    cdict_units,
    det,
    det_p,
    inv,
    inv_p,
    matmul,
    matvec,
    vecdot,
    vecmat,
)
