"""Orthogonal Mixin classes for Quantity classes."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from astropy.units import CompositeUnit
from jaxtyping import Array, ArrayLike

from dataclassish import replace

from .api import uconvert, ustrip
from unxt._src.units.core import AbstractUnits


class AstropyQuantityCompatMixin:
    """Mixin for compatibility with `astropy.units.Quantity`."""

    value: ArrayLike
    unit: AbstractUnits
    to_units: Callable[[Any], "AbstractQuantity"]
    to_units_value: Callable[[Any], ArrayLike]

    def to(self, u: Any, /) -> "AbstractQuantity":
        """Convert the quantity to the given units.

        See `AbstractQuantity.to_units`.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.to("cm")
        Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

        """
        return uconvert(u, self)  # redirect to the standard method

    def to_value(self, u: Any, /) -> ArrayLike:
        """Return the value in the given units.

        See `AbstractQuantity.to_units_value`.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.to_value("cm")
        Array(100., dtype=float32, weak_type=True)

        """
        return ustrip(u, self)  # redirect to the standard method

    # TODO: support conversion of elements to Unit
    def decompose(self, bases: Sequence[AbstractUnits], /) -> "AbstractQuantity":
        """Decompose the quantity into the given bases.

        Examples
        --------
        >>> import astropy.units as u
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.decompose([u.cm, u.s])
        Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

        """
        du = self.unit.decompose(bases)  # decomposed units
        base_units = CompositeUnit(scale=1, bases=du.bases, powers=du.powers)
        return replace(self, value=self.value * du.scale, unit=base_units)


class IPythonReprMixin:
    """Mixin class for IPython representation of a Quantity."""

    value: Array
    unit: AbstractUnits

    def _repr_latex_(self) -> str:
        r"""Return a LaTeX representation of the Quantity.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity([1., 2, 3, 4], "m")
        >>> q._repr_latex_()
        '$[1.\\, 2.\\, 3.\\, 4.] \\; \\mathrm{m}$'

        """
        unit_repr = getattr(self.unit, "_repr_latex_", self.unit.__repr__)()
        value_repr = np.array2string(self.value, separator=",~")

        return f"${value_repr} \\; {unit_repr[1:-1]}$"

    # TODO: implement:
    # - _repr_mimebundle_
    # - _repr_html_
    # - _repr_markdown_
    # - _repr_json_
