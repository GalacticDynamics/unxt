"""Astropy interop for ``ParametricQuantity`` (registered on import).

``astropy`` is a hard dependency of ``unxts.parametric`` (its core
``PhysicalType`` machinery requires it), so this conversion is always
registered.
"""

__all__: tuple[str, ...] = ()

from astropy.units import Quantity as AstropyQuantity
from plum import conversion_method

import unxts.api as uapi

from .parametric import ParametricQuantity


@conversion_method(type_from=AstropyQuantity, type_to=ParametricQuantity)  # type: ignore[arg-type]
def convert_astropy_quantity_to_parametric_quantity(
    q: AstropyQuantity, /
) -> ParametricQuantity:
    """Convert an `astropy.units.Quantity` to a `ParametricQuantity`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxts.parametric import ParametricQuantity

    >>> convert(AstropyQuantity(1.0, "cm"), ParametricQuantity)
    ParametricQuantity(Array(1., dtype=float32), unit='cm')

    """
    u = uapi.unit_of(q)
    return ParametricQuantity(uapi.ustrip(u, q), u)
