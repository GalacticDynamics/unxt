"""Compatibility layer for `astropy.units`."""

__all__: list[str] = []


from astropy.units import Quantity as AstropyQuantity
from plum import conversion_method

from .base import AbstractQuantity
from .core import Quantity
from .fast import UncheckedQuantity


@conversion_method(type_from=AbstractQuantity, type_to=AstropyQuantity)  # type: ignore[misc]
def convert_quantity_to_astropyquantity(obj: AbstractQuantity, /) -> AstropyQuantity:
    """`AbstractQuantity` -> `astropy.AbstractQuantity`."""
    return AstropyQuantity(obj.value, obj.unit)


@conversion_method(type_from=AstropyQuantity, type_to=UncheckedQuantity)  # type: ignore[misc]
def convert_astropyquantity_to_uncheckedquantity(
    obj: AstropyQuantity, /
) -> UncheckedQuantity:
    """`astropy.AbstractQuantity` -> `UncheckedQuantity`."""
    return UncheckedQuantity(obj.value, obj.unit)


@conversion_method(type_from=AstropyQuantity, type_to=Quantity)  # type: ignore[misc]
def convert_astropyquantity_to_quantity(obj: AstropyQuantity, /) -> Quantity:
    """`astropy.AbstractQuantity` -> `Quantity`."""
    return Quantity(obj.value, obj.unit)
