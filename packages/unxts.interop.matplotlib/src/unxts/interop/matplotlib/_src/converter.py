"""Interoperability with `matplotlib`."""
# pylint: disable=import-error

# This module was adapted from both the `astropy` and `pint` package's
# implementations.

__all__ = ("UnxtConverter", "setup_matplotlib_support_for_unxt")


import warnings
from collections.abc import Iterable, Sized
from dataclasses import dataclass, field
from typing import Any

import matplotlib.units
from jaxtyping import Array
from matplotlib.axes import Axes

from zeroth import zeroth

from unxt.quantity import AbstractQuantity, Quantity, ustrip


@dataclass
class UnxtConverter(matplotlib.units.ConversionInterface):  # type: ignore[misc]
    """Support `unxt` in `matplotlib`'s unit conversion framework.

    This class is a subclass of `matplotlib.units.ConversionInterface`
    and is used to convert `unxt.Quantity` instances for use with
    `matplotlib`.

    """

    unit_format: str = "latex_inline"
    """`astropy` unit format for the axis label (see ``Unit.to_string``)."""

    axisinfo_kw: dict[str, Any] | None = field(default=None, init=True, repr=False)
    """Deprecated: use ``unit_format`` instead.

    .. deprecated:: 2.0.0
        Use ``unit_format`` directly instead of ``axisinfo_kw={"format": ...}``.
        This parameter will be removed in a future release.
    """

    def __post_init__(self) -> None:
        """Handle deprecated ``axisinfo_kw`` parameter."""
        if self.axisinfo_kw is not None:
            warnings.warn(
                "The `axisinfo_kw` parameter is deprecated and will be removed "
                "in a future release. Use `unit_format` directly instead. "
                "For example, use `UnxtConverter(unit_format='latex')` instead of "
                "`UnxtConverter(axisinfo_kw={'format': 'latex'})`.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            # Extract format from the dict and set unit_format
            if "format" in self.axisinfo_kw:
                # Use object.__setattr__ since this is a frozen dataclass in spirit
                object.__setattr__(self, "unit_format", self.axisinfo_kw["format"])

    def convert(self, obj: Any, unit: Any, axis: Axes) -> Array | list[Array]:
        """Convert *obj* using *unit* for the specified *axis*."""
        # Hot-path Quantity
        if isinstance(obj, AbstractQuantity):
            return ustrip(unit, obj)
        # Need to recurse (singly) into iterables, but a 0-d array is nominally
        # `Iterable` yet raises when iterated, so treat it as a scalar value.
        if isinstance(obj, Iterable) and getattr(obj, "ndim", None) != 0:
            return [self._convert_value(v, unit, axis) for v in obj]

        return self._convert_value(obj, unit, axis)

    @staticmethod
    def _convert_value(obj: Any, unit: Any, axis: Axes) -> Array:
        """Handle converting using attached unit or falling back to axis units."""
        if isinstance(obj, AbstractQuantity):
            return ustrip(unit, obj)

        return Quantity.from_(obj, axis.get_units()).ustrip(unit)

    def axisinfo(self, unit: Any, _: Axes) -> matplotlib.units.AxisInfo:
        """Return axis information for this particular unit."""
        # matplotlib may query axisinfo before any unit has been set on the axis.
        if unit is None:
            return matplotlib.units.AxisInfo()
        return matplotlib.units.AxisInfo(label=unit.to_string(self.unit_format))

    @staticmethod
    def default_units(x: Any, _: Axes) -> Any:
        """Get the default unit to use for the given combination of unit and axis."""
        if hasattr(x, "unit"):
            return x.unit
        if isinstance(x, Iterable) and isinstance(x, Sized):
            x = zeroth(x)
        # `unxt` quantities expose the singular `.unit` (not the astropy/pint
        # `.units`), so an unwrapped element must be checked with `.unit`.
        return getattr(x, "unit", getattr(x, "units", None))


def setup_matplotlib_support_for_unxt(*, enable: bool = True) -> None:
    """Set up matplotlib's unit support for `unxt`.

    :param enable: Whether support should be enabled or disabled.

    """
    if enable:
        matplotlib.units.registry[AbstractQuantity] = UnxtConverter()
    else:
        matplotlib.units.registry.pop(AbstractQuantity, None)
