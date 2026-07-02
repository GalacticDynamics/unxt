"""Interoperability with `matplotlib`."""
# pylint: disable=import-error

# This module was adapted from both the `astropy` and `pint` package's
# implementations.

__all__ = ("UnxtConverter", "setup_matplotlib_support_for_unxt")


from collections.abc import Iterable, Sized
from dataclasses import dataclass, field
from typing import Any

import matplotlib.units
from jaxtyping import Array
from matplotlib.axes import Axes

from zeroth import zeroth

from unxt.quantity import AbstractQuantity, BareQuantity, ustrip


@dataclass
class UnxtConverter(matplotlib.units.ConversionInterface):  # type: ignore[misc]
    """Support `unxt` in `matplotlib`'s unit conversion framework.

    This class is a subclass of `matplotlib.units.ConversionInterface`
    and is used to convert `unxt.Quantity` instances for use with
    `matplotlib`.

    """

    axisinfo_kw: dict[str, Any] = field(
        default_factory=lambda: {"format": "latex_inline"}
    )
    """Keyword arguments to use when making the :meth:`matplotlib.units.AxisInfo`."""

    def convert(self, obj: Any, unit: Any, axis: Axes) -> Array | list[Array]:
        """Convert *obj* using *unit* for the specified *axis*."""
        # Hot-path Quantity
        if isinstance(obj, AbstractQuantity):
            return ustrip(unit, obj)
        # Need to recurse (singly) into iterables
        if isinstance(obj, Iterable):
            return [self._convert_value(v, unit, axis) for v in obj]

        return self._convert_value(obj, unit, axis)

    @staticmethod
    def _convert_value(obj: Any, unit: Any, axis: Axes) -> Array:
        """Handle converting using attached unit or falling back to axis units."""
        if isinstance(obj, AbstractQuantity):
            return ustrip(unit, obj)

        return BareQuantity.from_(obj, axis.get_units()).ustrip(unit)

    def axisinfo(self, unit: Any, _: Axes) -> matplotlib.units.AxisInfo:
        """Return axis information for this particular unit."""
        fmt = self.axisinfo_kw.get("format", "latex_inline")
        return matplotlib.units.AxisInfo(label=unit.to_string(fmt))

    @staticmethod
    def default_units(x: Any, _: Axes) -> Any:
        """Get the default unit to use for the given combination of unit and axis."""
        if hasattr(x, "unit"):
            return x.unit
        if isinstance(x, Iterable) and isinstance(x, Sized):
            x = zeroth(x)
        return getattr(x, "units", None)


def setup_matplotlib_support_for_unxt(*, enable: bool = True) -> None:
    """Set up matplotlib's unit support for `unxt`.

    :param enable: Whether support should be enabled or disabled.

    """
    if enable:
        matplotlib.units.registry[AbstractQuantity] = UnxtConverter()
    else:
        matplotlib.units.registry.pop(AbstractQuantity, None)
