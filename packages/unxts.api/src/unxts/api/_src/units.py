"""Units API for unxt.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("unit", "unit_of")

from typing import TYPE_CHECKING, Any

import plum

if TYPE_CHECKING:
    # `unit` is liberal in what it accepts (Postel's law): a unit object, a
    # string, ... -> a unit. Declared here as a plain callable so static
    # checkers can read the `Any` parameter -- in particular so it works as an
    # `equinox.field(converter=unit)`, making `Quantity(1, "m")` type-check.
    # (`plum.dispatch.abstract` erases the signature to an opaque `Function`
    # that the converter field-specifier cannot introspect.) The `raise` body
    # keeps pylint from inferring a `None` return for callers (this branch is
    # type-checking only; the runtime definition is dispatched by plum).
    def unit(obj: Any, /) -> Any:
        raise NotImplementedError

else:

    @plum.dispatch.abstract
    def unit(obj: Any, /) -> Any:
        """Construct the units from a units object."""


@plum.dispatch.abstract
def unit_of(obj: Any, /) -> Any:
    """Return the units of an object."""
