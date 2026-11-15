"""The `unitful` NumPyro handler."""

__all__ = ("unitful",)

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
from numpyro.primitives import Messenger

import unxt as u
from .rules import (
    UnitfulError,
    _has_quantity,
    _is_q,
    carrying_params,
    site_dimension,
)


def _is_concrete_zero(x: object) -> bool:
    if isinstance(x, jax.core.Tracer):
        return False
    return bool(np.all(np.asarray(x) == 0))


class unitful(Messenger):  # noqa: N801  (NumPyro handler naming)
    """Run a model whose distributions take Quantity parameters.

    At each sample site the parameters and any observed value are stripped in
    `usys`, so NumPyro computes on bare numbers and log p is per `usys` unit.
    The sampled value goes back to the model as a Quantity in the `usys` unit of
    the site's dimension (bare when ``rewrap=False``). ``units={"site": unit}``
    sets the dimension of a site whose family has no rule; the site is still
    expressed in `usys`.

    Examples
    --------
    >>> import numpyro, numpyro.distributions as dist
    >>> from numpyro import handlers
    >>> import unxt as u
    >>> from unxts.interop.numpyro import unitful

    >>> def model():
    ...     return numpyro.sample("m", dist.Normal(u.Q(0.0, "pc"), u.Q(3.0, "pc")))

    >>> m = handlers.seed(unitful(model, u.unitsystems.galactic), 0)()
    >>> m.unit
    Unit("kpc")

    """

    def __init__(
        self,
        fn: Callable[..., Any] | None,
        usys: u.AbstractUnitSystem,
        *,
        rewrap: bool = True,
        units: Mapping[str, Any] | None = None,
    ) -> None:
        if fn is None:
            msg = "unitful needs a model: wrap it as unitful(model, usys)."
            raise TypeError(msg)
        self.usys = usys
        self.rewrap = rewrap
        self._overrides = {
            k: u.dimension_of(u.unit(v)) for k, v in (units or {}).items()
        }
        self._site_units: dict[str, Any] = {}  # static units only
        super().__init__(fn)

    @property
    def site_units(self) -> Mapping[str, Any]:
        """Read-only map of site name to its recorded unit (None: unitless)."""
        return MappingProxyType(self._site_units)

    def _dimension(self, fn: Any, name: str) -> Any:
        if name not in self._overrides:
            return site_dimension(fn, name)
        dim = self._overrides[name]
        if not carrying_params(fn) and _has_quantity(fn):
            msg = (
                f"Site '{name}': units= needs bare parameters, but the "
                "distribution has Quantity parameters with no unit rule."
            )
            raise UnitfulError(msg)
        rule = site_dimension(fn, name) if carrying_params(fn) else None
        if rule is not None and rule != dim:
            msg = f"Site '{name}': units= gives {dim}, the distribution gives {rule}."
            raise UnitfulError(msg)
        return dim

    def _unit(self, dim: Any, name: str) -> Any:
        if dim is None:
            return None
        try:
            return self.usys[dim]
        except (KeyError, ValueError) as e:
            msg = f"Site '{name}': the unit system has no unit for {dim}."
            raise UnitfulError(msg) from e

    def process_message(self, msg: dict[str, Any]) -> None:
        """Strip the units of a sample site and record the site's unit."""
        if msg["type"] != "sample":
            return
        name, fn = msg["name"], msg["fn"]
        dim = self._dimension(fn, name)
        unit = self._unit(dim, name)

        if dim is not None:
            for k, v, _ in carrying_params(fn):
                if not _is_q(v) and not _is_concrete_zero(v):
                    err = (
                        f"Site '{name}': parameter '{k}' is bare next to Quantity "
                        "parameters; give it units."
                    )
                    raise UnitfulError(err)

        # Any value here comes from obs= or an inner handler: inference handlers
        # sit outside `unitful` and set the value after this runs.
        value = msg["value"]
        if _is_q(value):
            expected = dim if dim is not None else u.dimension("dimensionless")
            if u.dimension_of(value) != expected:
                err = (
                    f"Site '{name}': value has dimension {u.dimension_of(value)}, "
                    f"expected {expected}."
                )
                raise UnitfulError(err)
        elif dim is not None and value is not None:
            err = f"Site '{name}': value is bare; give it units of {dim}."
            raise UnitfulError(err)

        if self._site_units.setdefault(name, unit) != unit:
            prev = self._site_units[name]
            err = f"Site '{name}': unit changed from {prev} to {unit} between calls."
            raise UnitfulError(err)

        usys = self.usys
        msg["fn"] = jax.tree.map(
            lambda v: u.ustrip(usys, v) if _is_q(v) else v, fn, is_leaf=_is_q
        )
        if _is_q(value):
            msg["value"] = u.ustrip(usys, value)

    def postprocess_message(self, msg: dict[str, Any]) -> None:
        """Give a sample site's value back to the model as a Quantity."""
        if msg["type"] != "sample" or not self.rewrap:
            return
        unit = self._site_units.get(msg["name"])
        if unit is not None:
            msg["value"] = u.Q(msg["value"], unit)
