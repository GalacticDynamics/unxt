"""Per-family rules for the physical dimension of a NumPyro sample site."""

__all__ = (
    "UnitfulError",
    "carrying_params",
    "dimension_params",
    "site_dimension",
    "unwrap",
)

from typing import Any

import numpyro.distributions as dist
from numpyro.distributions import transforms as tfm
from plum import dispatch

import unxt as u

# Wrappers whose ``base_dist`` carries the parameters.
_BASE_WRAPPERS = (
    dist.Independent,
    dist.ExpandedDistribution,
    dist.MaskedDistribution,
    dist.LeftTruncatedDistribution,
    dist.RightTruncatedDistribution,
    dist.TwoSidedTruncatedDistribution,
)


class UnitfulError(ValueError):
    """A sample site's units are missing, inconsistent or unsupported.

    Examples
    --------
    >>> import numpyro, numpyro.distributions as dist
    >>> from numpyro import handlers
    >>> import unxt as u
    >>> from unxts.interop.numpyro import UnitfulError, unitful

    >>> def model():
    ...     numpyro.sample("m", dist.Normal(1.0, u.Q(3.0, "kpc")))

    >>> try:
    ...     handlers.seed(unitful(model, u.unitsystems.galactic), 0)()
    ... except UnitfulError as e:
    ...     print(e)
    Site 'm': parameter 'loc' is bare next to Quantity parameters; give it units.

    """


@dispatch
def dimension_params(d: dist.Distribution) -> dict[str, int]:
    """Map parameter name to its power of the site dimension (none: unregistered)."""
    return {}


@dispatch
def dimension_params(
    d: dist.Normal
    | dist.Cauchy
    | dist.StudentT
    | dist.Laplace
    | dist.Logistic
    | dist.Gumbel,
) -> dict[str, int]:
    return {"loc": 1, "scale": 1}


@dispatch
def dimension_params(d: dist.VonMises) -> dict[str, int]:
    return {"loc": 1}


@dispatch
def dimension_params(d: dist.HalfNormal | dist.HalfCauchy) -> dict[str, int]:
    return {"scale": 1}


@dispatch
def dimension_params(
    d: dist.Uniform | dist.TwoSidedTruncatedDistribution,
) -> dict[str, int]:
    return {"low": 1, "high": 1}


@dispatch
def dimension_params(d: dist.LeftTruncatedDistribution) -> dict[str, int]:
    return {"low": 1}


@dispatch
def dimension_params(d: dist.RightTruncatedDistribution) -> dict[str, int]:
    return {"high": 1}


@dispatch
def dimension_params(d: dist.Exponential | dist.Gamma) -> dict[str, int]:
    return {"rate": -1}


@dispatch
def dimension_params(d: dist.Delta) -> dict[str, int]:
    return {"v": 1}


def unwrap(d: dist.Distribution) -> dist.Distribution:
    """Follow wrappers and mixtures down to the distribution that has the units."""
    while True:
        if isinstance(d, _BASE_WRAPPERS):
            d = d.base_dist
        elif isinstance(d, dist.MixtureSameFamily):
            d = d.component_distribution
        else:
            return d


def _is_q(x: object) -> bool:
    return isinstance(x, u.quantity.AbstractQuantity)


def _children(d: dist.Distribution) -> list[dist.Distribution]:
    """Return the distributions directly inside `d`."""
    if isinstance(d, (*_BASE_WRAPPERS, dist.TransformedDistribution)):
        return [d.base_dist]
    if isinstance(d, dist.MixtureSameFamily):
        return [d.component_distribution]
    return []


def _has_quantity(d: dist.Distribution) -> bool:
    """Whether any parameter at any depth, or any transform parameter, is a Quantity."""
    if any(_is_q(getattr(d, k, None)) for k in d.arg_constraints):
        return True
    if isinstance(d, dist.TransformedDistribution) and any(
        _is_q(getattr(t, k, None)) for t in d.transforms for k in ("loc", "scale")
    ):
        return True
    return any(_has_quantity(c) for c in _children(d))


def carrying_params(d: dist.Distribution) -> list[tuple[str, Any, int]]:
    """Return ``(name, value, power)`` for every parameter carrying the site dimension.

    The walk goes through wrappers (including truncation bounds), mixtures and a
    single `AffineTransform` over a unitless base. Values may be bare or Quantity.
    Other transformed distributions give an empty list.
    """
    if isinstance(d, dist.TransformedDistribution):
        (t,) = d.transforms if len(d.transforms) == 1 else (None,)
        if isinstance(t, tfm.AffineTransform) and not _has_quantity(d.base_dist):
            return [("loc", t.loc, 1), ("scale", t.scale, 1)]
        return []
    own = [(k, getattr(d, k), p) for k, p in dimension_params(d).items()]
    return own + [e for c in _children(d) for e in carrying_params(c)]


def site_dimension(d: dist.Distribution, name: str) -> u.AbstractDimension | None:
    """Return the dimension of the support of `d`, or None for a unitless site.

    Raises `UnitfulError` naming the site `name` when the dimensioned parameters
    disagree, when the family has no rule but a Quantity parameter, or when a
    VonMises site is not an angle.
    """
    entries = carrying_params(d)
    dims = [u.dimension_of(v) ** p for _, v, p in entries if _is_q(v)]
    if not dims:
        if _has_quantity(d):
            msg = (
                f"Site '{name}': {type(unwrap(d)).__name__} has Quantity parameters "
                f"but no unit rule; pass units={{'{name}': ...}} to unitful."
            )
            raise UnitfulError(msg)
        return None
    if any(dm != dims[0] for dm in dims):
        msg = f"Site '{name}': parameters have different dimensions: {dims}."
        raise UnitfulError(msg)
    if isinstance(unwrap(d), dist.VonMises) and dims[0] != u.dimension("angle"):
        msg = f"Site '{name}': VonMises needs an angle, got {dims[0]}."
        raise UnitfulError(msg)
    return dims[0]
