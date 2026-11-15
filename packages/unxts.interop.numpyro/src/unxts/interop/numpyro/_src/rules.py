"""Per-family rules for the physical dimension of a NumPyro sample site."""

__all__ = ("UnitfulError", "dimension_params", "site_dimension", "unwrap")

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
    """A sample site's units are missing, inconsistent or unsupported."""


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
    | dist.Gumbel
    | dist.VonMises,
) -> dict[str, int]:
    return {"loc": 1, "scale": 1} if hasattr(d, "scale") else {"loc": 1}


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


def _layers(d: dist.Distribution) -> list[dist.Distribution]:
    """Return the wrapper chain from `d` down to `unwrap(d)`, inclusive."""
    out = [d]
    while out[-1] is not unwrap(out[-1]):
        d = out[-1]
        out.append(
            d.base_dist if isinstance(d, _BASE_WRAPPERS) else d.component_distribution
        )
    return out


def _affine_dimension(
    d: dist.TransformedDistribution, name: str
) -> u.AbstractDimension | None:
    """Dimension of a unitless base pushed through a single `AffineTransform`."""
    (t,) = d.transforms
    dims = [u.dimension_of(v) for v in (t.loc, t.scale) if _is_q(v)]
    if not _is_q(t.scale) or any(dm != dims[-1] for dm in dims):
        msg = f"Site '{name}': the AffineTransform loc and scale differ in dimension."
        raise UnitfulError(msg)
    return dims[-1]


def site_dimension(d: dist.Distribution, name: str) -> u.AbstractDimension | None:
    """Return the dimension of the support of `d`, or None for a unitless site.

    Raises `UnitfulError` naming the site `name` when the dimensioned parameters
    disagree, or when the family has no rule but a Quantity parameter.
    """
    if isinstance(d, dist.TransformedDistribution):
        if not _has_quantity(d):
            return None
        single = len(d.transforms) == 1 and isinstance(
            d.transforms[0], tfm.AffineTransform
        )
        if single and not _has_quantity(d.base_dist) and _is_q(d.transforms[0].scale):
            return _affine_dimension(d, name)
        _unregistered(name, d)

    dims = []
    for layer in _layers(d):
        for pname, power in dimension_params(layer).items():
            v = getattr(layer, pname)
            if _is_q(v):
                dims.append(u.dimension_of(v) ** power)
    if not dims:
        if _has_quantity(unwrap(d)):
            _unregistered(name, d)
        return None
    if any(dm != dims[0] for dm in dims):
        msg = f"Site '{name}': parameters have different dimensions: {dims}."
        raise UnitfulError(msg)
    return dims[0]


def _has_quantity(d: dist.Distribution) -> bool:
    """Whether any parameter, or any affine transform parameter, is a Quantity."""
    if isinstance(d, dist.TransformedDistribution):
        tparams = [getattr(t, k, None) for t in d.transforms for k in ("loc", "scale")]
        return any(map(_is_q, tparams)) or _has_quantity(d.base_dist)
    return any(_is_q(getattr(d, k, None)) for k in d.arg_constraints)


def _unregistered(name: str, d: dist.Distribution) -> None:
    msg = (
        f"Site '{name}': {type(d).__name__} has Quantity parameters but no unit "
        f"rule; pass units={{'{name}': ...}} to unitful."
    )
    raise UnitfulError(msg)
