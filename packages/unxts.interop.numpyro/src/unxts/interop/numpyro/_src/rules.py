"""Per-family rules for the physical dimension of a NumPyro sample site."""

__all__ = (
    "UnitfulError",
    "carrying_params",
    "dimension_params",
    "leaf_path",
    "site_dimension",
    "strip_powers",
    "unwrap",
)

import contextlib
from typing import Any

import jax
import numpyro.distributions as dist
from numpyro.distributions import transforms as tfm
from plum import Dispatcher

import unxt as u

# Private, so the package's methods never mix with other `plum.dispatch` users.
dispatch = Dispatcher()

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


UnitfulError.__module__ = "unxts.interop.numpyro"  # the public path, for tracebacks


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
def dimension_params(d: dist.Chi2) -> dict[str, int]:
    # `Chi2` is a `Gamma`, but its only parameter, `df`, is a shape parameter.
    return {}


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


def _children(d: dist.Distribution) -> list[tuple[str, dist.Distribution]]:
    """Return ``(attribute, distribution)`` for each distribution directly in `d`."""
    if isinstance(d, (*_BASE_WRAPPERS, dist.TransformedDistribution)):
        return [("base_dist", d.base_dist)]
    if isinstance(d, dist.MixtureSameFamily):
        return [("component_distribution", d.component_distribution)]
    return []


def _has_quantity(d: dist.Distribution) -> bool:
    """Whether any leaf of `d`, at any depth, is a Quantity.

    Distributions and transforms are pytrees, so this also sees composites
    without a unit rule (`MixtureGeneral`, the censored distributions,
    `ComposeTransform` parts).
    """
    return any(_is_q(x) for x in jax.tree.leaves(d, is_leaf=_is_q))


def _dimensioned(d: dist.Distribution) -> bool:
    """Whether any leaf of `d`, at any depth, is a Quantity with a dimension."""
    none = u.dimension("dimensionless")
    leaves = jax.tree.leaves(d, is_leaf=_is_q)
    return any(_is_q(x) and u.dimension_of(x) != none for x in leaves)


# A support constraint holds the bounds it was built from.
_BOUNDS = {"_support.lower_bound": "low", "_support.upper_bound": "high"}


def leaf_path(tree: Any, path: tuple[Any, ...]) -> str:
    """Return the dotted attribute path of the leaf at the JAX key `path`.

    NumPyro distributions, constraints and transforms flatten to unnamed
    children, so the names come from their ``tree_flatten`` field lists. A
    support bound is named after the parameter it comes from, so the upper
    bound of a `Uniform` support is ``"high"``.
    """
    names = []
    for key in path:
        if isinstance(key, jax.tree_util.FlattenedIndexKey):
            children, aux = tree.tree_flatten()
            fields = (
                type(tree).gather_pytree_data_fields()
                if isinstance(tree, dist.Distribution)
                else aux[0]
            )
            names.append(fields[key.key])
            tree = children[key.key]
        elif isinstance(key, jax.tree_util.SequenceKey):
            names.append(str(key.idx))
            tree = tree[key.idx]
        elif isinstance(key, jax.tree_util.DictKey):
            names.append(str(key.key))
            tree = tree[key.key]
        else:  # GetAttrKey
            names.append(key.name)
            tree = getattr(tree, key.name)
    out = ".".join(names)
    for b, p in _BOUNDS.items():
        if out == b or out.endswith("." + b):
            out = out[: -len(b)] + p
    return out


def carrying_params(d: dist.Distribution) -> list[tuple[str, Any, int]]:
    """Return ``(path, value, power)`` for every parameter carrying the site dimension.

    The path is the dotted attribute path from `d`, as `leaf_path` gives it.
    The walk goes through wrappers (including truncation bounds), mixtures and a
    single `AffineTransform` over a unitless base. Values may be bare or Quantity.
    Other transformed distributions give an empty list.
    """
    if isinstance(d, dist.TransformedDistribution):
        (t,) = d.transforms if len(d.transforms) == 1 else (None,)
        if isinstance(t, tfm.AffineTransform) and not _has_quantity(d.base_dist):
            return [("transforms.0.loc", t.loc, 1), ("transforms.0.scale", t.scale, 1)]
        return []
    own = [(k, getattr(d, k), p) for k, p in dimension_params(d).items()]
    return own + [
        (f"{a}.{k}", v, p) for a, c in _children(d) for k, v, p in carrying_params(c)
    ]


def strip_powers(d: dist.Distribution) -> dict[str, int]:
    """Map the path of every leaf that carries the site dimension to its power.

    Like `carrying_params`, but it also follows a family's internal
    distributions, such as the ``_cauchy`` that `HalfCauchy` builds from its
    ``scale``, so their copies of the parameters are stripped the same way.
    """
    if isinstance(d, dist.TransformedDistribution):
        return {k: p for k, _, p in carrying_params(d)}
    out = dict(dimension_params(d))
    for f in type(d).gather_pytree_data_fields():
        c = d.__dict__.get(f)
        if isinstance(c, dist.Distribution):
            out |= {f"{f}.{k}": p for k, p in strip_powers(c).items()}
    return out


def _no_rule_hint(d: dist.Distribution, name: str, usys: Any) -> str:
    """Say what to do with dimensioned Quantities in a family with no unit rule."""
    support = getattr(type(unwrap(d)), "support", None)
    try:
        count = isinstance(support, dist.constraints.Constraint) and support.is_discrete
    except NotImplementedError:  # a support that depends on the parameters
        count = False
    if count:
        return "its samples are counts, so its parameters must be dimensionless."
    none = u.dimension("dimensionless")
    leaves = jax.tree.leaves(d, is_leaf=_is_q)
    dims = {u.dimension_of(x) for x in leaves if _is_q(x)} - {none}
    hint = "<the site's dimension>"
    if len(dims) == 1:
        (dim,) = dims
        hint = repr(str(dim))
        with contextlib.suppress(KeyError, ValueError, TypeError):
            hint = repr(str(usys[dim]))
    return (
        "give it bare parameters in the units of usys and pass "
        f"units={{'{name}': {hint}}} to unitful."
    )


def site_dimension(
    d: dist.Distribution, name: str, usys: Any = None
) -> u.AbstractDimension | None:
    """Return the dimension of the support of `d`, or None for a unitless site.

    A site is unitless when none of the parameters that carry its dimension is
    a Quantity and no Quantity anywhere in `d` has a dimension. A site whose
    carrying Quantities are dimensionless has dimension ``dimensionless``. Raises
    `UnitfulError` naming the site `name` when the dimensioned parameters
    disagree, when a Quantity with a dimension is in a family with no rule or
    next to bare dimension-carrying parameters, or when a VonMises site is not
    an angle. `usys`, if given, names the unit to suggest in the error.
    """
    entries = carrying_params(d)
    dims = [u.dimension_of(v) ** p for _, v, p in entries if _is_q(v)]
    if not dims:
        # A dimensionless Quantity says nothing about the site's dimension.
        if not _dimensioned(d):
            return None
        fam = type(unwrap(d)).__name__
        if entries:
            bare = ", ".join(f"'{k.rsplit('.', 1)[-1]}'" for k, _, _ in entries)
            msg = (
                f"Site '{name}': {fam} has Quantity parameters, but the parameters "
                f"that carry the site's dimension ({bare}) are bare; give them units."
            )
        else:
            msg = (
                f"Site '{name}': {fam} has Quantity parameters with dimensions but "
                f"no unit rule; {_no_rule_hint(d, name, usys)}"
            )
        raise UnitfulError(msg)
    if any(dm != dims[0] for dm in dims):
        names = ", ".join(sorted({str(dm) for dm in dims}))
        msg = f"Site '{name}': parameters have different dimensions: {names}."
        raise UnitfulError(msg)
    if isinstance(unwrap(d), dist.VonMises) and dims[0] != u.dimension("angle"):
        msg = f"Site '{name}': VonMises needs an angle, got {dims[0]}."
        raise UnitfulError(msg)
    return dims[0]
