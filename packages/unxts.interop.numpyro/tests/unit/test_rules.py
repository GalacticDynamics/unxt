"""Tests for the per-family site dimension rules."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from unxts.interop.numpyro._src.rules import (
    UnitfulError,
    carrying_params,
    dimension_params,
    leaf_path,
    site_dimension,
    strip_powers,
    unwrap,
)

import unxt as u

Q = u.Q


@pytest.mark.parametrize(
    ("d", "dim"),
    [
        (dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")), "length"),
        (dist.Normal(0.0, Q(1.0, "kpc")), "length"),
        (dist.Normal(Q(0.0, ""), Q(1.0, "")), "dimensionless"),
        (dist.HalfNormal(Q(1.0, "km/s")), "speed"),
        (dist.Cauchy(Q(0.0, "kpc"), Q(1.0, "kpc")), "length"),
        (dist.HalfCauchy(Q(1.0, "kpc")), "length"),
        (dist.StudentT(3.0, Q(0.0, "kpc"), Q(1.0, "kpc")), "length"),
        (dist.Laplace(Q(0.0, "kpc"), Q(1.0, "kpc")), "length"),
        (dist.Logistic(Q(0.0, "kpc"), Q(1.0, "kpc")), "length"),
        (dist.Gumbel(Q(0.0, "kpc"), Q(1.0, "kpc")), "length"),
        (dist.Exponential(Q(2.0, "1/kpc")), "length"),
        (dist.Gamma(2.0, Q(2.0, "1/Myr")), "time"),
        (dist.Uniform(Q(0.0, "kpc"), Q(3.0, "kpc")), "length"),
        (dist.Delta(Q(1.0, "kpc")), "length"),
        (
            dist.TruncatedNormal(Q(1.0, "kpc"), Q(1.0, "kpc"), low=Q(0.0, "kpc")),
            "length",
        ),
        (dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")).expand([3]).to_event(1), "length"),
        (dist.VonMises(Q(1.0, "rad"), 2.0), "angle"),
        (
            dist.MixtureSameFamily(
                dist.Categorical(jnp.ones(2) / 2),
                dist.Normal(Q(jnp.zeros(2), "kpc"), Q(1.0, "kpc")),
            ),
            "length",
        ),
        (
            dist.TransformedDistribution(
                dist.Normal(0.0, 1.0),
                dist.transforms.AffineTransform(Q(1.0, "kpc"), Q(2.0, "kpc")),
            ),
            "length",
        ),
        (
            dist.TransformedDistribution(
                dist.Normal(0.0, 1.0),
                dist.transforms.AffineTransform(0.0, Q(2.0, "kpc")),
            ),
            "length",
        ),
    ],
)
def test_site_dimension(d, dim):
    assert site_dimension(d, "x") == u.dimension(dim)


def test_bare_distribution_has_no_dimension():
    assert site_dimension(dist.Normal(0.0, 1.0), "x") is None
    assert site_dimension(dist.LogNormal(0.0, 1.0), "x") is None


def test_mismatched_params_raise():
    with pytest.raises(UnitfulError, match="'x'"):
        site_dimension(dist.Normal(Q(0.0, "kpc"), Q(1.0, "s")), "x")


def test_mismatched_truncation_bound_raises():
    d = dist.TruncatedNormal(Q(1.0, "kpc"), Q(1.0, "kpc"), low=Q(0.0, "s"))
    with pytest.raises(UnitfulError, match="'x'"):
        site_dimension(d, "x")


def test_mismatched_transform_raises():
    t = dist.transforms.AffineTransform(Q(1.0, "s"), Q(2.0, "kpc"))
    with pytest.raises(UnitfulError, match="'x'"):
        site_dimension(dist.TransformedDistribution(dist.Normal(0.0, 1.0), t), "x")


def test_transformed_with_quantity_base_raises():
    t = dist.transforms.AffineTransform(0.0, Q(2.0, "kpc"))
    base = dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc"))
    with pytest.raises(UnitfulError, match="units="):
        site_dimension(dist.TransformedDistribution(base, t), "x")


def test_unregistered_family_with_units_raises():
    with pytest.raises(UnitfulError, match="units="):
        site_dimension(dist.LogNormal(Q(0.0, "kpc"), Q(1.0, "kpc")), "x")


def test_unitful_error_is_value_error():
    assert issubclass(UnitfulError, ValueError)


def test_dimension_params():
    assert dimension_params(dist.Normal(0.0, 1.0)) == {"loc": 1, "scale": 1}
    assert dimension_params(dist.Exponential(1.0)) == {"rate": -1}
    assert dimension_params(dist.Uniform(0.0, 1.0)) == {"low": 1, "high": 1}
    assert dimension_params(dist.LogNormal(0.0, 1.0)) == {}


def test_unwrap():
    base = dist.Normal(Q(jnp.zeros(2), "kpc"), Q(1.0, "kpc"))
    assert unwrap(base.expand([3, 2]).to_event(1)) is base
    assert unwrap(base) is base
    mix = dist.MixtureSameFamily(dist.Categorical(jnp.ones(2) / 2), base)
    assert unwrap(mix) is base


@pytest.mark.parametrize("wrap", [lambda b: b.to_event(1), lambda b: b.expand([3, 2])])
def test_quantity_base_under_wrapper_and_affine_raises(wrap):
    base = wrap(dist.Normal(Q(jnp.zeros(2), "kpc"), Q(1.0, "kpc")))
    t = dist.transforms.AffineTransform(0.0, Q(2.0, "s"))
    with pytest.raises(UnitfulError, match="units="):
        site_dimension(dist.TransformedDistribution(base, t), "x")


def test_wrapped_transformed_distribution():
    t = dist.transforms.AffineTransform(0.0, Q(2.0, "kpc"))
    d = dist.TransformedDistribution(dist.Normal(0.0, 1.0), t).expand([2])
    assert site_dimension(d, "x") == u.dimension("length")


def test_vonmises_requires_angle():
    with pytest.raises(UnitfulError, match="'x'"):
        site_dimension(dist.VonMises(Q(1.0, "kpc"), 2.0), "x")


def test_carrying_params_includes_truncation_bounds():
    d = dist.TruncatedNormal(Q(1.0, "kpc"), Q(1.0, "kpc"), low=5.0)
    assert ("low", 5.0, 1) in carrying_params(d)
    want = {"low", "base_dist.loc", "base_dist.scale"}
    assert {n for n, _, _ in carrying_params(d)} == want


def test_carrying_params_affine_and_mixture():
    t = dist.transforms.AffineTransform(0.0, Q(2.0, "kpc"))
    d = dist.TransformedDistribution(dist.Normal(0.0, 1.0), t)
    got = [(n, p) for n, _, p in carrying_params(d)]
    assert got == [("transforms.0.loc", 1), ("transforms.0.scale", 1)]
    assert carrying_params(dist.Exponential(Q(1.0, "1/kpc")))[0][0] == "rate"


def _paths(d):
    leaves = jax.tree_util.tree_flatten_with_path(d, is_leaf=u.quantity.is_any_quantity)
    return {leaf_path(d, kp) for kp, _ in leaves[0]}


def test_leaf_path_names_support_bounds_after_their_parameter():
    # NumPyro reshapes `low` but the support keeps the original object.
    d = dist.Uniform(Q(jnp.zeros(3), "pc"), Q(1000.0, "pc"))
    assert _paths(d) == {"low", "high"}
    t = dist.TruncatedNormal(Q(1.0, "kpc"), Q(1.0, "kpc"), low=Q(0.0, "kpc"))
    assert _paths(t) == {"low", "base_dist.loc", "base_dist.scale"}


def test_leaf_path_names_container_keys():
    t = [{"a": Q(1.0, "m")}]
    ((kp, _),) = jax.tree_util.tree_flatten_with_path(t)[0]
    assert leaf_path(t, kp) == "0.a.value"


def test_strip_powers_follows_internal_distributions():
    assert strip_powers(dist.HalfCauchy(Q(1.0, "kpc")))["_cauchy.scale"] == 1
    assert strip_powers(dist.Gamma(2.0, Q(1.0, "1/kpc"))) == {"rate": -1}


def test_chi2_df_is_a_shape_parameter():
    assert dimension_params(dist.Chi2(3.0)) == {}
    assert site_dimension(dist.Chi2(Q(3.0, "")), "r") is None
    with pytest.raises(UnitfulError, match="Chi2 has Quantity parameters") as e:
        site_dimension(dist.Chi2(Q(3.0, "kpc")), "r")
    assert "rate" not in str(e.value)
