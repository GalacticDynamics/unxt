"""Tests for `unitful` at sample sites."""

import ast
import re

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from unxts.interop.numpyro import UnitfulError, unitful

import unxt as u

Q = u.Q


def model_ref(x=None):
    m = numpyro.sample("m", dist.Normal(0.0, 3.0))
    s = numpyro.sample("s", dist.HalfNormal(1.0))
    with numpyro.plate("n", 200):
        numpyro.sample("x", dist.Normal(m, s), obs=x)


def model(x=None):
    m = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
    s = numpyro.sample("s", dist.HalfNormal(Q(1.0, "kpc")))
    with numpyro.plate("n", 200):
        numpyro.sample("x", dist.Normal(m, s), obs=x)


def ld(fn, x, params):
    return log_density(fn, (x,), {}, params)[0]


def test_log_density_matches_reference(usys, x, p):
    ref = ld(model_ref, u.ustrip("kpc", x), p)
    m = unitful(model, usys)
    np.testing.assert_allclose(ld(m, x, p), ref, rtol=1e-5)
    np.testing.assert_allclose(jax.jit(lambda q: ld(m, x, q))(p), ref, rtol=1e-5)
    assert not any(isinstance(v, jax.core.Tracer) for v in m.site_units.values())


def test_vector_loc_scalar_scale_vmap(usys, x, p):
    def model_vec(x=None):
        m = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
        s = numpyro.sample("s", dist.HalfNormal(Q(1.0, "kpc")))
        with numpyro.plate("n", 200):
            numpyro.sample("x", dist.Normal(m + Q(jnp.zeros(200), "kpc"), s), obs=x)

    ps = {"m": jnp.asarray([0.9, 1.0, 1.1]), "s": jnp.asarray([0.2, 0.3, 0.2])}
    xr = u.ustrip("kpc", x)
    ref = jax.vmap(lambda q: ld(model_ref, xr, q))(ps)
    got = jax.vmap(lambda q: ld(unitful(model_vec, usys), x, q))(ps)
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_grad_matches_reference(usys, x, p):
    ref = jax.grad(lambda q: ld(model_ref, u.ustrip("kpc", x), q))(p)
    got = jax.grad(lambda q: ld(unitful(model, usys), x, q))(p)
    for k in p:
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-5)


def test_latent_returned_as_quantity(usys, x, p):
    seen = {}

    def m():
        seen["m"] = numpyro.sample("m", dist.Normal(Q(0.0, "pc"), Q(3.0, "pc")))

    handlers.seed(unitful(m, usys), 0)()
    assert isinstance(seen["m"], u.AbstractQuantity)
    assert seen["m"].unit == u.unit("kpc")


def test_rewrap_false_returns_bare(usys):
    seen = {}

    def m():
        seen["m"] = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))

    handlers.seed(unitful(m, usys, rewrap=False), 0)()
    assert not isinstance(seen["m"], u.AbstractQuantity)


def test_trace_stores_bare_values(usys, x, p):
    tr = handlers.trace(handlers.substitute(unitful(model, usys), p)).get_trace(x)
    assert not any(isinstance(s["value"], u.AbstractQuantity) for s in tr.values())


def test_obs_in_other_unit_matches_reference(usys, x, p):
    ref = ld(model_ref, u.ustrip("kpc", x), p)
    got = ld(unitful(model, usys), x.uconvert("pc"), p)
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_obs_wrong_dimension_raises_under_jit(usys, p):
    xs = Q(jnp.ones(200), "km/s")
    with pytest.raises(UnitfulError, match="'x'"):
        jax.jit(lambda q: ld(unitful(model, usys), xs, q))(p)


def test_bare_obs_at_dimensioned_site_raises(usys, x, p):
    with pytest.raises(UnitfulError, match="'x'"):
        ld(unitful(model, usys), u.ustrip("kpc", x), p)


def test_bare_param_next_to_quantity_raises(usys):
    def bad():
        numpyro.sample("a", dist.Normal(Q(0.0, "pc"), 1.0))

    def ok():
        numpyro.sample("a", dist.Normal(0.0, Q(1.0, "kpc")))

    with pytest.raises(UnitfulError, match="'a'"):
        handlers.seed(unitful(bad, usys), 0)()
    handlers.seed(unitful(ok, usys), 0)()


def test_traced_bare_param_raises(usys):
    def bad(_=None):
        m = numpyro.sample("m", dist.Normal(0.0, 1.0))
        numpyro.sample("a", dist.Normal(m, Q(1.0, "kpc")))

    with pytest.raises(UnitfulError, match="'a'"):
        jax.jit(lambda q: ld(unitful(bad, usys), None, q))({"m": 0.0, "a": 1.0})


def test_bare_model_is_unchanged(usys, x, p):
    xr = u.ustrip("kpc", x)
    np.testing.assert_allclose(
        ld(unitful(model_ref, usys), xr, p), ld(model_ref, xr, p), rtol=0
    )


def test_dimension_missing_from_usys_raises(usys):
    def m():
        numpyro.sample("t", dist.Normal(Q(1.0, "K"), Q(1.0, "K")))

    with pytest.raises(UnitfulError, match="'t'"):
        handlers.seed(unitful(m, usys), 0)()


def test_units_override(usys):
    seen = {}

    def m():
        seen["r"] = numpyro.sample(
            "r", dist.ImproperUniform(dist.constraints.positive, (), ())
        )

    h = unitful(m, usys, units={"r": "length"})
    handlers.seed(handlers.substitute(h, {"r": 2.0}), 0)()
    assert seen["r"].unit == u.unit("kpc")
    assert h.site_units["r"] == u.unit("kpc")


def test_units_override_disagreeing_with_rule_raises(usys):
    def m():
        numpyro.sample("r", dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")))

    with pytest.raises(UnitfulError, match="'r'"):
        handlers.seed(unitful(m, usys, units={"r": "s"}), 0)()


def test_site_unit_change_raises(usys):
    def m(unit):
        numpyro.sample("a", dist.Normal(Q(0.0, unit), Q(1.0, unit)))

    h = handlers.seed(unitful(m, usys), 0)
    h("kpc")
    h("pc")  # same dimension, same usys unit
    with pytest.raises(UnitfulError, match="'a'"):
        h("s")


def test_fn_required(usys):
    with pytest.raises(TypeError, match="wrap"):
        unitful(None, usys)


def test_condition_inside_handler(usys, x, p):
    ref = ld(model_ref, u.ustrip("kpc", x), p)
    c = unitful(handlers.condition(model, {"m": Q(1000.0, "pc")}), usys)
    np.testing.assert_allclose(ld(c, x, {"s": p["s"]}), ref, rtol=1e-5)


def test_bare_value_from_inner_substitute_raises(usys, x):
    h = unitful(handlers.substitute(model, {"m": 1.0}), usys)
    with pytest.raises(UnitfulError, match="'m'"):
        ld(h, x, {"s": 0.2})


def test_units_override_with_unruled_quantity_params_raises(usys):
    def m():
        numpyro.sample("r", dist.LogNormal(Q(1.0, "pc"), Q(1.0, "s")))

    with pytest.raises(UnitfulError, match="'r'"):
        handlers.seed(unitful(m, usys, units={"r": "length"}), 0)()


def test_nuts_runs(usys, x):
    mc = MCMC(
        NUTS(unitful(model, usys)), num_warmup=50, num_samples=50, progress_bar=False
    )
    mc.run(jax.random.PRNGKey(0), x)
    m = mc.get_samples()["m"]
    assert not isinstance(m, u.AbstractQuantity)
    assert abs(float(jnp.mean(m)) - 1.0) < 0.1


def test_units_key_matching_no_site_raises(usys):
    def m():
        with handlers.scope(prefix="gal"):
            numpyro.sample("r", dist.LogNormal(0.0, 1.0))

    with pytest.raises(UnitfulError, match=r"\['r'\]"):
        handlers.seed(unitful(m, usys, units={"r": "length"}), 0)()

    def maybe(*, draw):
        if draw:
            numpyro.sample("r", dist.LogNormal(0.0, 1.0))

    match = r"\['r'\], which no call of the model has reached yet"
    with pytest.raises(UnitfulError, match=match):
        handlers.seed(unitful(maybe, usys, units={"r": "length"}), 0)(draw=False)
    r = handlers.seed(unitful(m, usys, units={"gal/r": "kpc"}), 0)
    assert handlers.trace(r).get_trace()["gal/r"]["value"].ndim == 0


@pytest.mark.parametrize(
    "site",
    [
        lambda: numpyro.deterministic("r", Q(1.0, "kpc")),
        lambda: numpyro.param("r", 1.0),
        lambda: numpyro.factor("r", -1.0),
    ],
    ids=["deterministic", "param", "factor"],
)
def test_units_key_at_non_sample_site_raises(usys, site):
    with pytest.raises(UnitfulError, match="'r': units= applies to sample sites"):
        handlers.seed(unitful(site, usys, units={"r": "length"}), 0)()


def _pc(x):
    return Q(x, "pc")


_TN = dist.TruncatedNormal
_MIX = dist.Categorical(jnp.array([0.3, 0.7]))
_AFF = dist.transforms.AffineTransform


@pytest.mark.parametrize(
    ("d", "ref", "value"),
    [
        (dist.Normal(_pc(0.0), _pc(3000.0)), dist.Normal(0.0, 3.0), 1.0),
        (dist.Exponential(Q(1.0, "1/pc")), dist.Exponential(1000.0), 0.001),
        (dist.Uniform(_pc(0.0), _pc(2000.0)), dist.Uniform(0.0, 2.0), 1.0),
        (
            _TN(_pc(1000.0), _pc(500.0), low=_pc(500.0), high=_pc(3000.0)),
            _TN(1.0, 0.5, low=0.5, high=3.0),
            1.2,
        ),
        (_TN(_pc(1000.0), _pc(500.0), low=_pc(500.0)), _TN(1.0, 0.5, low=0.5), 1.2),
        (_TN(_pc(1000.0), _pc(500.0), high=_pc(3000.0)), _TN(1.0, 0.5, high=3.0), 1.2),
        (
            dist.TruncatedDistribution(dist.Cauchy(_pc(0.0), _pc(500.0)), _pc(0.0)),
            dist.TruncatedDistribution(dist.Cauchy(0.0, 0.5), 0.0),
            1.2,
        ),
        (dist.HalfNormal(_pc(500.0)), dist.HalfNormal(0.5), 1.2),
        (dist.HalfCauchy(_pc(500.0)), dist.HalfCauchy(0.5), 1.2),
        (dist.Gamma(2.0, Q(1.0, "1/pc")), dist.Gamma(2.0, 1000.0), 0.001),
        (
            dist.TransformedDistribution(
                dist.Normal(0.0, 1.0), _AFF(_pc(1000.0), _pc(500.0))
            ),
            dist.TransformedDistribution(dist.Normal(0.0, 1.0), _AFF(1.0, 0.5)),
            1.2,
        ),
        (
            dist.MixtureSameFamily(
                _MIX, dist.Normal(_pc(jnp.array([0.0, 2000.0])), _pc(500.0))
            ),
            dist.MixtureSameFamily(_MIX, dist.Normal(jnp.array([0.0, 2.0]), 0.5)),
            1.2,
        ),
        (
            dist.StudentT(3.0, _pc(1000.0), _pc(500.0)),
            dist.StudentT(3.0, 1.0, 0.5),
            1.2,
        ),
        (dist.Laplace(_pc(1000.0), _pc(500.0)), dist.Laplace(1.0, 0.5), 1.2),
    ],
    ids=[
        "normal",
        "exponential",
        "uniform",
        "two-sided",
        "low",
        "high",
        "truncated-cauchy",
        "halfnormal",
        "halfcauchy",
        "gamma",
        "affine",
        "mixture",
        "studentt",
        "laplace",
    ],
)
def test_params_stripped_in_usys_not_their_own_unit(usys, d, ref, value):
    # galactic: the bare value and the reference are in kpc
    sub = {"a": jnp.asarray(value)}
    got = log_density(unitful(lambda: numpyro.sample("a", d), usys), (), {}, sub)[0]
    want = log_density(lambda: numpyro.sample("a", ref), (), {}, sub)[0]
    np.testing.assert_allclose(got, want, rtol=1e-5)


_T = dist.transforms


def _mixgen(loc, scale):
    comps = [dist.Normal(loc, scale), dist.Normal(loc, scale)]
    return dist.MixtureGeneral(dist.Categorical(jnp.array([0.5, 0.5])), comps)


def _censored(loc, scale):
    return dist.RightCensoredDistribution(dist.Normal(loc, scale), censored=0)


def _compose(loc, scale):
    t = _T.ComposeTransform(
        [_T.AffineTransform(loc, scale), _T.AffineTransform(0.0, 2.0)]
    )
    return dist.TransformedDistribution(dist.Normal(0.0, 1.0), t)


_COMPOSITES = pytest.mark.parametrize(
    "make", [_mixgen, _censored, _compose], ids=["mixgen", "censored", "compose"]
)


@_COMPOSITES
@pytest.mark.parametrize("obs", [None, 0.5, Q(500.0, "pc")], ids=["none", "bare", "q"])
def test_unruled_composite_with_quantity_raises(usys, make, obs):
    def m():
        numpyro.sample("x", make(Q(0.0, "pc"), Q(1.0, "pc")), obs=obs)

    with pytest.raises(UnitfulError, match=r"'x'.*no unit rule"):
        handlers.seed(unitful(m, usys), 0)()


@_COMPOSITES
def test_unruled_composite_all_bare_unchanged(usys, make):
    def m():
        numpyro.sample("x", make(0.0, 1.0), obs=0.5)

    h = unitful(m, usys)
    np.testing.assert_allclose(
        log_density(h, (), {}, {})[0], log_density(m, (), {}, {})[0], rtol=0
    )
    assert h.site_units["x"] is None


@_COMPOSITES
def test_unruled_composite_bare_with_units(usys, make):
    # Bare parameters in usys units plus units= give the site its dimension.
    def m():
        numpyro.sample("x", make(0.0, 1.0), obs=Q(500.0, "pc"))

    def ref():
        numpyro.sample("x", make(0.0, 1.0), obs=0.5)

    h = unitful(m, usys, units={"x": "length"})
    np.testing.assert_allclose(
        log_density(h, (), {}, {})[0], log_density(ref, (), {}, {})[0], rtol=1e-6
    )


def test_vonmises_always_in_radians():
    deg = u.unitsystem("kpc", "Myr", "Msun", "deg")
    seen = {}

    def obs():
        numpyro.sample("t", dist.VonMises(Q(0.0, "rad"), 2.0), obs=Q(0.5, "rad"))

    def latent():
        seen["t"] = numpyro.sample("t", dist.VonMises(Q(0.0, "deg"), 2.0))

    want = dist.VonMises(0.0, 2.0).log_prob(0.5)
    np.testing.assert_allclose(log_density(unitful(obs, deg), (), {}, {})[0], want)
    h = unitful(latent, deg)
    tr = handlers.trace(handlers.seed(h, 0)).get_trace()
    assert seen["t"].unit == u.unit("rad")
    assert h.site_units["t"] == u.unit("rad")
    assert -np.pi <= float(tr["t"]["value"]) <= np.pi


@pytest.mark.parametrize(
    ("d", "param"),
    [
        (dist.Gamma(Q(2.0, "pc"), Q(1.0, "1/kpc")), "concentration"),
        (dist.VonMises(Q(0.0, "rad"), Q(2.0, "pc")), "concentration"),
    ],
    ids=["gamma", "vonmises"],
)
def test_dimensioned_shape_parameter_raises(usys, d, param):
    with pytest.raises(UnitfulError, match=f"'x': shape parameter '{param}'"):
        handlers.seed(unitful(lambda: numpyro.sample("x", d), usys), 0)()


def test_dimensionless_shape_parameter_stripped(usys):
    def m():
        d = dist.Gamma(Q(2.0, ""), Q(1.0, "1/kpc"))
        numpyro.sample("x", d, obs=Q(1.0, "kpc"))

    want = dist.Gamma(2.0, 1.0).log_prob(1.0)
    np.testing.assert_allclose(log_density(unitful(m, usys), (), {}, {})[0], want)


def test_traced_zero_hint_and_numpy_zeros(usys):
    def with_jnp(_=None):
        numpyro.sample("x", dist.Normal(jnp.zeros(3), Q(3.0, "kpc")).to_event(1))

    def with_np(_=None):
        numpyro.sample("x", dist.Normal(np.zeros(3), Q(3.0, "kpc")).to_event(1))

    sub = {"x": jnp.ones(3)}
    with pytest.raises(UnitfulError, match=r"0\.0 or np\.zeros"):
        jax.jit(lambda q: log_density(unitful(with_jnp, usys), (), {}, q)[0])(sub)
    jax.jit(lambda q: log_density(unitful(with_np, usys), (), {}, q)[0])(sub)
    mc = MCMC(
        NUTS(unitful(with_np, usys)), num_warmup=10, num_samples=10, progress_bar=False
    )
    mc.run(jax.random.PRNGKey(0))
    assert mc.get_samples()["x"].shape == (10, 3)


@pytest.mark.parametrize(
    "unit", ["length", u.dimension("length"), "kpc", u.unit("kpc")]
)
def test_units_dimension_or_usys_unit(usys, unit):
    def m():
        return numpyro.sample("r", dist.LogNormal(0.0, 1.0))

    assert handlers.seed(unitful(m, usys, units={"r": unit}), 0)().unit == "kpc"


def test_units_other_unit_of_dimension_raises(usys):
    with pytest.raises(
        UnitfulError, match=r"'r'.*read in the unit of usys; pass units=\{'r': 'kpc'\}"
    ):
        unitful(lambda: None, usys, units={"r": "pc"})


def test_units_on_bare_family_with_rule_message(usys):
    def m():
        numpyro.sample("x", dist.Normal(10.0, 1.0))

    with pytest.raises(UnitfulError, match="Normal has a unit rule") as e:
        handlers.seed(unitful(m, usys, units={"x": "kpc"}), 0)()
    assert "next to Quantity" not in str(e.value)


def _ld(model, usys, sub, **kw):
    return log_density(unitful(model, usys, **kw), (), {}, sub)[0]


_BROADCAST = pytest.mark.parametrize(
    ("d", "ref", "value"),
    [
        (
            dist.Uniform(Q(jnp.zeros(3), "pc"), Q(1000.0, "pc")),
            dist.Uniform(jnp.zeros(3), 1.0),
            [0.1, 0.2, 0.3],
        ),
        (
            dist.Uniform(Q(0.0, "pc"), Q(jnp.array([1000.0, 2000.0]), "pc")),
            dist.Uniform(0.0, jnp.array([1.0, 2.0])),
            [0.5, 1.5],
        ),
        (
            dist.TruncatedNormal(
                Q(jnp.ones(3), "kpc"), Q(1.0, "kpc"), low=Q(0.0, "kpc")
            ),
            dist.TruncatedNormal(jnp.ones(3), 1.0, low=0.0),
            [0.5, 1.0, 1.5],
        ),
        (
            dist.TruncatedNormal(
                Q(jnp.ones(3), "kpc"),
                Q(1.0, "kpc"),
                low=Q(0.0, "kpc"),
                high=Q(jnp.array([2.0, 3.0, 4.0]), "kpc"),
            ),
            dist.TruncatedNormal(
                jnp.ones(3), 1.0, low=0.0, high=jnp.array([2.0, 3.0, 4.0])
            ),
            [0.5, 1.0, 1.5],
        ),
    ],
    ids=["vector-low", "vector-high", "truncnormal-low", "two-sided-vector-high"],
)


@_BROADCAST
def test_broadcast_bounds_match_reference(usys, d, ref, value):
    # NumPyro reshapes the bounds, and the support keeps the original objects.
    sub = {"x": jnp.asarray(value)}
    want = log_density(lambda: numpyro.sample("x", ref), (), {}, sub)[0]
    got = _ld(lambda: numpyro.sample("x", d), usys, sub)
    np.testing.assert_allclose(got, want, rtol=1e-6)
    got = jax.jit(lambda q: _ld(lambda: numpyro.sample("x", d), usys, q))(sub)
    np.testing.assert_allclose(got, want, rtol=1e-6)


@_BROADCAST
def test_broadcast_bounds_sample_in_support(usys, d, ref, value):
    del value
    x = handlers.seed(unitful(lambda: numpyro.sample("x", d), usys), 0)()
    assert bool(jnp.all(ref.support(u.ustrip("kpc", x))))


def test_dimensionless_value_at_unitless_site(usys):
    # kpc/pc, so stripping the raw value instead of converting is off by 1000.
    half = Q(0.5, "kpc") / Q(1000.0, "pc")

    def beta(v):
        numpyro.sample("x", dist.Beta(2.0, 2.0), obs=v)

    def normal(v):
        numpyro.sample("x", dist.Normal(0.0, 1.0).expand([2]), obs=v)

    def ld(m, v):
        return log_density(unitful(m, usys), (v,), {}, {})[0]

    np.testing.assert_allclose(ld(beta, half), dist.Beta(2.0, 2.0).log_prob(0.5))
    v = Q(jnp.array([1.0, 2.0]), "kpc") / Q(jnp.array([500.0, 800.0]), "pc")
    want = dist.Normal(0.0, 1.0).log_prob(jnp.array([2.0, 2.5])).sum()
    np.testing.assert_allclose(ld(normal, v), want, rtol=1e-6)
    vs = Q(jnp.array([0.25, 0.5]), "kpc") / Q(1000.0, "pc")
    got = jax.vmap(lambda q: ld(beta, q))(vs)
    np.testing.assert_allclose(
        got, dist.Beta(2.0, 2.0).log_prob(jnp.array([0.25, 0.5]))
    )
    # Set by a handler outside unitful.
    c = handlers.condition(unitful(lambda: beta(None), usys), {"x": half})
    tr = handlers.trace(handlers.seed(c, 0)).get_trace()
    np.testing.assert_allclose(tr["x"]["value"], 0.5)


@pytest.mark.parametrize(
    ("make", "bare", "obs"),
    [
        (
            lambda: dist.Bernoulli(logits=Q(2.0, "1/kpc") * Q(500.0, "pc")),
            lambda: dist.Bernoulli(logits=1.0),
            1.0,
        ),
        (lambda: dist.Poisson(Q(3.0, "")), lambda: dist.Poisson(3.0), 2.0),
        (
            lambda: dist.Poisson(Q(3.0, "kpc") / Q(1000.0, "pc")),
            lambda: dist.Poisson(3.0),
            2.0,
        ),
        (
            lambda: dist.Binomial(10, probs=Q(0.3, "kpc") / Q(1000.0, "pc")),
            lambda: dist.Binomial(10, probs=0.3),
            4.0,
        ),
        (lambda: dist.Gamma(Q(2.0, ""), 0.5), lambda: dist.Gamma(2.0, 0.5), 1.5),
    ],
    ids=["bernoulli-logits", "poisson", "poisson-kpc-per-pc", "binomial", "gamma"],
)
def test_dimensionless_quantity_in_unruled_family(usys, make, bare, obs):
    h = unitful(lambda: numpyro.sample("x", make(), obs=obs), usys)
    got = log_density(h, (), {}, {})[0]
    np.testing.assert_allclose(got, bare().log_prob(obs), rtol=1e-6)
    assert h.site_units["x"] is None


def test_dimensioned_shape_with_bare_rule_params_names_them(usys):
    def m():
        numpyro.sample("x", dist.Gamma(Q(2.0, "kpc"), 0.5))

    with pytest.raises(UnitfulError, match=r"'x'.*\('rate'\) are bare"):
        handlers.seed(unitful(m, usys), 0)()


def test_units_suggestion_works_when_pasted_back(usys):
    def m():
        return numpyro.sample("v", dist.LogNormal(0.0, 1.0))

    with pytest.raises(UnitfulError, match="units=") as e:
        unitful(m, usys, units={"v": "km/s"})
    units = ast.literal_eval(re.search(r"units=(\{.*\})", str(e.value)).group(1))
    v = handlers.seed(unitful(m, usys, units=units), 0)()
    assert u.dimension_of(v) == u.dimension("speed")
    assert v.unit == usys["speed"]


def test_quantity_as_units_value_raises(usys):
    with pytest.raises(UnitfulError, match="'r': units= takes a dimension or a unit"):
        unitful(lambda: None, usys, units={"r": Q(1.0, "kpc")})


def test_units_key_under_outer_scope(usys):
    def ln():
        return numpyro.sample("r", dist.LogNormal(0.0, 1.0))

    h = unitful(ln, usys, units={"a/r": "length"})
    with pytest.raises(UnitfulError, match=r"\['a/r'\].*name inside unitful"):
        handlers.seed(handlers.scope(h, "a"), 0)()
    h = unitful(ln, usys, units={"r": "length"})
    assert handlers.seed(handlers.scope(h, "a"), 0)().unit == "kpc"
    assert h.site_units["a/r"] == u.unit("kpc")


def test_bare_infinite_bound_allowed(usys):
    def m():
        d = dist.TruncatedNormal(
            Q(1.0, "kpc"), Q(1.0, "kpc"), low=-jnp.inf, high=Q(2000.0, "pc")
        )
        numpyro.sample("x", d, obs=Q(1.5, "kpc"))

    want = dist.TruncatedNormal(1.0, 1.0, low=-jnp.inf, high=2.0).log_prob(1.5)
    np.testing.assert_allclose(log_density(unitful(m, usys), (), {}, {})[0], want)


# A dimensionless site takes bare numbers like a unitless one.
_R = Q(jnp.linspace(1.0, 10.0, 5), "kpc")
_Y = jnp.linspace(0.0, 1.0, 5)


@pytest.mark.parametrize(
    ("make", "bare", "obs"),
    [
        (
            lambda: dist.Normal(0.5 + 0.1 * (_R / Q(1.0, "kpc")), 0.3),
            lambda: dist.Normal(0.5 + 0.1 * jnp.linspace(1.0, 10.0, 5), 0.3),
            _Y,
        ),
        (lambda: dist.Normal(Q(0.0, ""), 1.0), lambda: dist.Normal(0.0, 1.0), 0.4),
        (lambda: dist.Normal(1.0, Q(1.0, "")), lambda: dist.Normal(1.0, 1.0), 0.4),
        (
            lambda: dist.Normal(1.0, Q(2.0, "kpc") / Q(1000.0, "pc")),
            lambda: dist.Normal(1.0, 2.0),
            0.4,
        ),
    ],
    ids=["gradient", "dimless-loc", "dimless-scale", "kpc-per-pc"],
)
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_bare_next_to_dimensionless(usys, make, bare, obs, jit):
    h = unitful(lambda: numpyro.sample("y", make(), obs=obs), usys)

    def f():
        return log_density(h, (), {}, {})[0]

    got = jax.jit(f)() if jit else f()
    np.testing.assert_allclose(got, bare().log_prob(obs).sum(), rtol=1e-6)
    assert u.dimension_of(h.site_units["y"]) == u.dimension("dimensionless")


def test_bare_next_to_dimensionless_latent_rewrapped(usys):
    # Rule: a site with a dimensionless Quantity parameter is re-wrapped in "",
    # mixed with bare parameters or not.
    def m():
        return numpyro.sample("y", dist.Normal(Q(0.0, ""), 1.0))

    y = handlers.seed(unitful(m, usys), 0)()
    assert isinstance(y, u.AbstractQuantity)
    assert u.dimension_of(y) == u.dimension("dimensionless")
    assert not isinstance(handlers.seed(unitful(m, usys, rewrap=False), 0)(), u.Q)


def test_bare_obs_message_names_usys_unit(usys):
    def m():
        numpyro.sample("v", dist.Normal(Q(0.0, "km/s"), Q(1.0, "km/s")), obs=1.0)

    msg = r"read in kpc / Myr, the unit of usys.*u\.Q\(value, <your unit>\)"
    with pytest.raises(UnitfulError, match=msg):
        handlers.seed(unitful(m, usys), 0)()


def test_mismatch_message_is_readable(usys):
    def m():
        numpyro.sample("v", dist.Normal(Q(0.0, "km/s"), Q(1.0, "kpc")))

    with pytest.raises(UnitfulError, match=r"dimensions: length, speed/velocity\.$"):
        handlers.seed(unitful(m, usys), 0)()


def test_vonmises_bare_obs_message_says_rad():
    dsys = u.unitsystem("kpc", "Myr", "Msun", "deg")

    def m():
        numpyro.sample("a", dist.VonMises(Q(30.0, "deg"), 2.0), obs=0.5)

    with pytest.raises(UnitfulError, match=r"would be read in rad;") as e:
        handlers.seed(unitful(m, dsys), 0)()
    assert "unit of usys" not in str(e.value)


def test_context_manager_use_names_the_fix(usys, x, p):
    def m(x):
        mu = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
        numpyro.sample("x", dist.Normal(mu, Q(0.2, "kpc")), obs=x)

    with (
        pytest.raises(UnitfulError, match=r"context manager.*log_density\(unitful"),
        unitful(m, usys),
    ):
        log_density(m, (x,), {}, p)
    # Called through the handler, the bare-data message has no such hint.
    with pytest.raises(UnitfulError) as e:
        log_density(unitful(m, usys), (x.value,), {}, p)
    assert "context manager" not in str(e.value)


def test_unit_change_message_fits_cond_branches(usys):
    def m(unit):
        numpyro.sample("a", dist.Normal(Q(0.0, unit), Q(1.0, unit)))

    h = handlers.seed(unitful(m, usys), 0)
    h("kpc")
    with pytest.raises(UnitfulError, match="'a': has unit kpc in one place and Myr"):
        h("Myr")


def test_reentry_message_names_both_causes(usys):
    def rec(depth=1):
        numpyro.sample(f"m{depth}", dist.Normal(Q(0.0, "pc"), Q(1.0, "pc")))
        if depth < 2:
            M(depth + 1)

    M = unitful(rec, usys)
    with pytest.raises(
        UnitfulError, match=r"'m2'.*same unitful instance.*handlers\.do"
    ):
        handlers.seed(M, 0)()


def test_count_family_message_says_dimensionless(usys):
    def m():
        numpyro.sample("n", dist.Poisson(Q(3.0, "1/s")))

    with pytest.raises(UnitfulError, match=r"'n'.*counts.*must be dimensionless") as e:
        handlers.seed(unitful(m, usys), 0)()
    assert "units=" not in str(e.value)


def test_no_rule_message_prints_usys_unit(usys):
    def m():
        numpyro.sample("x", dist.LogNormal(Q(0.0, "pc"), Q(1.0, "pc")))

    with pytest.raises(UnitfulError, match=r"units=\{'x': 'kpc'\}"):
        handlers.seed(unitful(m, usys), 0)()


def test_usys_is_checked_at_construction():
    def m():
        return numpyro.sample("m", dist.Normal(Q(0.0, "pc"), Q(3.0, "pc")))

    assert unitful(m, "galactic").usys == u.unitsystems.galactic
    with pytest.raises(TypeError, match=r"unitful needs a unit system.*None"):
        unitful(m, None)
    with pytest.raises(TypeError, match="unitful needs a unit system"):
        unitful(m, {"length": "kpc"})


def test_error_has_public_module():
    assert UnitfulError.__module__ == "unxts.interop.numpyro"


@pytest.mark.parametrize("obs", [[Q(1.0, "kpc"), Q(2.0, "kpc")], {"a": Q(1.0, "kpc")}])
@pytest.mark.parametrize("scale", [Q(1.0, "kpc"), 1.0], ids=["kpc", "unitless"])
def test_observed_container_of_quantities_raises(usys, obs, scale):
    def m(o):
        numpyro.sample("x", dist.Normal(0 * scale, scale), obs=o)

    with pytest.raises(UnitfulError, match="'x': the value is a container"):
        log_density(unitful(m, usys), (obs,), {}, {})


def test_outer_condition_container_of_quantities_raises(usys):
    def m():
        numpyro.sample("x", dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")).expand([2]))

    c = handlers.condition(unitful(m, usys), {"x": [Q(1.0, "pc"), Q(2.0, "pc")]})
    with pytest.raises(UnitfulError, match="'x': the value is a container"):
        log_density(c, (), {}, {})


def test_angle_latent_comes_back_as_angle(usys):
    def m():
        a = numpyro.sample("a", dist.Normal(u.Angle(10.0, "deg"), u.Angle(1.0, "deg")))
        v = numpyro.sample("v", dist.VonMises(u.Angle(90.0, "deg"), 50.0))
        q = numpyro.sample("q", dist.Normal(Q(10.0, "deg"), u.Angle(1.0, "deg")))
        return a, v, q

    a, v, q = handlers.seed(unitful(m, usys), 0)()
    assert type(a) is type(v) is u.Angle
    assert a.unit == v.unit == "rad"
    assert type(q) is u.Q  # mixed types give a plain Quantity
