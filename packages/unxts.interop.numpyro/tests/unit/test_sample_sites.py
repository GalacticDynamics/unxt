"""Tests for `unitful` at sample sites."""

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

    h = unitful(m, usys, units={"r": "pc"})
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
        handlers.seed(unitful(m, usys, units={"r": "pc"}), 0)()


def test_nuts_runs(usys, x):
    mc = MCMC(
        NUTS(unitful(model, usys)), num_warmup=50, num_samples=50, progress_bar=False
    )
    mc.run(jax.random.PRNGKey(0), x)
    m = mc.get_samples()["m"]
    assert not isinstance(m, u.AbstractQuantity)
    assert abs(float(jnp.mean(m)) - 1.0) < 0.1
