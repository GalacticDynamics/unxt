"""Tests for Quantities that handlers outside `unitful` set at a site."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.primitives as prim
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_value
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


@pytest.fixture
def ref(x, p):
    return ld(model_ref, u.ustrip("kpc", x), p)


def test_condition_outside(usys, x, p, ref):
    m = handlers.condition(unitful(model, usys), {"m": Q(1000.0, "pc")})
    np.testing.assert_allclose(ld(m, x, {"s": p["s"]}), ref, rtol=1e-5)


def test_log_density_params_quantities(usys, x, ref):
    params = {"m": Q(1000.0, "pc"), "s": Q(200.0, "pc")}
    np.testing.assert_allclose(ld(unitful(model, usys), x, params), ref, rtol=1e-5)


def model_det(x=None):
    model(x)
    numpyro.deterministic("d", jnp.zeros(()))


def _run_init_to_value(m, x):
    init = init_to_value(values={"m": Q(1.0, "kpc"), "s": Q(200.0, "pc")})
    mcmc = MCMC(
        NUTS(m, init_strategy=init), num_warmup=20, num_samples=20, progress_bar=False
    )
    mcmc.run(jax.random.PRNGKey(0), x)
    return mcmc.get_samples()


def test_init_to_value_quantities(usys, x):
    # A deterministic site makes NumPyro replay the model to transform the
    # init values, so they pass through the handlers.
    samples = _run_init_to_value(unitful(model_det, usys), x)
    assert samples["m"].shape == (20,)


@pytest.mark.xfail(
    reason="With no deterministic site NumPyro transforms init values outside "
    "any handler, so a Quantity never reaches unitful.",
    raises=TypeError,
)
def test_init_to_value_quantities_without_replay(usys, x):
    _run_init_to_value(unitful(model, usys), x)


def test_outer_wrong_dimension_raises(usys, x, p):
    m = handlers.condition(unitful(model, usys), {"m": Q(1.0, "s")})
    with pytest.raises(UnitfulError, match="'m'"):
        ld(m, x, {"s": p["s"]})


def test_outer_quantity_at_bare_site_raises(usys):
    def bare():
        numpyro.sample("b", dist.Normal(0.0, 1.0))

    m = handlers.condition(unitful(bare, usys), {"b": Q(1.0, "kpc")})
    with pytest.raises(UnitfulError, match="'b'"):
        handlers.seed(m, 0)()


def test_outer_bare_value_untouched(usys, x, p, ref):
    np.testing.assert_allclose(ld(unitful(model, usys), x, p), ref, rtol=1e-5)


def test_substitute_deterministic_and_param(usys):
    def mdl():
        w = numpyro.param("w", Q(1.0, "kpc"))
        return w, numpyro.deterministic("d", 2 * w)

    sub = {"w": Q(500.0, "pc"), "d": Q(3000.0, "pc")}
    tr = handlers.trace(handlers.substitute(unitful(mdl, usys), sub)).get_trace()
    np.testing.assert_allclose(tr["w"]["value"], 0.5)
    np.testing.assert_allclose(tr["d"]["value"], 3.0)


def test_nested_unitful(usys, x, p, ref):
    m = handlers.condition(unitful(unitful(model, usys), usys), {"m": Q(1000.0, "pc")})
    np.testing.assert_allclose(ld(m, x, {"s": p["s"]}), ref, rtol=1e-5)


def test_stack_restored_after_exception(usys):
    depth = []

    def bad():
        numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")))
        depth.append(len(prim._PYRO_STACK))
        raise RuntimeError

    before = list(prim._PYRO_STACK)
    with handlers.trace(), pytest.raises(RuntimeError):
        handlers.seed(unitful(bad, usys), 0)()
    assert depth == [len(before) + 4]  # trace, seed, unitful and its stripper
    assert before == prim._PYRO_STACK
