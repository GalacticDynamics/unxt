"""Tests for Quantities that handlers outside `unitful` set at a site."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.primitives as prim
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
from numpyro.infer.util import log_density
from unxts.interop.numpyro import UnitfulError, attach_units, unitful

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
    # The init values are converted to usys: 1 kpc and 200 pc land at 1 and 0.2.
    init = init_to_value(values={"m": Q(1.0, "kpc"), "s": Q(200.0, "pc")})
    kernel = NUTS(unitful(model_det, usys), init_strategy=init)
    state = kernel.init(jax.random.PRNGKey(0), 0, None, (x,), {})
    np.testing.assert_allclose(state.z["m"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(jnp.exp(state.z["s"]), 0.2, rtol=1e-6)


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


def _sub():
    return numpyro.sample("m", dist.Normal(Q(1000.0, "pc"), Q(1.0, "pc")))


def test_scope_outside_rewraps(usys):
    h = unitful(_sub, usys)
    m = handlers.seed(handlers.scope(h, "a"), 0)()
    assert m.unit == u.unit("kpc")
    np.testing.assert_allclose(m.value, 1.0, atol=1e-2)
    assert h.site_units["a/m"] == u.unit("kpc")


def test_scope_outside_attach_units(usys):
    h = unitful(_sub, usys)
    s = Predictive(handlers.scope(h, "a"), num_samples=2)(jax.random.PRNGKey(0))
    got = attach_units(s, h)["a/m"]
    assert got.unit == u.unit("kpc")
    np.testing.assert_allclose(got.value, 1.0, atol=1e-2)


def test_scope_inside_and_outside(usys):
    h = unitful(handlers.scope(_sub, "b"), usys)
    tr = handlers.trace(handlers.seed(handlers.scope(h, "a"), 0)).get_trace()
    m = handlers.seed(handlers.scope(h, "a"), 0)()
    assert list(tr) == ["a/b/m"]
    assert m.unit == u.unit("kpc")
    assert attach_units({"a/b/m": tr["a/b/m"]["value"]}, h)["a/b/m"].unit == "kpc"


def test_scope_outside_condition_quantity(usys):
    # The outer stripper finds the unit by message, after the rename.
    h = handlers.condition(
        handlers.scope(unitful(_sub, usys), "a"), {"a/m": Q(500.0, "pc")}
    )
    tr = handlers.trace(handlers.seed(h, 0)).get_trace()
    np.testing.assert_allclose(tr["a/m"]["value"], 0.5)


def test_do_outside_raises(usys):
    def m():
        return numpyro.sample("m", dist.Normal(Q(0.0, "pc"), Q(3000.0, "pc")))

    d = handlers.do(unitful(m, usys), {"m": Q(500.0, "pc")})
    with pytest.raises(UnitfulError, match=r"'m'.*handlers\.do.*inside"):
        handlers.seed(d, 0)()
    # The order the message gives works.
    inner = unitful(handlers.do(m, {"m": Q(500.0, "pc")}), usys)
    np.testing.assert_allclose(u.ustrip("kpc", handlers.seed(inner, 0)()), 0.5)


def test_outer_quantity_under_block(usys, x, p, ref):
    """A `block` around the stack stops messages before the bottom stripper."""
    with handlers.block():
        c = handlers.condition(unitful(model, usys), {"s": Q(200.0, "pc")})
        tr = handlers.trace(handlers.seed(c, 0)).get_trace(x)
        s = handlers.substitute(unitful(model, usys), {"m": Q(500.0, "pc")})
        tr2 = handlers.trace(handlers.seed(s, 0)).get_trace(x)
        params = {"m": Q(1000.0, "pc"), "s": Q(200.0, "pc")}
        got = ld(unitful(model, usys), x, params)
    np.testing.assert_allclose(tr["s"]["value"], 0.2)
    np.testing.assert_allclose(tr2["m"]["value"], 0.5)
    np.testing.assert_allclose(got, ref, rtol=1e-5)


class _SetLate(numpyro.primitives.Messenger):
    """Set a Quantity value after every stripper has post-processed."""

    def postprocess_message(self, msg):
        if msg.get("name") == "m":
            msg["value"] = Q(1.0, "kpc")


def test_quantity_reaching_postprocess_raises(usys):
    with pytest.raises(UnitfulError, match="'m': a Quantity value set outside"):
        handlers.seed(_SetLate(unitful(model, usys)), 0)()
