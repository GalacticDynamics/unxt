"""Tests for `attach_units`."""

import warnings

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import Predictive
from unxts.interop.numpyro import UnitfulError, attach_units, unitful

import unxt as u

Q = u.Q


def model(x=None):
    m = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
    s = numpyro.sample("s", dist.HalfNormal(Q(1.0, "kpc")))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    with numpyro.plate("n", 5):
        numpyro.sample("x", dist.Normal(m, s), obs=x)
    return b


@pytest.fixture
def run(usys):
    M = unitful(model, usys)
    handlers.seed(M, 0)()
    return M


def test_wraps_recorded_sites(run):
    out = attach_units({"m": jnp.ones((4, 2)), "s": jnp.ones(4), "b": jnp.ones(4)}, run)
    assert out["m"].unit == "kpc"
    assert out["m"].shape == (4, 2)
    assert out["s"].unit == "kpc"
    assert not isinstance(out["b"], u.AbstractQuantity)


def test_passes_unknown_and_quantities(run):
    q = Q(jnp.ones(3), "pc")
    samples = {"_sig_log_det": jnp.zeros(3), "m": q}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = attach_units(samples, run)
    assert out["_sig_log_det"] is samples["_sig_log_det"]
    assert out["m"] is q
    assert out is not samples


def test_unrun_handler_raises(usys):
    with pytest.raises(UnitfulError, match="run the model first"):
        attach_units({"m": jnp.ones(2)}, unitful(model, usys))


def test_factor_only_model_after_run(usys):
    M = unitful(lambda: numpyro.factor("f", 1.0), usys)
    handlers.seed(M, 0)()
    samples = {"_x": jnp.ones(2)}
    assert attach_units(samples, M) == samples


def test_predictive_round_trip(usys):
    M = unitful(model, usys)
    prior = Predictive(M, num_samples=4)(jax.random.PRNGKey(0))
    samples = attach_units({k: v for k, v in prior.items() if k != "x"}, M)
    assert samples["m"].unit == "kpc"
    out = Predictive(M, posterior_samples=samples)(jax.random.PRNGKey(1))
    assert out["x"].shape == (4, 5)
