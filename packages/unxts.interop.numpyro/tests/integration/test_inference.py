"""Inference (NUTS, SVI, Predictive) through `unitful`, against known truth."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import quax
from numpyro.contrib.control_flow import scan
from numpyro.handlers import reparam
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Predictive,
    Trace_ELBO,
)
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.infer.reparam import LocScaleReparam
from numpyro.infer.util import log_density, log_likelihood
from numpyro.optim import Adam
from unxts.interop.numpyro import attach_units, unitful

import unxt as u

Q = u.Q
KEY = jax.random.PRNGKey(0)
TOL = 0.05  # kpc


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


def check_truth(samples):
    assert samples["m"].unit == u.unit("kpc")
    m, s = (u.ustrip("kpc", samples[k]) for k in ("m", "s"))
    assert abs(float(jnp.mean(m)) - 1.0) < TOL
    assert abs(float(jnp.mean(s)) - 0.2) < TOL


def run_nuts(m, x, **kw):
    mcmc = MCMC(NUTS(m), num_warmup=300, num_samples=300, progress_bar=False, **kw)
    mcmc.run(KEY, x)
    return mcmc


def test_nuts(usys, x):
    m = unitful(model, usys)
    mcmc = run_nuts(m, x, num_chains=2, chain_method="vectorized")
    samples = mcmc.get_samples()
    assert samples["m"].shape == (600,)
    check_truth(attach_units(samples, m))


@pytest.mark.parametrize("guide", [AutoNormal, AutoDelta, AutoMultivariateNormal])
def test_svi(usys, x, guide):
    m = unitful(model, usys)
    g = guide(m)
    svi = SVI(m, g, Adam(0.05), Trace_ELBO())
    res = svi.run(KEY, 2000, x, progress_bar=False)
    post = Predictive(g, params=res.params, num_samples=200)(jax.random.PRNGKey(1), x)
    check_truth(attach_units(post, m))


def test_predictive_prior_and_posterior(usys, x):
    m = unitful(model, usys)
    prior = Predictive(m, num_samples=50, return_sites=["m", "s", "x"])(KEY)
    prior = attach_units(prior, m)
    assert prior["x"].unit == u.unit("kpc")
    assert prior["x"].shape == (50, 200)

    mcmc = run_nuts(m, x)
    post = Predictive(
        m, posterior_samples=mcmc.get_samples(), return_sites=["m", "s", "x"]
    )(jax.random.PRNGKey(2))
    post = attach_units(post, m)
    assert post["x"].unit == u.unit("kpc")
    check_truth(post)
    assert abs(float(jnp.mean(u.ustrip("kpc", post["x"]))) - 1.0) < TOL


def test_log_likelihood(usys, x, p):
    m = unitful(model, usys)
    p1 = jax.tree.map(lambda v: v[None], p)
    got = log_likelihood(m, p1, x)["x"]
    ref = log_likelihood(model_ref, p1, u.ustrip("kpc", x))["x"]
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_reparam_outside(usys, x):
    m = unitful(model, usys)
    rm = reparam(m, config={"m": LocScaleReparam(0)})
    mcmc = run_nuts(rm, x)
    samples = mcmc.get_samples()
    # the reparam adds "m_decentered"; "m" itself is a deterministic site
    m_s = u.ustrip("kpc", attach_units(samples, m)["m"])
    assert abs(float(jnp.mean(m_s)) - 1.0) < TOL


def test_scan_with_rewrap_false(usys, x):
    xs = x[:20]

    def scan_model(xs):
        m = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
        s = numpyro.sample("s", dist.HalfNormal(Q(1.0, "kpc")))

        # rewrap=False: m and s are bare, so give them their unit back
        mq, sq = Q(m, "kpc"), Q(s, "kpc")

        def step(c, xi):
            numpyro.sample("x", dist.Normal(mq, sq), obs=xi)
            return c, None

        scan(step, 0.0, xs)

    def scan_ref(xs):
        m = numpyro.sample("m", dist.Normal(0.0, 3.0))
        s = numpyro.sample("s", dist.HalfNormal(1.0))

        def step(c, xi):
            numpyro.sample("x", dist.Normal(m, s), obs=xi)
            return c, None

        scan(step, 0.0, xs)

    p = {"m": jnp.asarray(1.0), "s": jnp.asarray(0.2)}
    got = log_density(unitful(scan_model, usys, rewrap=False), (xs,), {}, p)[0]
    ref = log_density(scan_ref, (u.ustrip("kpc", xs),), {}, p)[0]
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_quaxified_jnp_helper(usys, x, p):
    def chi2(x, m, s):  # plain-jnp code, untouched
        return jnp.sum(jnp.square((x - m) / s))

    qchi2 = quax.quaxify(chi2)

    def qmodel(xq):
        m = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
        s = numpyro.sample("s", dist.HalfNormal(Q(1.0, "kpc")))
        c2 = qchi2(xq, m, s)  # dimensionless Quantity
        numpyro.factor(
            "ll", -0.5 * u.ustrip("", c2) - xq.shape[0] * jnp.log(u.ustrip("kpc", s))
        )

    def qref(xq):
        m = numpyro.sample("m", dist.Normal(0.0, 3.0))
        s = numpyro.sample("s", dist.HalfNormal(1.0))
        numpyro.factor("ll", -0.5 * chi2(xq, m, s) - xq.shape[0] * jnp.log(s))

    got = log_density(unitful(qmodel, usys), (x,), {}, p)[0]
    ref = log_density(qref, (u.ustrip("kpc", x),), {}, p)[0]
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_nuts_then_predictive_same_handler(usys, x):
    m = unitful(model, usys)
    mcmc = run_nuts(m, x)
    samples = mcmc.get_samples()
    post = Predictive(m, posterior_samples=samples, return_sites=["x"])(KEY)
    both = attach_units({**samples, **post}, m)
    assert both["m"].unit == both["x"].unit == u.unit("kpc")
    check_truth(both)
