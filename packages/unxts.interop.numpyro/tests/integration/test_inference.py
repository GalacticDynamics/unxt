"""Inference (NUTS, SVI, Predictive) through `unitful`, against known truth."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import quax
from numpyro import handlers
from numpyro.contrib.control_flow import cond, scan
from numpyro.handlers import reparam
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    DiscreteHMCGibbs,
    Predictive,
    Trace_ELBO,
)
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from numpyro.infer.reparam import LocScaleReparam
from numpyro.infer.util import log_density, log_likelihood
from numpyro.optim import Adam
from unxts.interop.numpyro import UnitfulError, attach_units, unitful

import unxt as u

Q = u.Q
KEY = jax.random.PRNGKey(0)
TOL = 0.05  # kpc
# A second unit system, so a bug that strips each parameter in its own unit
# instead of `usys` shows up as a factor of 1000.
USYSTEMS = pytest.mark.parametrize(
    "usys",
    [u.unitsystems.galactic, u.unitsystem("pc", "Myr", "Msun", "rad")],
    ids=["galactic", "pc"],
)


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


def check_truth(samples, unit="kpc"):
    assert samples["m"].unit == samples["s"].unit == u.unit(unit)
    m, s = (u.ustrip("kpc", samples[k]) for k in ("m", "s"))
    assert abs(float(jnp.mean(m)) - 1.0) < TOL
    assert abs(float(jnp.mean(s)) - 0.2) < TOL


def run_nuts(m, x, **kw):
    mcmc = MCMC(NUTS(m), num_warmup=300, num_samples=300, progress_bar=False, **kw)
    mcmc.run(KEY, x)
    return mcmc


@USYSTEMS
def test_nuts(usys, x):
    m = unitful(model, usys)
    mcmc = run_nuts(m, x, num_chains=2, chain_method="vectorized")
    samples = mcmc.get_samples()
    assert samples["m"].shape == (600,)
    check_truth(attach_units(samples, m), usys["length"])


def test_two_chains_closing_over_data(usys, x):
    # The data is not an MCMC argument, so sequential chains work.
    m = unitful(functools.partial(model, x), usys)
    mcmc = MCMC(
        NUTS(m),
        num_warmup=300,
        num_samples=300,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(KEY)
    assert mcmc.get_samples()["m"].shape == (600,)
    check_truth(attach_units(mcmc.get_samples(), m))


@pytest.mark.xfail(
    raises=OverflowError,
    # Not strict: the failure needs the argument's `id()` to overflow int32,
    # which depends on where the allocator puts it (always on macOS, not always
    # on Linux CI).
    strict=False,
    reason="NumPyro's cache key for sequential chains fails on a Quantity "
    "argument (pyro-ppl/numpyro#2434, fixed by #2436).",
)
def test_two_sequential_chains_quantity_argument(usys, x):
    run_nuts(unitful(model, usys), x, num_chains=2, chain_method="sequential")


@pytest.mark.parametrize(
    "guide",
    [AutoNormal, AutoDelta, AutoMultivariateNormal, AutoLowRankMultivariateNormal],
)
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


@USYSTEMS
def test_log_likelihood(usys, x, p):
    # the latents and the reference data are in the usys unit of length
    unit = usys["length"]
    p1 = jax.tree.map(lambda v: u.ustrip(unit, Q(v[None], "kpc")), p)
    got = log_likelihood(unitful(model, usys), p1, x)["x"]
    ref = log_likelihood(model_ref, p1, u.ustrip(unit, x))["x"]
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


def _walk(rewrap):
    """A random walk whose latent ``z`` is sampled inside the scan body."""

    def walk(ys):
        s = numpyro.sample("s", dist.HalfNormal(Q(1000.0, "pc")))
        if rewrap:  # the body gets z as a Quantity in the usys unit

            def step(prev, y):
                z = numpyro.sample("z", dist.Normal(prev, s))
                numpyro.sample("y", dist.Normal(z, Q(0.1, "kpc")), obs=y)
                return z, None

            scan(step, Q(0.0, "kpc"), ys)
        else:  # the body gets z as a bare number in kpc

            def step(prev, y):
                z = numpyro.sample("z", dist.Normal(Q(prev, "kpc"), Q(s, "kpc")))
                numpyro.sample("y", dist.Normal(Q(z, "kpc"), Q(0.1, "kpc")), obs=y)
                return z, None

            scan(step, 0.0, ys)

    return walk


def walk_ref(ys):
    s = numpyro.sample("s", dist.HalfNormal(1.0))

    def step(prev, y):
        z = numpyro.sample("z", dist.Normal(prev, s))
        numpyro.sample("y", dist.Normal(z, 0.1), obs=y)
        return z, None

    scan(step, 0.0, ys)


@pytest.mark.parametrize("rewrap", [True, False], ids=["rewrap", "bare"])
def test_scan_latent_in_body(rewrap):
    usys = u.unitsystems.galactic
    ys = Q(jnp.array([100.0, 300.0, 200.0, 500.0]), "pc")
    yk = u.ustrip("kpc", ys)
    M = unitful(_walk(rewrap), usys, rewrap=rewrap)
    p = {"s": jnp.asarray(0.5), "z": jnp.array([0.1, 0.2, 0.3, 0.4])}

    def f(p):
        return log_density(M, (ys,), {}, p)[0]

    def g(p):
        return log_density(walk_ref, (yk,), {}, p)[0]

    np.testing.assert_allclose(f(p), g(p), rtol=1e-5)
    np.testing.assert_allclose(jax.jit(f)(p), g(p), rtol=1e-5)
    np.testing.assert_allclose(jax.grad(f)(p)["z"], jax.grad(g)(p)["z"], rtol=1e-5)

    mcmc = MCMC(NUTS(M), num_warmup=200, num_samples=200, progress_bar=False)
    mcmc.run(KEY, ys)
    z = attach_units(mcmc.get_samples(), M)["z"]
    assert z.unit == "kpc"
    ref = MCMC(NUTS(walk_ref), num_warmup=200, num_samples=200, progress_bar=False)
    ref.run(KEY, yk)
    zr = ref.get_samples()["z"]
    np.testing.assert_allclose(jnp.mean(z.value, 0), jnp.mean(zr, 0), atol=0.05)

    def svi_loss(m, y):
        svi = SVI(m, AutoNormal(m), Adam(0.05), Trace_ELBO())
        return svi.run(KEY, 300, y, progress_bar=False).losses[-1]

    np.testing.assert_allclose(svi_loss(M, ys), svi_loss(walk_ref, yk), rtol=1e-4)

    # Prior draws match the reference too, so the body sampled in kpc.
    got = handlers.trace(handlers.seed(M, 0)).get_trace(ys)["z"]["value"]
    want = handlers.trace(handlers.seed(walk_ref, 0)).get_trace(yk)["z"]["value"]
    np.testing.assert_allclose(got, want, rtol=1e-5)


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


def test_scan_body_checks_and_units():
    usys = u.unitsystems.galactic

    def bare_obs(ys):
        def step(c, y):
            numpyro.sample("y", dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")), obs=y)
            return c, None

        scan(step, 0.0, ys)

    with pytest.raises(UnitfulError, match="'y': value is bare"):
        handlers.seed(unitful(bare_obs, usys), 0)(jnp.ones(3))

    def lognormal(n):
        def step(c, _):
            numpyro.sample("r", dist.LogNormal(0.0, 1.0))
            return c, None

        scan(step, 0.0, None, length=n)

    M = unitful(lognormal, usys, units={"r": "length"})
    tr = handlers.trace(handlers.seed(M, 0)).get_trace(3)
    assert tr["r"]["value"].shape == (3,)
    assert M.site_units["r"] == "kpc"


@pytest.mark.parametrize("guide", [AutoNormal, AutoDelta])
def test_autoguide_outer_condition_quantity(usys, x, guide):
    """Autoguides trace the model under `handlers.block`."""

    def loss(m, data):
        svi = SVI(m, guide(m), Adam(0.01), Trace_ELBO())
        return svi.run(KEY, 50, data, progress_bar=False).losses[-1]

    got = loss(handlers.condition(unitful(model, usys), {"s": Q(200.0, "pc")}), x)
    want = loss(handlers.condition(model_ref, {"s": 0.2}), u.ustrip("kpc", x))
    np.testing.assert_allclose(got, want, rtol=1e-4)


def _pick(near):
    def close(_):
        return numpyro.sample("z", dist.Normal(Q(1000.0, "pc"), Q(100.0, "pc")))

    def far(_):
        return numpyro.sample("z", dist.Normal(Q(2.0, "kpc"), Q(0.1, "kpc")))

    return cond(near, close, far, None)


def _pick_ref(near):
    def close(_):
        return numpyro.sample("z", dist.Normal(1.0, 0.1))

    def far(_):
        return numpyro.sample("z", dist.Normal(2.0, 0.1))

    return cond(near, close, far, None)


@USYSTEMS
@pytest.mark.parametrize("near", [True, False], ids=["true", "false"])
def test_cond_latent_in_branch(usys, near):
    M = unitful(_pick, usys)
    unit = usys["length"]
    z = {"z": jnp.asarray(u.ustrip(unit, Q(1.05, "kpc")))}
    got = log_density(M, (near,), {}, z)[0]
    want = log_density(_pick_ref, (near,), {}, {"z": 1.05})[0]
    # The log density is per usys unit, so it shifts by log(kpc / unit).
    shift = np.log(float(u.ustrip(unit, Q(1.0, "kpc"))))
    np.testing.assert_allclose(got, want - shift, rtol=1e-5)

    drawn = handlers.seed(M, 0)(near)
    assert drawn.unit == unit
    ref = handlers.seed(_pick_ref, 0)(near)
    np.testing.assert_allclose(u.ustrip("kpc", drawn), ref, rtol=1e-5)
    assert M.site_units["z"] == unit

    mcmc = MCMC(NUTS(M), num_warmup=200, num_samples=200, progress_bar=False)
    mcmc.run(KEY, near)
    zs = attach_units(mcmc.get_samples(), M)["z"]
    rm = MCMC(NUTS(_pick_ref), num_warmup=200, num_samples=200, progress_bar=False)
    rm.run(KEY, near)
    np.testing.assert_allclose(
        jnp.mean(u.ustrip("kpc", zs)), jnp.mean(rm.get_samples()["z"]), atol=0.02
    )


def _guide(x=None):
    loc = numpyro.param("loc", Q(0.5, "kpc"))
    sc = numpyro.param("sc", Q(0.1, "kpc"), constraint=dist.constraints.positive)
    numpyro.sample("m", dist.Normal(loc, sc))
    numpyro.sample("s", dist.Delta(numpyro.param("s_map", Q(0.3, "kpc"))))


def _guide_ref(x=None):
    loc = numpyro.param("loc", 0.5)
    sc = numpyro.param("sc", 0.1, constraint=dist.constraints.positive)
    numpyro.sample("m", dist.Normal(loc, sc))
    numpyro.sample("s", dist.Delta(numpyro.param("s_map", 0.3)))


def test_custom_guide_matching_usys_trains_to_reference(x):
    M = unitful(model, u.unitsystems.galactic)
    G = unitful(_guide, M.usys)
    res = SVI(M, G, Adam(0.05), Trace_ELBO()).run(KEY, 300, x, progress_bar=False)
    ref = SVI(model_ref, _guide_ref, Adam(0.05), Trace_ELBO()).run(
        KEY, 300, x.value, progress_bar=False
    )
    np.testing.assert_allclose(res.losses, ref.losses, rtol=1e-4)
    loc = attach_units(res.params, G)["loc"]
    assert loc.unit == "kpc"
    np.testing.assert_allclose(loc.value, ref.params["loc"], rtol=1e-4)


def test_custom_guide_other_usys_raises(x):
    M = unitful(model, u.unitsystems.galactic)
    G = unitful(_guide, u.unitsystem("pc", "Myr", "Msun", "rad"))
    svi = SVI(M, G, Adam(0.05), Trace_ELBO())
    with pytest.raises(UnitfulError, match=r"'m'.*unitsystem\(pc.*unitsystem\(kpc"):
        svi.run(KEY, 10, x, progress_bar=False)


def test_autoguide_has_no_usys_tag(x):
    M = unitful(model, u.unitsystems.galactic)
    svi = SVI(M, AutoNormal(M), Adam(0.05), Trace_ELBO())
    assert np.isfinite(svi.run(KEY, 10, x, progress_bar=False).losses[-1])


def test_scan_bare_start_with_rewrap_names_the_carry():
    """A bare start value with rewrap=True says the carry needs units."""

    def walk(ys):
        s = numpyro.sample("s", dist.HalfNormal(Q(1.0, "kpc")))

        def step(prev, y):
            z = numpyro.sample("z", dist.Normal(prev, s))
            numpyro.sample("y", dist.Normal(z, Q(0.1, "kpc")), obs=y)
            return z, None

        scan(step, 0.0, ys)

    ys = Q(jnp.array([100.0, 300.0, 200.0]), "pc")
    p = {"s": jnp.asarray(0.5), "z": jnp.array([0.1, 0.2, 0.3])}
    M = unitful(walk, u.unitsystems.galactic)
    with pytest.raises(
        UnitfulError,
        # The body hint follows directly, with no concrete-zero hint before it.
        match=r"'z'.*give it units\. Inside a scan or cond body.*rewrap=False",
    ):
        log_density(M, (ys,), {}, p)


def test_cond_bare_operand_names_the_operand():
    """A bare cond operand gets the operand hint, not only the scan one."""

    def pick(flag):
        def branch(x0):
            return numpyro.sample("d", dist.Normal(x0, Q(1.0, "kpc")))

        return cond(flag, branch, branch, jnp.asarray(0.5))

    M = unitful(pick, u.unitsystems.galactic)
    with pytest.raises(UnitfulError, match=r"'d'.*the cond's operand"):
        log_density(M, (True,), {}, {"d": jnp.asarray(0.1)})


def _guide_myr(x=None):
    loc = numpyro.param("loc", Q(1.0, "Myr"))
    numpyro.sample("m", dist.Normal(loc, Q(0.1, "Myr")))
    numpyro.sample("s", dist.Delta(numpyro.param("s_map", Q(0.3, "kpc"))))


def test_custom_guide_other_dimension_raises(x):
    M = unitful(model, u.unitsystems.galactic)
    svi = SVI(M, unitful(_guide_myr, M.usys), Adam(0.05), Trace_ELBO())
    with pytest.raises(UnitfulError, match=r"'m'.*is in Myr.*in kpc"):
        svi.run(KEY, 10, x, progress_bar=False)


def test_custom_guide_degrees_into_von_mises_raises():
    usys = u.unitsystem("kpc", "Myr", "Msun", "deg")

    def vm_model():
        numpyro.sample("phi", dist.VonMises(Q(90.0, "deg"), 50.0))

    def guide():
        numpyro.sample("phi", dist.Normal(Q(90.0, "deg"), Q(1.0, "deg")))

    gt = handlers.trace(handlers.seed(unitful(guide, usys), 0)).get_trace()
    m = handlers.replay(handlers.seed(unitful(vm_model, usys), 1), gt)
    with pytest.raises(UnitfulError, match=r"'phi'.*is in deg.*in rad.*VonMises"):
        m()


def test_custom_guide_other_unit_same_usys_trains_to_reference(x):
    # A guide written in pc is stripped in kpc like the model, so it matches.
    def guide_pc(x=None):
        loc = numpyro.param("loc", Q(500.0, "pc"))
        sc = numpyro.param("sc", Q(100.0, "pc"), constraint=dist.constraints.positive)
        numpyro.sample("m", dist.Normal(loc, sc))
        numpyro.sample("s", dist.Delta(numpyro.param("s_map", Q(300.0, "pc"))))

    M = unitful(model, u.unitsystems.galactic)
    res = SVI(M, unitful(guide_pc, M.usys), Adam(0.05), Trace_ELBO()).run(
        KEY, 100, x, progress_bar=False
    )
    ref = SVI(model_ref, _guide_ref, Adam(0.05), Trace_ELBO()).run(
        KEY, 100, x.value, progress_bar=False
    )
    np.testing.assert_allclose(res.losses, ref.losses, rtol=1e-4)


def test_predictive_parallel(x):
    M = unitful(model, u.unitsystems.galactic)
    post = {"m": jnp.full(4, 1.0), "s": jnp.full(4, 0.2)}
    got = Predictive(M, posterior_samples=post, parallel=True)(KEY)
    want = Predictive(model_ref, posterior_samples=post, parallel=True)(KEY)
    out = attach_units(got, M)
    assert out["x"].unit == u.unit("kpc")
    np.testing.assert_allclose(out["x"].value, want["x"], rtol=1e-6)


def _mix(x=None):
    c = numpyro.sample("c", dist.Bernoulli(0.5))
    m = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
    with numpyro.plate("n", 200):
        numpyro.sample("x", dist.Normal(m + c * Q(0.0, "kpc"), Q(0.2, "kpc")), obs=x)


def test_discrete_hmc_gibbs(x):
    M = unitful(_mix, u.unitsystems.galactic)
    mcmc = MCMC(
        DiscreteHMCGibbs(NUTS(M)), num_warmup=100, num_samples=100, progress_bar=False
    )
    mcmc.run(KEY, x)
    m = attach_units(mcmc.get_samples(), M)["m"]
    assert m.unit == u.unit("kpc")
    assert abs(float(jnp.mean(m.value)) - 1.0) < TOL
