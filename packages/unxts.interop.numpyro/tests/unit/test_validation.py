"""`unitful` keeps NumPyro's support checks and other distribution state."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.distribution as npd
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from unxts.interop.numpyro import unitful

import unxt as u

Q = u.Q
_TN = dist.TruncatedNormal
_MIX = dist.Categorical(jnp.array([0.3, 0.7]))

pytestmark = pytest.mark.filterwarnings("ignore:Out-of-support values:UserWarning")


@pytest.fixture(params=[True, False], ids=["validate", "no-validate"])
def validation(request):
    """Run with NumPyro's global validation on, then off, then restore it."""
    old = npd._VALIDATION_ENABLED
    numpyro.enable_validation(request.param)
    yield request.param
    numpyro.enable_validation(old)


def _site(d, obs):
    def model():
        numpyro.sample("x", d, obs=obs)

    return model


def _states(d):
    """Family, `_validate_args` and `reparametrized_params` of `d` and its parts."""
    out = [(type(d).__name__, d._validate_args, list(d.reparametrized_params))]
    leaf = lambda x: x is not d and isinstance(x, dist.Distribution)
    for c in jax.tree.leaves(d, is_leaf=leaf):
        if isinstance(c, dist.Distribution):
            out += _states(c)
    return out


@pytest.mark.parametrize(
    ("d", "obs"),
    [(dist.Uniform(0.1, 2.0), 0.05), (dist.HalfNormal(1.0), -1.0)],
    ids=["Uniform", "HalfNormal"],
)
def test_bare_model_keeps_support_check(usys, validation, d, obs):
    """A bare distribution is passed through as is, so log p matches the plain model."""
    plain = log_density(_site(d, obs), (), {}, {})[0]
    got = log_density(unitful(_site(d, obs), usys), (), {}, {})[0]
    assert jnp.isneginf(plain) == validation
    assert got == plain
    t = handlers.trace(handlers.seed(unitful(_site(d, obs), usys), 0)).get_trace()
    assert t["x"]["fn"] is d


@pytest.mark.parametrize(
    ("d", "ref", "obs"),
    [
        (dist.Uniform(Q(0.0, "kpc"), Q(2.0, "kpc")), dist.Uniform(0.0, 2.0), -0.5),
        (dist.HalfNormal(Q(1.0, "kpc")), dist.HalfNormal(1.0), -1.0),
        (dist.HalfCauchy(Q(1.0, "kpc")), dist.HalfCauchy(1.0), -1.0),
        (
            _TN(Q(1.0, "kpc"), Q(0.5, "kpc"), low=Q(0.5, "kpc"), high=Q(3.0, "kpc")),
            _TN(1.0, 0.5, low=0.5, high=3.0),
            0.2,
        ),
        (
            _TN(Q(1.0, "kpc"), Q(0.5, "kpc"), low=Q(0.5, "kpc")),
            _TN(1.0, 0.5, low=0.5),
            0.2,
        ),
        (
            dist.HalfNormal(Q(jnp.ones(3), "kpc")).to_event(1),
            dist.HalfNormal(jnp.ones(3)).to_event(1),
            -jnp.ones(3),
        ),
        (
            dist.HalfNormal(Q(1.0, "kpc")).expand([3]).to_event(1),
            dist.HalfNormal(1.0).expand([3]).to_event(1),
            -jnp.ones(3),
        ),
        (
            dist.MixtureSameFamily(
                _MIX, dist.HalfNormal(Q(jnp.array([1.0, 2.0]), "kpc"))
            ),
            dist.MixtureSameFamily(_MIX, dist.HalfNormal(jnp.array([1.0, 2.0]))),
            -1.0,
        ),
    ],
    ids=[
        "Uniform",
        "HalfNormal",
        "HalfCauchy",
        "TruncatedNormal",
        "LeftTruncatedNormal",
        "Independent",
        "Expanded",
        "MixtureSameFamily",
    ],
)
def test_quantity_model_keeps_support_check(usys, validation, d, ref, obs):
    """Stripping keeps validation and the state a pytree round trip drops."""
    plain = log_density(_site(ref, obs), (), {}, {})[0]
    got = log_density(unitful(_site(d, Q(obs, "kpc")), usys), (), {}, {})[0]
    assert jnp.isneginf(plain) == validation
    assert jnp.allclose(got, plain)
    t = handlers.trace(handlers.seed(unitful(_site(d, Q(obs, "kpc")), usys), 0))
    assert _states(t.get_trace()["x"]["fn"]) == _states(ref)


def _bounded(x):
    h = numpyro.sample("h", dist.HalfNormal(5.0))
    with numpyro.plate("n", x.shape[0]):
        numpyro.sample("x", dist.Uniform(0.0, h), obs=x)


def _bounded_q(x):
    h = numpyro.sample("h", dist.HalfNormal(Q(5.0, "kpc")))
    with numpyro.plate("n", x.shape[0]):
        numpyro.sample("x", dist.Uniform(0.0, h), obs=x)


_DATA = jnp.array([0.2, 0.5, 0.9, 0.4, 0.7])


def _nuts(model, x):
    mc = MCMC(NUTS(model), num_warmup=100, num_samples=100, progress_bar=False)
    mc.run(jax.random.PRNGKey(0), x)
    return mc.get_samples()["h"]


def test_nuts_on_bare_model_is_unchanged(usys):
    """A bare model under `unitful` samples exactly like the plain model."""
    assert jnp.array_equal(
        _nuts(unitful(_bounded, usys), _DATA), _nuts(_bounded, _DATA)
    )


def test_nuts_respects_data_dependent_support(usys):
    """With x ~ Uniform(0, h), the posterior of h stays above the largest datum."""
    ref = _nuts(_bounded, _DATA)
    got = _nuts(unitful(_bounded_q, usys), Q(_DATA, "kpc"))
    assert float(got.min()) >= float(_DATA.max())
    assert jnp.allclose(got, ref, rtol=1e-5)
