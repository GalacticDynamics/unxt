"""Tests for `unitful` at factor, deterministic and param sites."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.util import log_density
from numpyro.primitives import apply_stack
from unxts.interop.numpyro import UnitfulError, unitful
from unxts.interop.numpyro._src.handler import _restore

import unxt as u

Q = u.Q


def _fac(lf):
    def model():
        numpyro.factor("f", lf)

    return model


def _ld(fn):
    return log_density(fn, (), {}, {})[0]


def test_dimensionless_factor(usys):
    np.testing.assert_allclose(_ld(unitful(_fac(Q(-1.0, "")), usys)), -1.0)


def test_ratio_factor(usys):
    np.testing.assert_allclose(_ld(unitful(_fac(Q(1.5, "m/km")), usys)), 0.0015)


def test_dimensioned_factor_raises(usys):
    m = unitful(_fac(Q(1.0, "kpc")), usys)
    with pytest.raises(UnitfulError, match="'f'"):
        _ld(m)
    with pytest.raises(UnitfulError, match="'f'"):
        jax.jit(lambda: _ld(m))()


def test_bare_factor_passes(usys):
    np.testing.assert_allclose(_ld(unitful(_fac(-2.0), usys)), -2.0)


def test_deterministic_trace_value_is_bare(usys):
    seen = {}

    def model():
        seen["v"] = numpyro.deterministic("d", Q(jnp.asarray(500.0), "pc"))

    h = unitful(model, usys)
    tr = handlers.trace(h).get_trace()
    np.testing.assert_allclose(tr["d"]["value"], 0.5)
    assert not isinstance(tr["d"]["value"], u.AbstractQuantity)
    assert seen["v"].unit == usys["length"]
    assert h.site_units["d"] == usys["length"]


def test_deterministic_rewrap_false(usys):
    def model():
        return numpyro.deterministic("d", Q(500.0, "pc"))

    assert not isinstance(unitful(model, usys, rewrap=False)(), u.AbstractQuantity)


def test_bare_deterministic_records_no_unit(usys):
    def model():
        numpyro.deterministic("d", jnp.asarray(1.0))

    h = unitful(model, usys)
    h()
    assert h.site_units["d"] is None


def test_deterministic_unit_change_raises(usys):
    def model(q):
        numpyro.deterministic("d", q)

    h = unitful(model, usys)
    h(Q(1.0, "kpc"))
    with pytest.raises(UnitfulError, match="'d'"):
        h(Q(1.0, "Myr"))


def test_print_summary_with_deterministic(usys, capsys):
    def model():
        m = numpyro.sample("m", dist.Normal(Q(0.0, "kpc"), Q(3.0, "kpc")))
        numpyro.deterministic("d", m * 2)

    mcmc = MCMC(NUTS(unitful(model, usys)), num_warmup=50, num_samples=50)
    mcmc.run(jax.random.key(0))
    mcmc.print_summary(exclude_deterministic=False)
    assert "d" in capsys.readouterr().out


def test_param_returns_quantity_and_trace_bare(usys):
    def model():
        return numpyro.param("w", Q(500.0, "pc"))

    h = unitful(model, usys)
    tr = handlers.trace(h).get_trace()
    np.testing.assert_allclose(tr["w"]["value"], 0.5)
    assert h().unit == usys["length"]


def test_param_with_constraint(usys):
    def model():
        w = numpyro.param("w", Q(1.0, "kpc"), constraint=constraints.positive)
        numpyro.sample("y", dist.Normal(w, Q(1.0, "kpc")), obs=Q(2.0, "kpc"))

    def guide():
        pass  # empty guide: `w` is a param of the model

    svi = SVI(unitful(model, usys), guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    res = svi.run(jax.random.key(0), 100, progress_bar=False)
    assert np.isfinite(res.params["w"])


def test_factor_in_plate(usys):
    def model():
        with numpyro.plate("n", 3):
            numpyro.factor("f", Q(-1.0, ""))

    def ref():
        with numpyro.plate("n", 3):
            numpyro.factor("f", -1.0)

    np.testing.assert_allclose(_ld(unitful(model, usys)), _ld(ref))


def test_factor_under_inner_mask(usys):
    mask = jnp.array([True, False])
    model = handlers.mask(_fac(Q(jnp.array([-1.0, -2.0]), "")), mask=mask)
    ref = handlers.mask(_fac(jnp.array([-1.0, -2.0])), mask=mask)
    np.testing.assert_allclose(_ld(unitful(model, usys)), _ld(ref))


def test_dimensioned_factor_in_plate_raises(usys):
    def model():
        with numpyro.plate("n", 3):
            numpyro.factor("f", Q(1.0, "kpc"))

    with pytest.raises(UnitfulError, match="'f'"):
        _ld(unitful(model, usys))


def test_callable_param_init_returning_quantity_raises(usys):
    def model():
        numpyro.param("p", lambda _key: Q(1.0, "kpc"))

    with pytest.raises(UnitfulError, match="'p'"):
        handlers.seed(unitful(model, usys), 0)()


@pytest.mark.parametrize(
    ("con", "want"),
    [
        (constraints.interval(Q(0.0, "pc"), Q(2000.0, "pc")), [0.0, 2.0]),
        (constraints.greater_than(Q(100.0, "pc")), [0.1]),
        (constraints.interval(0.0, Q(2000.0, "pc")), [0.0, 2.0]),
    ],
    ids=["interval", "greater_than", "mixed"],
)
def test_param_constraint_quantity_bounds_stripped(usys, con, want):
    def model():
        numpyro.param("w", Q(500.0, "pc"), constraint=con)

    tr = handlers.trace(unitful(model, usys)).get_trace()
    c = tr["w"]["kwargs"]["constraint"]
    assert type(c) is type(con)
    np.testing.assert_allclose(jax.tree.leaves(c), want)


def test_param_constraint_bound_wrong_dimension_raises(usys):
    def model():
        numpyro.param(
            "w", Q(500.0, "pc"), constraint=constraints.greater_than(Q(1.0, "s"))
        )

    with pytest.raises(UnitfulError, match=r"'w'.*constraint bound has dimension time"):
        unitful(model, usys)()


def test_param_constraint_quantity_bound_bare_init_raises(usys):
    def model():
        numpyro.param("w", 0.5, constraint=constraints.greater_than(Q(1.0, "pc")))

    with pytest.raises(UnitfulError, match=r"'w'.*Quantity bounds.*init value is bare"):
        unitful(model, usys)()


def test_callable_param_init_message_gives_fix(usys):
    def model():
        numpyro.param("p", lambda _key: Q(1.0, "kpc"))

    with pytest.raises(UnitfulError, match=r"'p'.*return a bare value.*init value"):
        handlers.seed(unitful(model, usys), 0)()


def test_dimensionless_quantity_scale_stripped(usys):
    def model():
        with handlers.scale(scale=Q(2.0, "")):
            numpyro.sample(
                "y", dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")), obs=Q(1.0, "kpc")
            )

    want = 2 * dist.Normal(0.0, 1.0).log_prob(1.0)
    np.testing.assert_allclose(_ld(unitful(model, usys)), want, rtol=1e-6)


def test_dimensioned_scale_raises(usys):
    def model():
        with handlers.scale(scale=Q(2.0, "kpc")):
            numpyro.sample(
                "y", dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")), obs=Q(1.0, "kpc")
            )

    with pytest.raises(UnitfulError, match="'y': the scale has dimension length"):
        _ld(unitful(model, usys))


def test_dimensionless_scale_outside_unitful(usys):
    def model():
        numpyro.sample(
            "y", dist.Normal(Q(0.0, "kpc"), Q(1.0, "kpc")), obs=Q(1.0, "kpc")
        )

    want = 2 * dist.Normal(0.0, 1.0).log_prob(1.0)
    got = _ld(handlers.scale(unitful(model, usys), scale=Q(2.0, "")))
    np.testing.assert_allclose(got, want, rtol=1e-6)
    with pytest.raises(UnitfulError, match="'y': the scale has dimension length"):
        _ld(handlers.scale(unitful(model, usys), scale=Q(2.0, "kpc")))


@pytest.mark.parametrize(
    "site",
    [
        lambda: numpyro.deterministic("d", {"a": Q(1.0, "pc")}),
        lambda: numpyro.param("d", {"a": Q(1.0, "pc")}),
    ],
    ids=["deterministic", "param"],
)
def test_container_of_quantities_raises(usys, site):
    with pytest.raises(UnitfulError, match="'d': the value is a container"):
        handlers.seed(unitful(site, usys), 0)()


def test_other_control_flow_is_left_alone(usys):
    """A control-flow message other than scan or cond passes through unchanged."""

    def m():
        msg = {"type": "control_flow", "fn": lambda x: x + 1, "args": (1,)}
        return apply_stack({**msg, "kwargs": {}, "value": None})["value"]

    assert unitful(m, usys)() == 2


def test_restore_ignores_non_distributions():
    c = constraints.positive
    _restore(c, c)
    assert c.lower_bound == 0.0


def _lift_model():
    w = numpyro.param("w", Q(1.0, "kpc"), constraint=constraints.positive)
    numpyro.sample("y", dist.Normal(w, Q(1.0, "kpc")), obs=Q(2.0, "kpc"))


def _lift_ref():
    w = numpyro.param("w", 1.0, constraint=constraints.positive)
    numpyro.sample("y", dist.Normal(w, 1.0), obs=2.0)


_LIFT_REF_PRIOR = {"w": dist.HalfNormal(1.0)}


def test_lift_inside_takes_quantity_prior(usys):
    prior = {"w": dist.HalfNormal(Q(1000.0, "pc"))}
    m = unitful(handlers.lift(_lift_model, prior=prior), usys)
    ref = handlers.lift(_lift_ref, prior=_LIFT_REF_PRIOR)
    got, _ = log_density(m, (), {}, {"w": 1.5})
    want, _ = log_density(ref, (), {}, {"w": 1.5})
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_lift_outside_takes_bare_prior(usys):
    m = handlers.lift(unitful(_lift_model, usys), prior=_LIFT_REF_PRIOR)
    ref = handlers.lift(_lift_ref, prior=_LIFT_REF_PRIOR)
    got, _ = log_density(m, (), {}, {"w": 1.5})
    want, _ = log_density(ref, (), {}, {"w": 1.5})
    np.testing.assert_allclose(got, want, rtol=1e-6)
