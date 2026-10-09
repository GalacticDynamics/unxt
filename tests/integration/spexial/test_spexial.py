"""Quantities through `spexial` special functions via `quax.quaxify`.

`spexial` is not a dependency of `unxt`; these run in the weekly ecosystem job
(the ``test-ecosystem`` dependency group) and skip elsewhere. Known gaps are
strict xfails pinned to the raised exception, so a fix anywhere upstream fails
the job and the marker can be dropped.
"""

import functools as ft
from importlib.metadata import version

import jax
import jax.numpy as jnp
import pytest
import quax
from astropy.units import UnitConversionError
from beartype.roar import BeartypeCallHintViolation
from packaging.version import Version

import unxt as u

sp = pytest.importorskip("spexial")

# quax < 0.4.4 has a `scan` rule that breaks on jax 0.10 and leaks tracers across
# tests, making results order-dependent. The weekly job installs the newest quax.
if Version(version("quax")) < Version("0.4.4"):
    pytest.skip("needs quax>=0.4.4", allow_module_level=True)


def _xfail(raises, issue):
    return pytest.mark.xfail(
        raises=raises, reason=f"https://github.com/GalacticDynamics/unxt/issues/{issue}"
    )


# Raised by whichever runtime type-checker is installed (jaxtyping wraps beartype).
_ARGMAX = _xfail((TypeError, BeartypeCallHintViolation), 956)

x = jnp.array([0.5, 1.0, 2.0])
z = jnp.array([0.1, 0.5])

# (function, raw dimensionless args); quaxified with every arg as Quantity(_, "").
CASES = [
    pytest.param(sp.k0, (x,), id="k0"),
    pytest.param(sp.k1, (x,), id="k1"),
    pytest.param(sp.k2, (x,), id="k2"),
    pytest.param(sp.k0e, (x,), id="k0e"),
    pytest.param(sp.gamma, (x,), id="gamma"),
    pytest.param(sp.zeta, (jnp.array([2.0, 3.0]),), id="zeta"),
    # spence then hits select_n_p (unxt#954) once argmax_p is fixed.
    pytest.param(sp.spence, (x,), id="spence", marks=_ARGMAX),
    pytest.param(ft.partial(sp.polylog, 2), (z,), id="polylog"),
    pytest.param(sp.comb, (jnp.array(5.0), jnp.array(2.0)), id="comb"),
    pytest.param(
        sp.incomplete_beta,
        (jnp.array(2.0), jnp.array(3.0), z),
        id="incomplete_beta",
    ),
    pytest.param(
        ft.partial(sp.eval_gegenbauer, 3),
        (jnp.array(0.5), z),
        id="eval_gegenbauer",
    ),
    pytest.param(
        ft.partial(sp.sph_harm_y_cart, 2, 1),
        (jnp.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]),),
        id="sph_harm_y_cart",
    ),
]


@pytest.mark.parametrize(("fn", "args"), CASES)
def test_dimensionless(fn, args):
    """A dimensionless Quantity in gives the raw result as dimensionless out."""
    got = quax.quaxify(fn)(*(u.Q(a, "") for a in args))

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("")
    assert jnp.allclose(got.value, fn(*args))


# Now runs to completion, but comparisons with a raw operand ignore the `%`
# scale (unxt#965), so it returns wrong values rather than raising.
@_xfail(AssertionError, 965)
def test_scaled_dimensionless():
    """A '%' input is converted to dimensionless before evaluating."""
    got = quax.quaxify(sp.k0)(u.Q(100 * x, "%"))
    assert jnp.allclose(u.ustrip("", got), sp.k0(x))


theta = jnp.array([20.0, 50.0])
phi = jnp.array([10.0, 70.0])


def test_dimensionful_raises():
    """A length where an angle is expected is an error, not a silent strip."""
    with pytest.raises(UnitConversionError):
        quax.quaxify(ft.partial(sp.sph_legendre_p, 2, 1))(u.Q(theta, "m"))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sph_legendre_p_angle(jit):
    """Angles in degrees give the same result as raw radians."""
    fn = quax.quaxify(ft.partial(sp.sph_legendre_p, 2, 1))
    fn = jax.jit(fn) if jit else fn

    got = fn(u.Q(theta, "deg"))

    assert jnp.allclose(u.ustrip("", got), sp.sph_legendre_p(2, 1, jnp.deg2rad(theta)))


# `exp` of an angle is undefined (as in astropy); spexial computes exp(i m phi).
@pytest.mark.xfail(raises=UnitConversionError, reason="exp_p rejects angles")
def test_sph_harm_y_angle():
    """Angles in degrees give the same result as raw radians."""
    got = quax.quaxify(ft.partial(sp.sph_harm_y, 2, 1))(
        u.Q(theta, "deg"), u.Q(phi, "deg")
    )
    want = sp.sph_harm_y(2, 1, jnp.deg2rad(theta), jnp.deg2rad(phi))
    assert jnp.allclose(u.ustrip("", got), want)
