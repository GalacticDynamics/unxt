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
from jax.errors import TracerArrayConversionError
from packaging.version import Version

import unxt as u

sp = pytest.importorskip("spexial")


def _xfail(raises, issue):
    return pytest.mark.xfail(
        raises=raises, reason=f"https://github.com/GalacticDynamics/unxt/issues/{issue}"
    )


# A traced exponent surfaces as TracerArrayConversionError instead of ValueError.
_POW = _xfail((ValueError, TracerArrayConversionError), 951)
_SHIFT = _xfail(RuntimeError, 952)
_GATHER = _xfail(RuntimeError, 953)
# Raised by whichever runtime type-checker is installed (jaxtyping wraps beartype).
_ARGMAX = _xfail((TypeError, BeartypeCallHintViolation), 956)
# quax < 0.4.4 passed `linear` to `scan_p`, which jax 0.10 no longer accepts.
_SCAN = pytest.mark.xfail(
    Version(version("quax")) < Version("0.4.4"),
    raises=TypeError,
    reason="quax < 0.4.4 scan rule passes `linear` to scan_p",
)

x = jnp.array([0.5, 1.0, 2.0])
z = jnp.array([0.1, 0.5])

# (function, raw dimensionless args); quaxified with every arg as Quantity(_, "").
CASES = [
    pytest.param(sp.k0, (x,), id="k0", marks=_POW),
    pytest.param(sp.k1, (x,), id="k1", marks=_POW),
    pytest.param(sp.k2, (x,), id="k2", marks=_POW),
    pytest.param(sp.k0e, (x,), id="k0e", marks=_POW),
    pytest.param(sp.gamma, (x,), id="gamma", marks=_SHIFT),
    pytest.param(sp.zeta, (jnp.array([2.0, 3.0]),), id="zeta", marks=_GATHER),
    # spence then hits select_n_p (unxt#954) once argmax_p is fixed.
    pytest.param(sp.spence, (x,), id="spence", marks=[_SCAN, _ARGMAX]),
    pytest.param(ft.partial(sp.polylog, 2), (z,), id="polylog", marks=_POW),
    pytest.param(sp.comb, (jnp.array(5.0), jnp.array(2.0)), id="comb"),
    pytest.param(
        sp.incomplete_beta,
        (jnp.array(2.0), jnp.array(3.0), z),
        id="incomplete_beta",
        marks=_SCAN,
    ),
    pytest.param(
        ft.partial(sp.eval_gegenbauer, 3),
        (jnp.array(0.5), z),
        id="eval_gegenbauer",
        marks=_SCAN,
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


# bitcast_convert_type_p keeps the `%` unit on the bits, which the bitwise ops
# then (since #955) refuse with a clear error.
@_xfail(ValueError, 958)
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
