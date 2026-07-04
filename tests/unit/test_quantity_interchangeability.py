"""`Quantity` must be usable everywhere `ParametricQuantity` is.

The v2 rename makes the non-parametric class the default. These tests run
the core API over both classes to guarantee interchangeability, and pin
the two intentional differences (no runtime dimension checking, no
dimension-specific dispatch).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from plum import convert

import unxt as u
from unxt.quantity import AbstractQuantity, ParametricQuantity, Quantity


@pytest.fixture(params=[Quantity, ParametricQuantity], ids=lambda c: c.__name__)
def Qcls(request):
    """Both public quantity classes."""
    return request.param


# ---------------------------------------------------------------- construction


def test_construct_scalar(Qcls):
    q = Qcls(1.5, "m")
    assert isinstance(q, AbstractQuantity)
    assert q.unit == u.unit("m")


def test_construct_array(Qcls):
    q = Qcls(jnp.asarray([1.0, 2.0]), "km")
    assert q.shape == (2,)


def test_construct_via_subscript(Qcls):
    # No-op for Quantity, checked for ParametricQuantity — both must accept.
    q = Qcls["length"](2.0, "m")
    assert q.unit == u.unit("m")


def test_from_(Qcls):
    q = Qcls.from_(jnp.asarray([1.0]), "m")
    assert isinstance(q, Qcls)


# ---------------------------------------------------------------- arithmetic


def test_arithmetic_same_class(Qcls):
    a, b = Qcls(2.0, "m"), Qcls(3.0, "m")
    assert (a + b).value == 5.0
    assert (a - b).value == -1.0
    assert (a * b).unit == u.unit("m2")
    assert (a / b).unit == u.unit("")
    assert (a**2).unit == u.unit("m2")
    assert bool(a < b)


def test_arithmetic_with_bare_array_dimensionless(Qcls):
    a = Qcls(2.0, "")
    assert (a + jnp.asarray(1.0)).value == 3.0


def test_pow_with_dimensionless_quantity_exponent(Qcls):
    """A dimensionless Quantity/ParametricQuantity used AS an exponent.

    `pow_p` with a Quantity exponent must accept a bare dimensionless
    Quantity exponent as well as a ParametricQuantity one.
    """
    base = Qcls(3.0, "m")
    exponent = Qcls(2.0, "")
    got = base**exponent
    assert got.value == 9.0
    assert got.unit == u.unit("m2")


# ------------------------------------------------------------------ unit API


def test_uconvert(Qcls):
    q = u.uconvert("km", Qcls(1000.0, "m"))
    assert q.value == 1.0


def test_ustrip(Qcls):
    assert u.ustrip("m", Qcls(1.0, "m")) == 1.0


def test_is_unit_convertible(Qcls):
    assert u.is_unit_convertible("km", Qcls(1.0, "m"))


def test_dimension_of_instance(Qcls):
    assert u.dimension_of(Qcls(1.0, "m")) == u.dimension("length")


# ------------------------------------------------------------ JAX transforms


def test_jit(Qcls):
    @jax.jit
    def f(x):
        return x * 2

    got = f(Qcls(3.0, "m"))
    assert got.value == 6.0


def test_grad(Qcls):
    from unxt import experimental

    def f(x):
        return x**2

    got = experimental.grad(f, units=("m",))(Qcls(3.0, "m"))
    assert got.value == 6.0


def test_vmap(Qcls):
    got = jax.vmap(lambda x: x + x)(Qcls(jnp.asarray([1.0, 2.0]), "s"))
    assert jnp.array_equal(got.value, jnp.asarray([2.0, 4.0]))


def test_tree_roundtrip(Qcls):
    q = Qcls(jnp.asarray([1.0, 2.0]), "m")
    leaves, treedef = jax.tree.flatten(q)
    q2 = jax.tree.unflatten(treedef, leaves)
    assert isinstance(q2, type(q))
    assert q2.unit == q.unit


def test_eqx_module_field(Qcls):
    class M(eqx.Module):
        x: AbstractQuantity

    m = M(x=Qcls(1.0, "m"))
    assert m.x.value == 1.0


# ------------------------------------------------------------------ promotion


def test_promotion_with_static_quantity(Qcls):
    s = u.StaticQuantity(1.0, "m")
    got = s + Qcls(1.0, "m")
    assert isinstance(got, Qcls)


def test_mixing_bare_and_parametric_promotes_to_parametric():
    got = Quantity(1.0, "m") + ParametricQuantity(1.0, "m")
    assert isinstance(got, ParametricQuantity)


# ----------------------------------------------------------------- conversion


def test_convert_between_classes(Qcls):
    q = convert(Qcls(1.0, "m"), Quantity)
    assert isinstance(q, Quantity)
    pq = convert(Qcls(1.0, "m"), ParametricQuantity)
    assert isinstance(pq, ParametricQuantity)


def test_convert_to_astropy(Qcls):
    apy = pytest.importorskip("astropy.units")
    got = convert(Qcls(1.0, "m"), apy.Quantity)
    assert got.unit == apy.m


def test_astropy_compat_methods(Qcls):
    pytest.importorskip("astropy.units")
    q = Qcls(1000.0, "m")
    assert q.to("km").value == 1.0
    assert q.to_value("km") == 1.0


# ------------------------------------------- the two intentional differences


def test_only_parametric_checks_dimensions():
    """Documented v2 difference #1: no runtime dimension checking on Quantity."""
    with pytest.raises(ValueError, match="[Pp]hysical type"):
        ParametricQuantity["time"](1.0, "m")
    # The same expression on the default Quantity is a silent no-op.
    q = Quantity["time"](1.0, "m")
    assert q.unit == u.unit("m")


def test_only_parametric_supports_dimension_dispatch():
    """Documented v2 difference #2: dimension-specific types for dispatch."""
    assert ParametricQuantity["length"] is not ParametricQuantity
    assert Quantity["length"] is Quantity
