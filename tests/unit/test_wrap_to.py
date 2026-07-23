"""Tests for ``wrap_to`` -- the half-open [min, max) contract and its export."""

import jax
import pytest

import unxt as u


def test_wrap_to_is_top_level_export():
    """``wrap_to`` is re-exported from ``unxt`` like the other api verbs."""
    assert "wrap_to" in u.__all__
    assert u.wrap_to is u.quantity.wrap_to


def test_wrap_to_normal_case():
    """A value above the range wraps down into it."""
    r = u.wrap_to(u.Angle(370, "deg"), u.Q(0, "deg"), u.Q(360, "deg"))
    assert float(r.value) == 10


@pytest.mark.parametrize("v", [-1e-8, -1e-6, -1e-4])
def test_wrap_to_never_returns_max(v):
    """A tiny negative angle must land in [0, 360), never exactly 360.

    Float rounding used to push ``(-1e-8) % 360`` up to exactly ``360.0``,
    violating the documented half-open contract.
    """
    r = u.Angle(v, "deg").wrap_to(u.Q(0.0, "deg"), u.Q(360.0, "deg"))
    assert 0.0 <= float(r.value) < 360.0


def test_wrap_to_half_open_holds_under_jit():
    """The fold-back is arithmetic, so it survives ``jax.jit``."""

    def wrap(a):
        return a.wrap_to(u.Q(0.0, "deg"), u.Q(360.0, "deg"))

    r = jax.jit(wrap)(u.Angle(-1e-8, "deg"))
    assert 0.0 <= float(r.value) < 360.0
