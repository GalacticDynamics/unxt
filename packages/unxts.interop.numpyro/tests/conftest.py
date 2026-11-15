"""Shared fixtures for the unxts.interop.numpyro tests."""

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u


@pytest.fixture
def usys():
    """The unit system every test strips in."""
    return u.unitsystems.galactic


@pytest.fixture
def x():
    """200 draws from Normal(1, 0.2), in kpc."""
    rng = np.random.default_rng(1)
    return u.Q(jnp.asarray(rng.normal(1.0, 0.2, 200)), "kpc")


@pytest.fixture
def p():
    """Latent values for the reference model."""
    return {"m": jnp.asarray(1.0), "s": jnp.asarray(0.2)}
