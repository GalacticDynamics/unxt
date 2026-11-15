"""Tests for the NumPyro patches."""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.util as ndu
import numpyro.util
import pytest
import unxts.interop.numpyro  # noqa: F401  (installs the patches)
from unxts.interop.numpyro._src import patches

import unxt as u


def test_normal_quantity_params_build_under_jit():
    out = jax.jit(lambda m: dist.Normal(u.Q(m, "kpc"), u.Q(0.2, "kpc")).loc)(1.0)
    assert out.unit == u.unit("kpc")


def test_invalid_concrete_param_still_raises():
    with pytest.raises(ValueError, match="invalid scale"):
        dist.Normal(u.Q(1.0, "kpc"), u.Q(-1.0, "kpc"))


def test_vector_loc_scalar_scale_builds():
    d = dist.Normal(u.Q(jnp.ones(3), "kpc"), u.Q(0.1, "kpc"))
    assert d.scale.shape == (1,)


def test_every_module_copy_is_patched():
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", "")
        if (name == "numpyro" or name.startswith("numpyro.")) and hasattr(
            mod, "not_jax_tracer"
        ):
            assert mod.not_jax_tracer is numpyro.util.not_jax_tracer, name


def test_every_module_reshape_is_patched():
    for name, mod in list(sys.modules.items()):
        if (name == "numpyro" or name.startswith("numpyro.")) and hasattr(
            mod, "_reshape"
        ):
            assert mod._reshape is ndu._reshape, name


def test_plain_arrays_unchanged():
    assert numpyro.util.not_jax_tracer(jnp.ones(2))
    assert numpyro.util.not_jax_tracer(1.0)
    seen = []
    jax.jit(lambda x: seen.append(numpyro.util.not_jax_tracer(x)) or x)(1.0)
    assert seen == [False]


def test_install_is_idempotent():
    before = numpyro.util.not_jax_tracer
    patches.install()
    assert numpyro.util.not_jax_tracer is before


def test_import_leaves_jax_backend_uninitialized():
    """Importing the package must not stop `set_host_device_count` working."""
    code = (
        "import numpyro, unxts.interop.numpyro\n"
        "from jax._src import xla_bridge as xb\n"
        "assert not xb._backends\n"
        "numpyro.set_host_device_count(4)\n"
        "import jax\n"
        "print(jax.local_device_count())\n"
    )
    out = subprocess.run(  # noqa: S603  (our own fixed code)
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "4"
