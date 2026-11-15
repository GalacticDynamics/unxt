"""Patches for two NumPyro bugs that break `unxt.Quantity` distribution parameters.

`install` is idempotent and runs when the package is imported. Each patch is
applied only if a feature test shows the installed NumPyro still has the bug,
and the patched functions behave like the originals for anything that is not a
pytree of several leaves or an array-like with its own ``reshape``.
"""

__all__ = ("NOT_JAX_TRACER_PATCHED", "RESHAPE_PATCHED", "install")

import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions  # noqa: F401  (load the modules whose copies we patch)
import numpyro.distributions.util as ndu
import numpyro.util as nu

import unxt as u

NOT_JAX_TRACER_PATCHED: bool = False
RESHAPE_PATCHED: bool = False

_original_not_jax_tracer = nu.not_jax_tracer
_original_reshape = ndu._reshape  # noqa: SLF001


def _not_jax_tracer(x: Any) -> bool:
    """Return True if no leaf of the pytree `x` is a JAX tracer."""
    return not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(x))


def _reshape(x: Any, shape: Any) -> Any:
    """Reshape `x`, using its own ``reshape`` for array-likes such as Quantity."""
    if isinstance(x, (int, float, np.ndarray, np.generic)):
        return np.reshape(x, shape)
    if hasattr(x, "reshape"):
        return x.reshape(shape)
    return jnp.reshape(x, shape)


# The feature tests only trace, so importing the package does not start the JAX
# backend (which would make `numpyro.set_host_device_count` a no-op).
_ABSTRACT_SCALAR = jax.ShapeDtypeStruct((), np.float32)


def _not_jax_tracer_is_broken() -> bool:
    """Whether the installed `not_jax_tracer` misses tracers inside a Quantity."""
    seen = []
    jax.make_jaxpr(lambda x: seen.append(nu.not_jax_tracer(u.Q(x, "m"))) or x)(
        _ABSTRACT_SCALAR
    )
    return seen[0] is not False


def _reshape_is_broken() -> bool:
    """Whether the installed `_reshape` rejects a Quantity."""
    try:
        jax.make_jaxpr(lambda x: ndu._reshape(u.Q(x, "m"), (1,)).value)(  # noqa: SLF001
            _ABSTRACT_SCALAR
        )
    except TypeError:
        return True
    return False


def _rebind(attr: str, original: Any, patched: Any) -> None:
    """Replace `original` with `patched` in every loaded numpyro module."""
    for name, mod in list(sys.modules.items()):
        if (name == "numpyro" or name.startswith("numpyro.")) and (
            getattr(mod, attr, None) is original
        ):
            setattr(mod, attr, patched)


def install() -> None:
    """Install the patches that the installed NumPyro still needs."""
    # pylint: disable-next=global-statement
    global NOT_JAX_TRACER_PATCHED, RESHAPE_PATCHED  # noqa: PLW0603

    # ponytail: delete when the numpyro floor includes pyro-ppl/numpyro#2428
    if not NOT_JAX_TRACER_PATCHED and _not_jax_tracer_is_broken():
        # numpyro.util first, so modules imported later bind the patched copy.
        nu.not_jax_tracer = _not_jax_tracer
        _rebind("not_jax_tracer", _original_not_jax_tracer, _not_jax_tracer)
        NOT_JAX_TRACER_PATCHED = True

    # ponytail: delete when the numpyro floor includes pyro-ppl/numpyro#2429
    if not RESHAPE_PATCHED and _reshape_is_broken():
        ndu._reshape = _reshape  # noqa: SLF001
        _rebind("_reshape", _original_reshape, _reshape)
        RESHAPE_PATCHED = True
