"""Give units back to samples."""

__all__ = ("attach_units",)

from collections.abc import Mapping
from typing import Any

import unxt as u
from .handler import unitful
from .rules import UnitfulError, _is_q


def attach_units(samples: Mapping[str, Any], handler: unitful, /) -> dict[str, Any]:
    """Wrap the samples of each site `handler` recorded in that site's unit.

    Quantities and keys the handler never saw (for example ``_sig_log_det``
    from MCMC) pass through unchanged. Batched values are wrapped as they are.

    Examples
    --------
    >>> import jax
    >>> import numpyro, numpyro.distributions as dist
    >>> from numpyro.infer import Predictive
    >>> import unxt as u
    >>> from unxts.interop.numpyro import attach_units, unitful

    >>> def model():
    ...     numpyro.sample("m", dist.Normal(u.Q(0.0, "pc"), u.Q(3.0, "pc")))

    >>> M = unitful(model, u.unitsystems.galactic)
    >>> s = Predictive(M, num_samples=3)(jax.random.PRNGKey(0))
    >>> attach_units(s, M)["m"].unit
    Unit("kpc")

    """
    units = handler.site_units
    if not units:
        msg = "The handler has no recorded sites: run the model first."
        raise UnitfulError(msg)
    return {
        k: u.Q(v, units[k]) if units.get(k) is not None and not _is_q(v) else v
        for k, v in samples.items()
    }
