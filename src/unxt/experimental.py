"""Experimental features.

.. warning::

    These features may be removed or changed in the future without notice.

Unit-aware wrappers for JAX's automatic-differentiation functions (`grad`,
`jacfwd`, `hessian`) and a unit-checked `where`. On some occasions JAX's autodiff
does not propagate units correctly; these wrappers strip and re-apply the units
around the transformed function. See :mod:`unxt._src.experimental` for details.

>>> from unxt import experimental

"""
# NB: this module is intentionally *not* wrapped in ``install_import_hook``
# (unlike ``unxt.dims`` etc.). Its functions raise their own domain-specific
# errors -- e.g. ``where`` tells you to wrap a raw array as a Quantity -- which
# beartype would pre-empt with a generic type-violation. Keep it unwrapped so
# those messages survive.

__all__ = ("grad", "hessian", "jacfwd", "where")

from ._src.experimental import grad, hessian, jacfwd, where
