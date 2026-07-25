"""Shared v1->v2 deprecation shim for module ``__getattr__``.

``unxt``, ``unxt.quantity`` and ``unxt._src.quantity`` all expose the same
deprecated names: ``BareQuantity`` (renamed to the new default ``Quantity``)
and the parametric classes that moved to the ``unxts.parametric`` package.
Routing each module's ``__getattr__`` through :func:`deprecated_getattr` keeps
the moved-name set and the deprecation message in one place.
"""

__all__: tuple[str, ...] = ()

_MOVED_TO_PARAMETRIC = frozenset(
    {"ParametricQuantity", "PQ", "AbstractParametricQuantity"}
)


def deprecated_getattr(name: str, module: str, quantity: type, /) -> object:
    """Resolve a deprecated module attribute, else raise ``AttributeError``.

    Shared ``__getattr__`` body for the quantity deprecation shims.

    Parameters
    ----------
    name
        The attribute being looked up on the module.
    module
        ``__name__`` of the calling module, used in the "no attribute" message.
    quantity
        The class ``BareQuantity`` now resolves to (the new default
        ``Quantity``).

    """
    if name in _MOVED_TO_PARAMETRIC:
        msg = (
            f"`{name}` moved to the `unxts.parametric` package. Install it "
            "(`pip install unxts.parametric`) and use "
            f"`from unxts.parametric import {name}`."
        )
        raise AttributeError(msg)
    if name == "BareQuantity":
        import warnings  # noqa: PLC0415

        warnings.warn(
            "`BareQuantity` has been renamed to `Quantity` and is now the "
            "default quantity class (unxt v2). The parametric class formerly "
            "named `Quantity` is now `ParametricQuantity`. `BareQuantity` "
            "will be removed in a future release.",
            category=DeprecationWarning,
            # 3 frames: user access -> module __getattr__ -> here.
            stacklevel=3,
        )
        return quantity
    msg = f"module {module!r} has no attribute {name!r}"
    raise AttributeError(msg)
