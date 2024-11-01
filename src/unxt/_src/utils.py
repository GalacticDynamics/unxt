"""Utilities.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing_extensions import Self


_singleton_insts: dict[type, object] = {}


class SingletonMixin:
    """Singleton class.

    This class is a mixin that can be used to create singletons.

    Examples
    --------
    >>> class MySingleton(SingletonMixin):
    ...     pass

    >>> a = MySingleton()
    >>> b = MySingleton()
    >>> a is b
    True

    """

    def __new__(cls, /, *_: Any, **__: Any) -> "Self":
        # Check if instance already exists
        if cls in _singleton_insts:
            return cast("Self", _singleton_insts[cls])
        # Create new instance and cache it
        self = object.__new__(cls)
        _singleton_insts[cls] = self
        return self
