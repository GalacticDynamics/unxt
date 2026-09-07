# Type stub for `unxts.parametric`.
#
# Exists for mypy alone -- see `unxt/quantity.pyi` in core `unxt` for the
# fuller rationale (same project, same constraint). `AbstractParametricQuantity`
# and `ParametricQuantity` are built with `plum.parametric`, whose real
# `__class_getitem__` mypy cannot see without a stub, and whose `value`/`unit`
# fields are `equinox.field(converter=...)`, whose externally-accepted type
# mypy cannot infer from the converter either.
#
# A `.pyi` fully shadows its `.py` module for every type checker, so this
# file must stay a complete, accurate mirror of `unxts.parametric`'s public
# surface -- an omission here does not fall back to the real source, it makes
# that name invisible to mypy/pyright/ty alike.
#
# Unlike core `unxt.Quantity` (a documented no-op `__class_getitem__`),
# `ParametricQuantity["length"]` is a *real*, runtime-checked dimension
# parameter -- `PQ["length"](1, "s")` raises `ValueError` at construction.
# This stub still does not declare the classes `Generic`: even with a
# `Generic[_T]`, mypy parses a bare string subscript like `PQ["length"]` as a
# forward-reference type expression (`Name "length" is not defined`) rather
# than a `Literal`, in *every* position -- annotation, bare expression, or (as
# in the class docstring's own `PQ["length"](1, "m")`) subscript-then-call --
# regardless of the TypeVar's bound. There is no honest `Generic` declaration
# that makes that specific, documented usage type-check under mypy; declaring
# one anyway would only add the same regression core `unxt.Quantity` hit
# (unparametrized construction inferring `[Never]` instead of the class
# itself) for a benefit neither the docstring's exact pattern nor mypy's
# forward-ref parsing lets it collect. pyright/ty already read the real
# `__class_getitem__` and need no stub for this at all.

from collections.abc import Callable
from typing import Any, ClassVar, Self, TypeAlias, final

from jaxtyping import Array, ArrayLike, Bool

from unxts.parametric.config import ParametricConfig

from unxt.quantity import AbstractQuantity
from unxt.units import AbstractUnit

#: See `unxt.quantity`'s `_ValueLike`: `ArrayLike` (jaxtyping) alone excludes
#: plain `list`/`tuple`, which `convert_to_quantity_value` accepts directly.
_ValueLike: TypeAlias = ArrayLike | list[Any] | tuple[Any, ...]

config: ParametricConfig

class AbstractParametricQuantity(AbstractQuantity):
    # ---------------------------------------------------------------
    # `plum.parametric` protocol (invoked by `plum`, not typically called
    # directly, but part of the class's real callable surface)
    @classmethod
    def __init_type_parameter__(cls, dims: Any, /) -> tuple[Any]: ...
    @classmethod
    def __infer_type_parameter__(
        cls, value: Any, unit: Any, **kwargs: Any
    ) -> tuple[Any]: ...
    @classmethod
    def __le_type_parameter__(cls, left: tuple[Any], right: tuple[Any]) -> bool: ...
    def __reduce__(
        self,
    ) -> tuple[Callable[..., AbstractParametricQuantity], tuple[Any, ...]]: ...

@final
class ParametricQuantity(AbstractParametricQuantity):
    short_name: ClassVar[str]

    def __init__(self, value: _ValueLike, unit: str | AbstractUnit) -> None: ...
    def __class_getitem__(cls, item: Any) -> type[Self]: ...
    def __hash__(self) -> int: ...

    # Repeats `AbstractQuantity.__eq__`/`__ne__` verbatim rather than just
    # inheriting them: pyright infers `q1 == q2` as plain `bool` for a class
    # that doesn't declare `__eq__` in its own body, even though the
    # inherited method is correctly typed -- see `unxt/quantity.pyi`'s
    # `Quantity` for the same workaround and a fuller explanation.
    def __eq__(self, other: object, /) -> Bool[Array, "*shape"]: ...  # type: ignore[override]
    def __ne__(self, other: object, /) -> Bool[Array, "*shape"]: ...  # type: ignore[override]

PQ = ParametricQuantity
