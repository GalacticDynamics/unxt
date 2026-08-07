"""Shared helpers for the `unxts.hypothesis` strategies."""

__all__: tuple[str, ...] = ()

from hypothesis import strategies as st


def draw_if_strategy[T](draw: st.DrawFn, v: T | st.SearchStrategy[T], /) -> T:
    """Draw a value if a strategy is given, else return the value.

    Every public strategy here accepts either a concrete value or a strategy
    that generates one, so each would otherwise repeat this conditional.
    """
    return draw(v) if isinstance(v, st.SearchStrategy) else v
