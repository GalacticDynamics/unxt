"""Tests for unxts.api.wrap_to dispatch behavior."""

import unxts.api


def test_wrap_to_keyword_form_redirects_to_positional():
    """``wrap_to(x, min=..., max=...)`` forwards to the positional method."""

    @unxts.api.wrap_to.dispatch
    def _(x: int, min: int, max: int, /) -> tuple:
        return (x, min, max)

    assert unxts.api.wrap_to(1, min=2, max=3) == (1, 2, 3)
