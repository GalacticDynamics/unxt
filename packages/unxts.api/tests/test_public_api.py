"""Smoke tests for the unxts.api public API."""

import unxts.api


def test_all_symbols_present():
    for name in unxts.api.__all__:
        assert hasattr(unxts.api, name), f"unxts.api missing: {name}"


def test_dispatch_functions_are_callable():
    for name in unxts.api.__all__:
        if name == "__version__":
            continue
        assert callable(getattr(unxts.api, name)), f"unxts.api.{name} not callable"


def test_version_is_a_string():
    assert isinstance(unxts.api.__version__, str)


def test_wrap_to_keyword_form_redirects_to_positional():
    """``wrap_to(x, min=..., max=...)`` forwards to the positional method."""

    @unxts.api.wrap_to.dispatch
    def _(x: int, min: int, max: int, /) -> tuple:
        return (x, min, max)

    assert unxts.api.wrap_to(1, min=2, max=3) == (1, 2, 3)
