"""Test that unxt_api re-exports the full public API from unxts.api."""

import unxts.api

import unxt_api  # noqa: ICN001


def test_all_symbols_re_exported():
    for name in unxts.api.__all__:
        assert hasattr(unxt_api, name), f"unxt_api missing: {name}"


def test_same_objects():
    for name in unxts.api.__all__:
        if name == "__version__":
            continue
        assert getattr(unxt_api, name) is getattr(unxts.api, name), (
            f"unxt_api.{name} is not unxts.api.{name}"
        )
