"""Test that unxt_hypothesis re-exports the full public API from unxts.hypothesis."""

import unxts.hypothesis

import unxt_hypothesis  # noqa: ICN001


def test_all_symbols_re_exported():
    for name in unxts.hypothesis.__all__:
        assert hasattr(unxt_hypothesis, name), f"unxt_hypothesis missing: {name}"


def test_same_objects():
    for name in unxts.hypothesis.__all__:
        if name == "__version__":
            continue
        assert getattr(unxt_hypothesis, name) is getattr(unxts.hypothesis, name), (
            f"unxt_hypothesis.{name} is not unxts.hypothesis.{name}"
        )
