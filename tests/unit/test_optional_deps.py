"""Tests for the internal optional-dependency detection (`unxt._interop`)."""

import pytest

from unxt._interop.optional_deps import OptDeps, is_installed


class TestOptDepsNoAliasing:
    """`OptDeps` members must never silently alias each other."""

    def test_optdeps_has_no_aliased_members(self) -> None:
        """Members sharing a value must not collapse into aliases.

        Enum members with equal values become aliases: the second is silently
        dropped from ``list(OptDeps)`` while remaining in ``__members__``. This
        happened when several ``unxts.interop.*`` packages shared the monorepo
        version, so matplotlib/xarray aliased gala.
        """
        assert len(OptDeps.__members__) == len(list(OptDeps))


class TestIsInstalled:
    """`is_installed` detects modules by import presence."""

    def test_returns_true_for_an_installed_module(self) -> None:
        """A module that is importable reports as installed."""
        assert is_installed("unxt") is True
        assert is_installed("importlib.util") is True

    def test_returns_false_for_absent_module(self) -> None:
        """A module that cannot be imported reports as not installed."""
        assert is_installed("unxts.interop.does_not_exist") is False
        assert is_installed("a_package_that_is_not_installed_xyz") is False

    @pytest.mark.parametrize(
        "module",
        ["unxts.interop.gala", "unxts.interop.matplotlib", "unxts.interop.xarray"],
    )
    def test_detects_interop_package_independently(self, module: str) -> None:
        """Same-versioned interop packages are each detected on their own.

        This is the behaviour the version-keyed enum could not provide: import
        presence is resolved per-module, so packages sharing a version never
        collapse together. The interop packages are optional extras, so skip
        when the extra isn't installed.
        """
        pytest.importorskip(module)
        assert is_installed(module) is True
