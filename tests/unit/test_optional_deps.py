"""Tests for the internal optional-dependency detection (`unxt._interop`)."""

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
    """`is_installed` detects each interop package independently."""

    def test_detects_each_interop_package_independently(self) -> None:
        """Same-versioned interop packages are each detected on their own.

        This is the behaviour the version-keyed enum could not provide: import
        presence is resolved per-module, so packages sharing a version never
        collapse together.
        """
        assert is_installed("unxts.interop.gala") is True
        assert is_installed("unxts.interop.matplotlib") is True
        assert is_installed("unxts.interop.xarray") is True

    def test_returns_false_for_absent_module(self) -> None:
        """A module that cannot be imported reports as not installed."""
        assert is_installed("unxts.interop.does_not_exist") is False
        assert is_installed("a_package_that_is_not_installed_xyz") is False
