"""Tests for the internal optional-dependency detection (`unxt._interop`)."""

import pytest

from unxt._interop.optional_deps import OptDeps, is_installed


class TestOptDepsExcludesNamespacePackages:
    """`OptDeps` must not contain unxt's own version-colliding `unxts.*`."""

    def test_optdeps_has_no_unxts_members(self) -> None:
        """Guard against reintroducing ``unxts.*`` packages into ``OptDeps``.

        ``OptionalDependencyEnum`` keys each member on its installed *version*,
        so any two members that share a version silently collapse into one enum
        alias. unxt's own ``unxts.*`` packages are released together and so
        typically share a version; that is the concrete aliasing this PR fixed
        (matplotlib/xarray aliasing gala). They must be detected with
        ``is_installed`` instead of being ``OptDeps`` members.

        Asserting on the aliasing directly (``len(__members__) == len(OptDeps)``)
        is version/environment-dependent, so assert the underlying invariant: no
        member is a ``unxts.*`` package.
        """
        offenders = [n for n in OptDeps.__members__ if n.startswith("UNXTS")]
        assert not offenders, (
            "unxts.* packages must not be OptDeps members (they alias by shared "
            f"version); detect them with is_installed() instead. Found: {offenders}"
        )


class TestIsInstalled:
    """`is_installed` detects modules by import presence."""

    def test_returns_true_for_an_installed_module(self) -> None:
        """A module that is importable reports as installed."""
        assert is_installed("unxt") is True
        assert is_installed("importlib.util") is True

    def test_returns_false_for_absent_module(self) -> None:
        """A module without a discoverable spec reports as not installed."""
        assert is_installed("unxts.interop.does_not_exist") is False
        assert is_installed("unxt._this_module_does_not_exist_xyz") is False

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
