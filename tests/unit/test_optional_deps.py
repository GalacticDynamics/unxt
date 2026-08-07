"""Tests for the internal optional-dependency detection (`unxt._interop`)."""

import pytest

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import is_installed

from unxt._interop.optional_deps import OptDeps


def test_no_members_alias() -> None:
    """Every declared member is its own member, not an alias of an earlier one.

    ``optional-dependencies`` < 0.4.1 keyed members on their resolved version,
    so co-released packages (unxt's own ``unxts.*``) silently collapsed into a
    single member, reporting the wrong package's state. 0.4.1 keys on an
    identity wrapper instead.
    """
    assert len(OptDeps.__members__) == len(list(OptDeps))


def test_absent_packages_do_not_alias() -> None:
    """Two *uninstalled* members stay distinct.

    The other half of the pre-0.4.1 aliasing: every absent package resolved to
    the same ``NOT_INSTALLED`` sentinel, so the second one folded into the
    first. Asserted on a throwaway enum because it needs packages that are
    guaranteed absent, which ``OptDeps`` (deliberately) has none of.
    """

    class Absent(OptionalDependencyEnum):
        NO_SUCH_PACKAGE_A = auto()
        NO_SUCH_PACKAGE_B = auto()

    assert not Absent.NO_SUCH_PACKAGE_A.installed
    assert not Absent.NO_SUCH_PACKAGE_B.installed
    assert Absent.NO_SUCH_PACKAGE_A is not Absent.NO_SUCH_PACKAGE_B
    assert len(Absent.__members__) == len(list(Absent))


@pytest.mark.parametrize(
    ("member", "module"),
    [
        (OptDeps.UNXTS_INTEROP_MATPLOTLIB, "unxts.interop.matplotlib"),
        (OptDeps.UNXTS_INTEROP_XARRAY, "unxts.interop.xarray"),
    ],
)
def test_detects_interop_package_independently(member: OptDeps, module: str) -> None:
    """Same-versioned interop packages are each detected on their own.

    These are optional extras, so ``importorskip`` skips the case where the
    module isn't installed.
    """
    pytest.importorskip(module)
    assert member.installed is True


def test_gala_member_requires_the_gala_backend() -> None:
    """``UNXTS_INTEROP_GALA`` carries *both* halves of the gala condition.

    The member is built with ``chain_checks`` because the interop extra can be
    installed while gala itself is not -- gala is skipped where it cannot build
    (e.g. the Windows CI job), and that is the env where dropping the second
    check would show up as a wrongly-`True` member.
    """
    expected = is_installed("unxts.interop.gala") and is_installed("gala")
    assert OptDeps.UNXTS_INTEROP_GALA.installed is expected
