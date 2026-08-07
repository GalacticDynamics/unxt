"""Tests for the internal optional-dependency detection (`unxt._interop`)."""

import pytest

from unxt._interop.optional_deps import OptDeps


def test_no_members_alias() -> None:
    """Every declared member is its own member, not an alias of an earlier one.

    ``optional-dependencies`` < 0.4.1 keyed members on their resolved version,
    so co-released packages (unxt's own ``unxts.*``) and any two uninstalled
    packages silently collapsed into a single member -- reporting the wrong
    package's state. 0.4.1 keys on an identity wrapper instead.
    """
    assert len(OptDeps.__members__) == len(list(OptDeps))


@pytest.mark.parametrize(
    ("member", "module"),
    [
        (OptDeps.UNXTS_INTEROP_GALA, "unxts.interop.gala"),
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
