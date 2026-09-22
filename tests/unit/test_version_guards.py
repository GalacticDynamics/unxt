"""Guard against JAX compatibility shims outliving the supported floor.

Every JAX version guard in `unxt` names the release it is there for, as a
comparison against `jax.version.__version_info__`::

    if JAX_VERSION >= (0, 11, 2):
        ...

Never probe with ``hasattr``: it hides *which* release changed, so the shim can
never be confidently deleted.

This test reads the ``jax>=`` floor from `pyproject.toml` and fails for any
guard at or below it, since such a guard is by then a constant. Bumping the
floor therefore produces a list of exactly the shims to delete.
"""

import operator
import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

# Resolved, so the self-skip in `_guards` is a reliable comparison whatever
# `__file__` looks like under a given pytest import mode, and through symlinked
# checkouts (on macOS a `/tmp` working copy is really `/private/tmp`).
SELF = Path(__file__).resolve()
REPO_ROOT = SELF.parents[2]
SCAN_DIRS = ("src", "tests")

# A `JAX_VERSION >= (0, 11, 2)`-style guard. The convention is `>=`; the other
# comparisons are matched too, so a stray one is still checked rather than
# silently skipped.
GUARD_RE = re.compile(r"JAX_VERSION\s*(>=|<=|>|<)\s*\((\d+),\s*(\d+),\s*(\d+)\)")

# When a floor makes each comparison a constant. Given `JAX_VERSION >= floor`,
# `>=` and `<` are constant as soon as the floor *reaches* the guarded version.
# The strict `>` and the inclusive `<=` only become constant once it *passes*:
# at exactly the floor both still discriminate, since JAX may be newer.
DEAD_AT = {">=": operator.ge, "<": operator.ge, ">": operator.gt, "<=": operator.gt}

# A `hasattr` probe of a JAX module -- the anti-pattern these tests ban. Matched
# as a pattern rather than a fixed substring so spacing and the module spelling
# (`lax`, `jax.lax`, `jnp`, ...) cannot slip a probe past the check.
HASATTR_RE = re.compile(r"\bhasattr\s*\(\s*(?:jax[\w.]*|lax|jnp)\s*,")


def _jax_floor() -> Version:
    """The lowest JAX release `unxt` claims to support."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    for dep in pyproject["project"]["dependencies"]:
        req = Requirement(dep)
        if req.name == "jax":
            (spec,) = [s for s in req.specifier if s.operator == ">="]
            return Version(spec.version)
    msg = "no `jax` dependency found in pyproject.toml"
    raise AssertionError(msg)


def _guards() -> list[tuple[Path, int, str, Version]]:
    """Every JAX version guard in the codebase, with its location and operator."""
    found = set()
    for directory in SCAN_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if path.resolve() == SELF:  # this file's own examples are not guards
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                found |= {
                    (path.relative_to(REPO_ROOT), lineno, op, Version(".".join(ver)))
                    for op, *ver in GUARD_RE.findall(line)
                }
    return sorted(found, key=lambda g: (str(g[0]), g[1], g[3]))


def test_no_version_guard_below_supported_floor() -> None:
    """Every JAX version guard is still meaningful at the supported floor."""
    floor = _jax_floor()
    guards = _guards()
    assert guards, "found no version guards at all -- the scan is broken"
    assert not [g for g in guards if g[0] == SELF.relative_to(REPO_ROOT)], (
        "this file's own docstring examples were scanned -- the self-skip broke"
    )

    dead = [(p, n, op, v) for p, n, op, v in guards if DEAD_AT[op](floor, v)]
    assert not dead, "JAX version guards made dead by the jax>={} floor:\n{}".format(
        floor, "\n".join(f"  {p}:{n}: `JAX_VERSION {op} {v}`" for p, n, op, v in dead)
    )


def test_no_hasattr_probing_of_jax_primitives() -> None:
    """Primitive registration names a JAX version rather than probing for it."""
    path = Path("src/unxt/_src/quantity/register_primitives.py")
    probes = [
        f"  {path}:{n}: {line.strip()}"
        for n, line in enumerate((REPO_ROOT / path).read_text().splitlines(), start=1)
        if HASATTR_RE.search(line)
    ]
    assert not probes, (
        "`hasattr` probing of `jax.lax`; find the JAX release that introduced "
        "the primitive and guard on `JAX_VERSION` instead:\n" + "\n".join(probes)
    )


@pytest.mark.parametrize(
    "line",
    [
        'if hasattr(lax, "stack_p"):',
        'if hasattr( lax , "stack_p"):',
        'if hasattr(jax.lax, "stack_p"):',
        'if hasattr(jnp, "float_"):',
        "if hasattr (jax, 'Inline'):",
    ],
)
def test_hasattr_pattern_catches_probe_spellings(line: str) -> None:
    """The probe pattern catches the spellings a fixed substring would miss."""
    assert HASATTR_RE.search(line)


@pytest.mark.parametrize(
    "line",
    ["missing = [n for n in names if not hasattr(self, n)]", "x = hasattr(obj, 'a')"],
)
def test_hasattr_pattern_ignores_unrelated_probes(line: str) -> None:
    """Probing non-JAX objects is ordinary Python, not the banned anti-pattern."""
    assert not HASATTR_RE.search(line)


if __name__ == "__main__":
    pytest.main([__file__])
