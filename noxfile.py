#!/usr/bin/env -S uv run --script  # noqa: EXE001
# /// script
#    dependencies = ["nox", "nox_uv"]
# ///
"""Nox setup."""

import argparse
import shutil
from enum import StrEnum, auto
from pathlib import Path
from typing import assert_never

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


class PackageEnum(StrEnum):
    """Enum for package names."""

    @staticmethod
    def _generate_next_value_(name: str, *_: object, **__: object) -> str:
        return name

    def __repr__(self) -> str:
        return f"{self.value!r}"

    unxt = auto()
    unxt_api = auto()
    unxt_hypothesis = auto()
    api = auto()
    hypothesis = auto()
    interop_gala = auto()
    interop_matplotlib = auto()
    interop_xarray = auto()
    parametric = auto()


# =============================================================================
# Comprehensive sessions


@session(
    uv_groups=["lint", "test", "docs"],
    uv_extras=["all"],
    reuse_venv=True,
    default=True,
)
def all(s: nox.Session, /) -> None:  # noqa: A001
    """Run all default sessions."""
    s.notify("lint")
    s.notify("test")
    s.notify("docs")


# =============================================================================
# Linting


@session(uv_groups=["lint"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    s.notify("precommit")
    for package in PackageEnum:
        s.notify(f"pylint(package={package.value!r})")


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run prek."""
    s.run("prek", "run", "--all-files", *s.posargs)


def _parse_pylint_paths(package: PackageEnum, /) -> list[str]:
    # Lint each package in isolation so the ``duplicate-code`` checker does not
    # flag the intentional similarity between a shim and its canonical package.
    match package:
        case PackageEnum.unxt:
            return ["src/unxt"]
        case PackageEnum.unxt_api:
            return ["packages/unxt-api/src"]
        case PackageEnum.unxt_hypothesis:
            return ["packages/unxt-hypothesis/src"]
        case PackageEnum.api:
            return ["packages/unxts.api/src"]
        case PackageEnum.hypothesis:
            return ["packages/unxts.hypothesis/src"]
        case PackageEnum.interop_gala:
            return ["packages/unxts.interop.gala/src"]
        case PackageEnum.interop_matplotlib:
            return ["packages/unxts.interop.matplotlib/src"]
        case PackageEnum.interop_xarray:
            return ["packages/unxts.interop.xarray/src"]
        case PackageEnum.parametric:
            return ["packages/unxts.parametric/src"]
        case _:
            assert_never(package)


@session(uv_groups=["lint"], uv_extras=["workspace"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def pylint(s: nox.Session, /, package: PackageEnum) -> None:
    """Run PyLint."""
    s.run("pylint", *_parse_pylint_paths(package), *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], uv_extras=["workspace"], reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    for package in PackageEnum:
        s.notify(f"pytest(package={package.value!r})", posargs=s.posargs)
    # s.notify("pytest_benchmark", posargs=s.posargs)


def _parse_pytest_paths(package: PackageEnum, /) -> list[str]:
    # The canonical ``unxts.*`` namespace packages point only at their ``tests``
    # directory: pytest's namespace-package path insertion would otherwise let a
    # leaf like ``unxts/interop/xarray`` shadow the real ``xarray`` when it
    # collects the src doctests. Those doctests are exercised via the docs pages.
    match package:
        case PackageEnum.unxt:
            return ["README.md", "docs", "src/", "tests/"]
        case PackageEnum.unxt_api:
            return ["packages/unxt-api/"]
        case PackageEnum.unxt_hypothesis:
            return ["packages/unxt-hypothesis/"]
        case PackageEnum.api:
            return ["packages/unxts.api/tests"]
        case PackageEnum.hypothesis:
            return ["packages/unxts.hypothesis/tests"]
        case PackageEnum.interop_gala:
            return ["packages/unxts.interop.gala/tests"]
        case PackageEnum.interop_matplotlib:
            return ["packages/unxts.interop.matplotlib/tests"]
        case PackageEnum.interop_xarray:
            return ["packages/unxts.interop.xarray/tests"]
        case PackageEnum.parametric:
            return ["packages/unxts.parametric/tests"]
        case _:
            assert_never(package)


# ``test-all`` (not ``test``) because the ``workspace`` extra pulls in
# matplotlib via ``unxt[interop-mpl]``, so the matplotlib integration tests are
# collected and need ``pytest-mpl`` to register the ``mpl_image_compare`` marker.
@session(uv_groups=["test-all"], uv_extras=["workspace"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def pytest(s: nox.Session, /, package: PackageEnum) -> None:
    """Run the unit and regular tests."""
    package_paths = _parse_pytest_paths(package)
    s.run("pytest", *package_paths, *s.posargs)


@session(uv_groups=["test-all"], uv_extras=["interop-mpl"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def pytest_all(s: nox.Session, /, package: PackageEnum) -> None:
    """Run the unit and regular tests."""
    package_paths = _parse_pytest_paths(package)
    s.run("pytest", *package_paths, *s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def pytest_benchmark(s: nox.Session, /) -> None:
    """Run the benchmarks."""
    s.run("pytest", "tests/benchmark", "--codspeed", *s.posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], uv_extras=["workspace"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("--offline", action="store_true", help="run in offline mode")
    parser.add_argument("--output-dir", dest="output_dir", default="_build")
    args, posargs = parser.parse_known_args(s.posargs)

    if args.builder != "html" and args.serve:
        s.error("Must not specify non-HTML builder with --serve")

    s.chdir("docs")

    # Generate custom intersphinx inventories
    s.run("python", "_static/generate_jaxtyping_inv.py")
    s.run("python", "_static/generate_equinox_inv.py")
    s.run("python", "_static/generate_quax_blocks_inv.py")

    # Convert jupytext markdown files to notebooks
    s.run(
        "jupytext",
        "--to",
        "notebook",
        "guides/perf.md",
        "--output",
        "guides/perf.ipynb",
    )

    if args.builder == "linkcheck":
        s.run("sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs)
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        f"-d={args.output_dir}/doctrees",
        "-D",
        "language=en",
        ".",
        f"{args.output_dir}/{args.builder}",
        *posargs,
    )

    if args.serve:
        s.run("sphinx-autobuild", *shared_args)
    else:
        s.run("sphinx-build", "--keep-going", *shared_args)


@session(uv_groups=["docs"], reuse_venv=True)
def build_api_docs(s: nox.Session, /) -> None:
    """Build (regenerate) API docs."""
    s.chdir("docs")
    s.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/unxt",
    )


# =============================================================================
# Packaging


@session(uv_groups=["build"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    s.run("python", "-m", "build")


# =============================================================================

if __name__ == "__main__":
    nox.main()
