#!/usr/bin/env -S uv run --script  # noqa: EXE001
# /// script
#    dependencies = ["nox", "nox_uv"]
# ///
"""Nox setup."""

import argparse
import shutil
from pathlib import Path

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()

# =============================================================================
# Comprehensive sessions


@session(uv_groups=["all"], reuse_venv=True, default=True)
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
    s.notify("pylint")


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run pre-commit."""
    s.run("pre-commit", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
def pylint(s: nox.Session, /) -> None:
    """Run PyLint."""
    s.run("pylint", "unxt", *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test-all"], reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.notify("pytest", posargs=s.posargs)
    # s.notify("pytest_benchmark", posargs=s.posargs)


@session(uv_groups=["test-all"], reuse_venv=True)
def pytest(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.run("pytest", *s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def pytest_minimal(s: nox.Session, /) -> None:
    """Run the tests with minimal dependencies."""
    s.run("pytest", *s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def pytest_benchmark(s: nox.Session, /) -> None:
    """Run the benchmarks."""
    s.run("pytest", "tests/benchmark", "--codspeed", *s.posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], reuse_venv=True)
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
