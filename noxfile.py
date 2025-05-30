"""Nox sessions."""
# pylint: disable=import-error

import argparse
import shutil
from pathlib import Path

import nox

nox.needs_version = ">=2024.3.2"
nox.options.sessions = [
    # Linting
    "lint",
    "pylint",
    # Testing
    "tests",
    "tests_all",
    "tests_benckmark",
    # Documentation
    "docs",
    "build_api_docs",
]
nox.options.default_venv_backend = "uv"


DIR = Path(__file__).parent.resolve()


# =============================================================================
# Linting


@nox.session
def lint(session: nox.Session, /) -> None:
    """Run the linter."""
    session.run("uv", "sync")
    session.run(
        "uv",
        "run",
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *session.posargs,
    )


@nox.session
def pylint(session: nox.Session, /) -> None:
    """Run PyLint."""
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install(".", "pylint")
    session.run("pylint", "unxt", *session.posargs)


# =============================================================================
# Testing


@nox.session
def tests(session: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    session.run("uv", "sync", "--group", "test")
    session.run("uv", "run", "pytest", *session.posargs)


@nox.session
def tests_all(session: nox.Session, /) -> None:
    """Run the tests with all optional dependencies."""
    session.run("uv", "sync", "--group", "test-all")
    session.run("uv", "run", "pytest", *session.posargs)


@nox.session
def tests_benckmark(session: nox.Session, /) -> None:
    """Run the benchmarks."""
    session.run("uv", "sync", "--group", "test")
    session.run(
        "uv",
        "run",
        "pytest",
        "tests/benchmark",
        "--codspeed",
        *session.posargs,
    )


# =============================================================================
# Documentation


@nox.session(reuse_venv=True)
def docs(session: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("--offline", action="store_true", help="run in offline mode")
    parser.add_argument("--output-dir", dest="output_dir", default="_build")
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    offline_command = ["--offline"] if args.offline else []

    session.run("uv", "sync", "--group", "docs", "--active", *offline_command)
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        f"-d {args.output_dir}/doctrees",
        "-D language=en",
        ".",
        f"{args.output_dir}/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build_api_docs(session: nox.Session, /) -> None:
    """Build (regenerate) API docs."""
    session.install("sphinx")
    session.chdir("docs")
    session.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/unxt",
    )


@nox.session
def build(session: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
