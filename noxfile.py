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
    "tests_benchmark",
    # Documentation
    "docs",
    "build_api_docs",
]
nox.options.default_venv_backend = "uv"


DIR = Path(__file__).parent.resolve()


# =============================================================================
# Linting


@nox.session(venv_backend="uv")
def lint(session: nox.Session, /) -> None:
    """Run the linter."""
    precommit(session)  # reuse pre-commit session
    pylint(session)  # reuse pylint session


@nox.session(venv_backend="uv")
def precommit(session: nox.Session, /) -> None:
    """Run pre-commit."""
    session.run_install(
        "uv",
        "sync",
        "--group=lint",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(venv_backend="uv")
def pylint(session: nox.Session, /) -> None:
    """Run PyLint."""
    session.run_install(
        "uv",
        "sync",
        "--group=lint",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pylint", "unxt", *session.posargs)


# =============================================================================
# Testing


@nox.session(venv_backend="uv")
def tests(session: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    session.run_install(
        "uv",
        "sync",
        "--group=test",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", *session.posargs)


@nox.session(venv_backend="uv")
def tests_all(session: nox.Session, /) -> None:
    """Run the tests with all optional dependencies."""
    session.run_install(
        "uv",
        "sync",
        "--group=test-all",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", *session.posargs)


@nox.session(venv_backend="uv")
def tests_benchmark(session: nox.Session, /) -> None:
    """Run the benchmarks."""
    session.run_install(
        "uv",
        "sync",
        "--group=test",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", "tests/benchmark", "--codspeed", *session.posargs)


# =============================================================================
# Documentation


@nox.session(venv_backend="uv")(reuse_venv=True)
def docs(session: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    # Parse command line arguments for docs building
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

    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--group=docs",
        f"--python={session.virtualenv.location}",
        "--active",
        *offline_command,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.chdir("docs")

    # Build the docs
    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
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
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session(venv_backend="uv")
def build_api_docs(session: nox.Session, /) -> None:
    """Build (regenerate) API docs."""
    session.run_install(
        "uv",
        "sync",
        "--group=docs",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
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


@nox.session(venv_backend="uv")
def build(session: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.run_install(
        "uv",
        "sync",
        "--group=build",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("python", "-m", "build")
