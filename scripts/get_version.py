#!/usr/bin/env python3
"""Custom version determination for unxt workspace.

Implements the versioning strategy:
- vX.Y.0 tags apply to ALL workspace packages (major/minor releases)
- package-name-vX.Y.Z tags for bug-fix releases only (Z > 0)
- Validates that .0 releases use shared tags
- Validates that package-specific tags are bug-fix releases (Z > 0)
"""

import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_cmd(cmd: str) -> str:
    """Run a command and return its output."""
    result = subprocess.run(  # noqa: S603
        shlex.split(cmd), check=False, shell=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        msg = f"Command failed: {cmd}\nError: {result.stderr}"
        raise RuntimeError(msg)
    return result.stdout.strip()


def parse_version_tag(tag: str) -> tuple[str, int, int, int] | None:
    """Parse a version tag into (package, major, minor, patch).

    Returns `None` if tag doesn't match expected format.
    """
    # Format: [package-]vX.Y.Z
    match = re.match(r"^(?:([a-z-]+)-)?v(\d+)\.(\d+)\.(\d+)$", tag)
    if match:
        package = match.group(1) or ""
        major, minor, patch = (
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        return (package, major, minor, patch)
    return None


def validate_tag(tag: str, *, strict: bool = False) -> tuple[bool, str]:
    """Validate a tag according to the versioning rules.

    Tags must always match the supported [package-]vX.Y.Z format.
    For matching legacy tags (<1.11), shared-vs-package policy checks are
    grandfathered in. In strict mode, 1.11+ policy is enforced strictly.

    Returns (is_valid, error_message)
    """
    parsed = parse_version_tag(tag)
    if not parsed:
        return False, f"Invalid tag format: {tag}"

    package, major, minor, patch = parsed

    # Legacy tags (1.10.x and lower) are grandfathered for policy checks only.
    # Format validation still applies via parse_version_tag() above.
    if (major, minor) < (1, 11):
        return True, ""

    # For new versions (1.11.0+), enforce the new rules.
    if patch == 0:
        # .0 releases must be shared (no package prefix)
        if package:
            msg = (
                f"Tag {tag}: .0 releases must use shared tags "
                f"(vX.Y.0 format, not {tag}). "
                f"Use shared tag v{major}.{minor}.0 for all packages."
            )
            return (False, msg) if strict else (True, msg)
    # Bug-fix releases (.Z where Z > 0) must be package-specific
    elif not package:
        msg = (
            f"Tag {tag}: Bug-fix releases must be package-specific "
            f"(package-vX.Y.Z format). "
            f"Cannot use shared tag for bug-fix release."
        )
        return (False, msg) if strict else (True, msg)

    return True, ""


def determine_version(package_name: str | None = None) -> str:  # noqa: C901
    """Determine the version for a package based on git tags.

    Parameters
    ----------
    package_name : str or None
        The package name (e.g., 'unxt-api', 'unxt-hypothesis').
        If None, determines version for main 'unxt' package.

    Returns
    -------
    str
        The computed version string.

    """
    # Get all tags reachable from HEAD.
    # This avoids selecting newer tags from unrelated branches.
    try:
        all_tags_raw = run_cmd("git tag --merged HEAD -l")
    except RuntimeError:
        # Fallback for environments where --merged is unavailable.
        try:
            all_tags_raw = run_cmd("git tag -l")
        except RuntimeError:
            all_tags_raw = ""

    all_tags = all_tags_raw.split("\n") if all_tags_raw else []

    # Filter and validate tags.
    # Use strict validation so policy-violating 1.11+ tags cannot influence
    # version determination.
    valid_tags = []
    for tag_raw in all_tags:
        tag = tag_raw.strip()
        if not tag:
            continue

        is_valid, warning = validate_tag(tag, strict=True)
        if not is_valid:
            # Skip completely invalid tags
            continue

        if warning:
            # Print warnings for policy violations
            logger.warning("Warning: %s", warning)

        valid_tags.append(tag)

    if not valid_tags:
        msg = "No valid version tags found"
        raise RuntimeError(msg)

    # Parse tags to find latest
    parsed_tags = []
    for tag in valid_tags:
        parsed = parse_version_tag(tag)
        if parsed:
            parsed_tags.append((parsed, tag))

    # Filter tags for this package
    relevant_tags = []
    for (pkg, major, minor, patch), tag in parsed_tags:
        if package_name:
            # For subpackages: include shared tags (.0) and package-specific tags
            if pkg in {"", package_name}:
                relevant_tags.append((major, minor, patch, tag))
        # For main package: include shared tags and 'unxt' package tags
        elif pkg in {"", "unxt"}:
            relevant_tags.append((major, minor, patch, tag))

    if not relevant_tags:
        msg = f"No valid version tags found for package {package_name or 'unxt'}"
        raise RuntimeError(msg)

    # Sort by version (descending)
    relevant_tags.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    # Get the latest tag
    major, minor, patch, latest_tag = relevant_tags[0]

    # Use git describe with the exact latest tag to get proper version info
    # We use rev-list to find the distance from the tag to HEAD
    try:
        # Get the full describe-like output with the exact tag
        # Count commits since the tag
        distance = run_cmd(f"git rev-list --count {latest_tag}..HEAD")
        distance = int(distance)

        # Get the current commit hash
        commit_hash = run_cmd("git rev-parse --short HEAD")

        # Check if working directory is dirty
        dirty_status = run_cmd("git status --porcelain")
        dirty = "-dirty" if dirty_status else ""

        # Construct git describe-like output manually, always using --long style.
        # Always include -0-g<sha> even when exactly on a tag to match the
        # behavior of git describe --long for consistent hatch-vcs parsing.
        return f"{latest_tag}-{distance}-g{commit_hash}{dirty}"  # noqa: TRY300
    except RuntimeError:
        # Fallback to the tag name
        return latest_tag


if __name__ == "__main__":
    # Configure logging to output warnings/errors to stderr
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Auto-detect package from current working directory or command line
    package = None

    if len(sys.argv) > 1:
        package = None if sys.argv[1] == "--main" else sys.argv[1]
    else:
        # Auto-detect from current working directory
        cwd = str(Path.cwd())

        if "packages/unxt-api" in cwd:
            package = "unxt-api"
        elif "packages/unxt-hypothesis" in cwd:
            package = "unxt-hypothesis"
        elif "packages/" in cwd:
            # Extract package name from path like .../packages/PACKAGE/...
            match = re.search(r"packages/([a-z-]+)", cwd)
            if match:
                package = match.group(1)

    try:
        version = determine_version(package)
        print(version)  # Output version to stdout for hatch-vcs to read  # noqa: T201
    except RuntimeError:
        logger.exception("Error determining version")
        sys.exit(1)
