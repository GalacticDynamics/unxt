#!/usr/bin/env python3
"""Validate git tags for the new versioning strategy in CI/CD.

New strategy (post-refactoring):
- Shared vX.Y.0 tags are coordinator tags (auto-create package-specific tags)
- Only package-specific tags (package-vX.Y.Z) trigger builds
- package-vX.Y.0 tags must have a corresponding vX.Y.0 coordinator tag
- package-vX.Y.Z (Z > 0) tags are independent bug-fix releases

"""

import logging
import re
import subprocess
import sys

logger = logging.getLogger(__name__)


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


def check_coordinator_tag_exists(version: str) -> bool:
    """Check if a coordinator tag (vX.Y.Z) exists in the repo.

    Parameters
    ----------
    version : str
        Version without 'v' prefix (e.g., "1.8.0")

    Returns
    -------
    bool
        True if vX.Y.Z tag exists, False otherwise

    Raises
    ------
    RuntimeError
        If git command fails (e.g., tags not fetched, git not available)

    """
    coordinator_tag = f"v{version}"
    result = subprocess.run(  # noqa: S603
        ["git", "tag", "-l", coordinator_tag],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )

    # Check for git command failure to avoid misleading "tag doesn't exist" errors
    if result.returncode != 0:
        error_msg = f"Failed to check for coordinator tag v{version}: git tag -l failed"
        if result.stderr:
            logger.error("%s\nStderr: %s", error_msg, result.stderr.strip())
        else:
            logger.error("%s (no stderr output)", error_msg)
        msg = (
            f"git tag -l failed with exit code {result.returncode}. "
            "This usually means tags weren't fetched. "
            "Ensure 'fetch-depth: 0' is set in actions/checkout."
        )
        raise RuntimeError(msg)

    return coordinator_tag in result.stdout.strip().split("\n")


def validate_tag_for_package(tag: str, package: str | None = None) -> tuple[bool, str]:
    """Validate a tag for a specific package with strict rules.

    Parameters
    ----------
    tag : str
        The git tag to validate (should be package-specific: package-vX.Y.Z)
    package : str or None
        The package being validated ('unxt', 'unxt-api', 'unxt-hypothesis', or None)
        If None, validates for main unxt package.

    Returns
    -------
    (is_valid, error_message)
        (True, "") if valid
        (False, error_message) if invalid

    """
    parsed = parse_version_tag(tag)
    if not parsed:
        return False, f"Invalid tag format: {tag}"

    tag_package, major, minor, patch = parsed

    # Legacy tags (1.10.x and lower) are grandfathered in to support
    # maintenance releases.
    if (major, minor) < (1, 11):
        return True, ""

    # Normalize package name (None means main unxt package)
    if package is None:
        package = "unxt"

    # RULE 1: Must be a package-specific tag
    if not tag_package:
        return False, (
            f"❌ Tag {tag}: Package CD workflows should only trigger on "
            f"package-specific tags (e.g., {package}-v{major}.{minor}.{patch}). "
            f"Shared vX.Y.Z tags are coordinator tags that auto-create package tags."
        )

    # RULE 2: Package-specific tag must match current package
    if tag_package != package:
        return False, (
            f"❌ Tag {tag}: This tag is for package '{tag_package}', "
            f"but this workflow is for package '{package}'. Skipping release."
        )

    # RULE 3: For .0 releases, verify coordinator tag exists
    if patch == 0:
        version = f"{major}.{minor}.{patch}"
        if not check_coordinator_tag_exists(version):
            return False, (
                f"❌ Tag {tag}: Package .0 releases must have a corresponding "
                f"coordinator tag v{version}. This ensures synchronized releases. "
                f"Create v{version} first, which will auto-create package tags."
            )

    # All checks passed
    return True, ""


def main() -> int:
    """CI validation entry point."""
    # Configure logging to output to stdout/stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    if len(sys.argv) < 2:
        logger.error("Usage: validate_tag.py TAG [PACKAGE]")
        logger.error("  TAG: The git tag to validate")
        logger.error("  PACKAGE: Package name ('unxt', 'unxt-api', 'unxt-hypothesis')")
        sys.exit(1)

    tag = sys.argv[1]
    package = sys.argv[2] if len(sys.argv) > 2 else None

    is_valid, error_msg = validate_tag_for_package(tag, package)

    if is_valid:
        if package:
            logger.info("✅ Tag %s is valid for package '%s'", tag, package)
        else:
            logger.info("✅ Tag %s is valid for main package", tag)
        return 0
    logger.error(error_msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
