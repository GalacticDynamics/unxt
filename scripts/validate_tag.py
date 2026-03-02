#!/usr/bin/env python3
"""Validate git tags for the new versioning strategy in CI/CD.

Ensures:
- vX.Y.0 (shared) tags are valid and triggers all packages
- package-vX.Y.Z (Z > 0) tags trigger only that package
- Invalid tags are rejected (e.g., package-vX.Y.0, vX.Y.Z with Z > 0)

"""

import logging
import sys

from get_version import parse_version_tag

logger = logging.getLogger(__name__)


def validate_tag_for_package(tag: str, package: str | None = None) -> tuple[bool, str]:
    """Validate a tag for a specific package with strict rules.

    Parameters
    ----------
    tag : str
        The git tag to validate
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

    # RULE 1: .0 releases must be shared (no package prefix)
    if patch == 0:
        if tag_package:
            return False, (
                f"❌ Tag {tag}: .0 releases must use shared tags "
                f"(vX.Y.0 format). Package-specific .0 tags are forbidden. "
                f"Use v{major}.{minor}.0 instead."
            )
        # Shared .0 tag triggers all packages - valid for any package
        return True, ""

    # RULE 2: Bug-fix releases (Z > 0) must be package-specific
    if not tag_package:
        return False, (
            f"❌ Tag {tag}: Bug-fix releases must be package-specific "
            f"(package-vX.Y.Z). Cannot use shared tag v{major}.{minor}.{patch} "
            f"for bug-fix release."
        )

    # RULE 3: Package-specific tag must match current package
    if tag_package != package:
        return False, (
            f"❌ Tag {tag}: This tag is for package '{tag_package}', "
            f"but this workflow is for package '{package}'. Skipping release."
        )

    # Package-specific bug-fix tag is valid for its package
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
