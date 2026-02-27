"""Tests for scripts/get_version.py behavior."""

from unittest.mock import patch

import scripts.get_version as gv


def test_determine_version_uses_long_style_on_exact_tag() -> None:
    """Return `tag-0-g<sha>` even when exactly on the tag."""
    responses = {
        "git tag --merged HEAD -l": "v1.11.0",
        "git rev-list --count v1.11.0..HEAD": "0",
        "git rev-parse --short HEAD": "abc1234",
        "git status --porcelain": "",
    }

    with patch.object(gv, "run_cmd", side_effect=lambda cmd: responses[cmd]):
        assert gv.determine_version(None) == "v1.11.0-0-gabc1234"


def test_determine_version_appends_dirty_suffix() -> None:
    """Append `-dirty` after the git-describe-like long version string."""
    responses = {
        "git tag --merged HEAD -l": "v1.11.0",
        "git rev-list --count v1.11.0..HEAD": "2",
        "git rev-parse --short HEAD": "def5678",
        "git status --porcelain": " M scripts/get_version.py",
    }

    with patch.object(gv, "run_cmd", side_effect=lambda cmd: responses[cmd]):
        assert gv.determine_version(None) == "v1.11.0-2-gdef5678-dirty"
