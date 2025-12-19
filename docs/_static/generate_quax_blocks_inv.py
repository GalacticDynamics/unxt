#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sphobjinv>=2.3",
# ]
# ///
"""Generate a minimal intersphinx inventory for quax-blocks.

Writes: quax_blocks.inv (next to this script, i.e. docs/_static/quax_blocks.inv)

Usage:
    ./docs/_static/generate_quax_blocks_inv.py
    uv run docs/_static/generate_quax_blocks_inv.py
"""

import logging
import pathlib
from collections.abc import Iterable
from typing import NamedTuple

import sphobjinv as soi

logger = logging.getLogger(__name__)

# ---- Configure your entries here -------------------------------------------------
# name: fully-qualified reference you will use in Sphinx roles, e.g.
#       :py:class:`quax_blocks.NumpyBinaryOpsMixin`
# role: Sphinx role to attach (e.g. "class", "data", "func", "attr", ...);
#       domain is "py"
# uri:  relative URL from the quax-blocks docs root to the target page/anchor
# disp: optional display text (link label); None -> use the shortname from `name`


class Entry(NamedTuple):
    """Intersphinx inventory entry for a quax-blocks type.

    Attributes
    ----------
    name
        Fully-qualified name (e.g., "quax_blocks.NumpyBinaryOpsMixin") or unqualified
        alias (e.g., "NumpyBinaryOpsMixin").
    role
        Sphinx role type: "class", "data", "func", "attr", etc.
    uri
        Relative URL from quax-blocks docs root to the target anchor.
    disp
        Display text for the link. If None, uses the shortname from `name`.

    """

    name: str
    role: str
    uri: str
    disp: str | None = None


ENTRIES: tuple[Entry, ...] = (
    # Qualified - Main operator mixins
    Entry(
        "quax_blocks.NumpyBinaryOpsMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "NumpyBinaryOpsMixin",
    ),
    Entry(
        "quax_blocks.LaxBinaryOpsMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "LaxBinaryOpsMixin",
    ),
    Entry(
        "quax_blocks.NumpyComparisonMixin",
        "class",
        "blob/main/src/quax_blocks/_src/rich.py",
        "NumpyComparisonMixin",
    ),
    Entry(
        "quax_blocks.LaxComparisonMixin",
        "class",
        "blob/main/src/quax_blocks/_src/rich.py",
        "LaxComparisonMixin",
    ),
    Entry(
        "quax_blocks.NumpyUnaryMixin",
        "class",
        "blob/main/src/quax_blocks/_src/unary.py",
        "NumpyUnaryMixin",
    ),
    Entry(
        "quax_blocks.LaxUnaryMixin",
        "class",
        "blob/main/src/quax_blocks/_src/unary.py",
        "LaxUnaryMixin",
    ),
    Entry(
        "quax_blocks.NumpyMathMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "NumpyMathMixin",
    ),
    Entry(
        "quax_blocks.LaxMathMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "LaxMathMixin",
    ),
    Entry(
        "quax_blocks.NumpyBitwiseMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "NumpyBitwiseMixin",
    ),
    Entry(
        "quax_blocks.LaxBitwiseMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "LaxBitwiseMixin",
    ),
    # Round operations
    Entry(
        "quax_blocks.NumpyRoundMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyRoundMixin",
    ),
    Entry(
        "quax_blocks.LaxRoundMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxRoundMixin",
    ),
    Entry(
        "quax_blocks.NumpyTruncMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyTruncMixin",
    ),
    Entry(
        "quax_blocks.LaxTruncMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxTruncMixin",
    ),
    Entry(
        "quax_blocks.NumpyFloorMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyFloorMixin",
    ),
    Entry(
        "quax_blocks.LaxFloorMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxFloorMixin",
    ),
    Entry(
        "quax_blocks.NumpyCeilMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyCeilMixin",
    ),
    Entry(
        "quax_blocks.LaxCeilMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxCeilMixin",
    ),
    # Container operations
    Entry(
        "quax_blocks.NumpyLenMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "NumpyLenMixin",
    ),
    Entry(
        "quax_blocks.LaxLenMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "LaxLenMixin",
    ),
    Entry(
        "quax_blocks.NumpyLengthHintMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "NumpyLengthHintMixin",
    ),
    Entry(
        "quax_blocks.LaxLengthHintMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "LaxLengthHintMixin",
    ),
    # Copy operations
    Entry(
        "quax_blocks.NumpyCopyMixin",
        "class",
        "blob/main/src/quax_blocks/_src/copy.py",
        "NumpyCopyMixin",
    ),
    Entry(
        "quax_blocks.NumpyDeepCopyMixin",
        "class",
        "blob/main/src/quax_blocks/_src/copy.py",
        "NumpyDeepCopyMixin",
    ),
    # Unqualified aliases to catch from signatures/docstrings
    Entry(
        "NumpyBinaryOpsMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "NumpyBinaryOpsMixin",
    ),
    Entry(
        "LaxBinaryOpsMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "LaxBinaryOpsMixin",
    ),
    Entry(
        "NumpyComparisonMixin",
        "class",
        "blob/main/src/quax_blocks/_src/rich.py",
        "NumpyComparisonMixin",
    ),
    Entry(
        "LaxComparisonMixin",
        "class",
        "blob/main/src/quax_blocks/_src/rich.py",
        "LaxComparisonMixin",
    ),
    Entry(
        "NumpyUnaryMixin",
        "class",
        "blob/main/src/quax_blocks/_src/unary.py",
        "NumpyUnaryMixin",
    ),
    Entry(
        "LaxUnaryMixin",
        "class",
        "blob/main/src/quax_blocks/_src/unary.py",
        "LaxUnaryMixin",
    ),
    Entry(
        "NumpyMathMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "NumpyMathMixin",
    ),
    Entry(
        "LaxMathMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "LaxMathMixin",
    ),
    Entry(
        "NumpyBitwiseMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "NumpyBitwiseMixin",
    ),
    Entry(
        "LaxBitwiseMixin",
        "class",
        "blob/main/src/quax_blocks/_src/binary.py",
        "LaxBitwiseMixin",
    ),
    Entry(
        "NumpyRoundMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyRoundMixin",
    ),
    Entry(
        "LaxRoundMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxRoundMixin",
    ),
    Entry(
        "NumpyTruncMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyTruncMixin",
    ),
    Entry(
        "LaxTruncMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxTruncMixin",
    ),
    Entry(
        "NumpyFloorMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyFloorMixin",
    ),
    Entry(
        "LaxFloorMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxFloorMixin",
    ),
    Entry(
        "NumpyCeilMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "NumpyCeilMixin",
    ),
    Entry(
        "LaxCeilMixin",
        "class",
        "blob/main/src/quax_blocks/_src/round.py",
        "LaxCeilMixin",
    ),
    Entry(
        "NumpyLenMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "NumpyLenMixin",
    ),
    Entry(
        "LaxLenMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "LaxLenMixin",
    ),
    Entry(
        "NumpyLengthHintMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "NumpyLengthHintMixin",
    ),
    Entry(
        "LaxLengthHintMixin",
        "class",
        "blob/main/src/quax_blocks/_src/container.py",
        "LaxLengthHintMixin",
    ),
    Entry(
        "NumpyCopyMixin",
        "class",
        "blob/main/src/quax_blocks/_src/copy.py",
        "NumpyCopyMixin",
    ),
    Entry(
        "NumpyDeepCopyMixin",
        "class",
        "blob/main/src/quax_blocks/_src/copy.py",
        "NumpyDeepCopyMixin",
    ),
)


# ----------------------------------------------------------------------------------


def build_inventory(
    entries: Iterable[Entry],
    project: str = "quax-blocks",
    version: str = "latest",
) -> soi.Inventory:
    inv = soi.Inventory()
    inv.project = project
    inv.version = version

    for e in entries:
        disp = e.disp if e.disp is not None else e.name.rsplit(".", 1)[-1]
        inv.objects.append(
            soi.DataObjStr(
                name=e.name,
                domain="py",
                role=e.role,
                priority="1",
                uri=e.uri,
                dispname=disp,
            )
        )

    return inv


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    here = pathlib.Path(__file__).resolve().parent
    out_path = here / "quax_blocks.inv"

    inv = build_inventory(ENTRIES)
    text = inv.data_file(contract=True)  # plaintext inventory bytes
    ztext = soi.compress(text)  # compressed objects.inv bytes
    soi.writebytes(str(out_path), ztext)

    logger.info("Wrote %s", out_path)
    logger.info("Inventory contents:")
    for obj in inv.objects:
        logger.info(
            "  %s:%s:`%s` -> %s (%s)",
            obj.domain,
            obj.role,
            obj.name,
            obj.uri,
            obj.dispname,
        )


if __name__ == "__main__":
    main()
