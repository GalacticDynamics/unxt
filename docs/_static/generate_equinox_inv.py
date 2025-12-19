#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sphobjinv>=2.3",
# ]
# ///
"""Generate a minimal intersphinx inventory for equinox.

Writes: equinox.inv (next to this script, i.e. docs/_static/equinox.inv)

Usage:
    ./docs/_static/generate_equinox_inv.py
    uv run docs/_static/generate_equinox_inv.py
"""

import logging
import pathlib
from collections.abc import Iterable
from typing import NamedTuple

import sphobjinv as soi

logger = logging.getLogger(__name__)

# ---- Configure your entries here -------------------------------------------------
# name: fully-qualified reference you will use in Sphinx roles, e.g.
#       :py:class:`equinox.Array`
# role: Sphinx role to attach (e.g. "class", "data", "func", "attr", ...);
#       domain is "py"
# uri:  relative URL from the equinox docs root to the target page/anchor
# disp: optional display text (link label); None -> use the shortname from `name`


class Entry(NamedTuple):
    """Intersphinx inventory entry for a equinox type.

    Attributes
    ----------
    name
        Fully-qualified name (e.g., "equinox.Array") or unqualified alias
        (e.g., "Array").
    role
        Sphinx role type: "class", "data", "func", "attr", etc.
    uri
        Relative URL from equinox docs root to the target anchor.
    disp
        Display text for the link. If None, uses the shortname from `name`.

    """

    name: str
    role: str
    uri: str
    disp: str | None = None


ENTRIES: tuple[Entry, ...] = (
    # Qualified
    Entry(
        "equinox.equinox._module._better_abstract.AbstractVar",
        "class",
        "api/dataclasses/#abstractvar",
        "AbstractVar",
    ),
)


# ----------------------------------------------------------------------------------


def build_inventory(
    entries: Iterable[Entry],
    project: str = "equinox",
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
    out_path = here / "equinox.inv"

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
