"""An example type for the engine's own documentation -- package side.

The engine's docstrings need a type to demonstrate on, and it cannot be a
quantity: this module travels to the standalone package, which `unxt` depends
on, so an example reaching back for `unxt.Quantity` would be circular.

`Version` is that type. A version number has parts and roles -- a major, a
minor, and a separator between them -- which is everything the engine needs to
show, and no units anywhere.

Importing this module registers a `pparts` method for `Version`, so it is
imported for its side effect as much as for the name. That is sound here
because the type is defined alongside the method; a module registering
behaviour for a type it does not own is the pattern that caused #1008.

Like `engine` and `generic`, this imports only the standard library,
`wadler_lindig`, `plum` and its sibling package-side modules.
"""

__all__ = ("Point", "Term", "Version")

import dataclasses

from .engine import PPart, ReprMixin, doc_part, pparts
from .generic import pvalue


@dataclasses.dataclass(repr=False)
class Version(ReprMixin):
    """A two-part version number, rendered through `pparts`.

    ``repr=False`` on the decorator is load-bearing: a generated ``__repr__``
    would shadow the one `ReprMixin` provides, and the shadowing is silent.

    Examples
    --------
    >>> from unxt._src.fmt.examples import Version

    >>> Version(2, 1)
    2.1

    The fragments carry roles, so each markup wraps them its own way:

    >>> f"{Version(2, 1):html}"
    '<span>2</span>.<span>1</span>'

    >>> f"{Version(2, 1):latex}"
    '$2.1$'

    """

    major: int
    minor: int

    __repr_spec__ = "bare"


@pparts.dispatch  # type: ignore[misc]
def pparts_version(
    obj: Version, /, *, markup: str = "text", **kw: object
) -> tuple[PPart, ...]:
    """Decompose a version into its parts.

    The separator is its own fragment with the ``sep`` kind, so a markup can
    style it differently from the numbers -- and so the ``sep`` axis leaves it
    alone, which is why ``mul`` renders ``2.1`` rather than ``2 * 1``.

    Examples
    --------
    >>> from unxt._src.fmt import pparts
    >>> from unxt._src.fmt.examples import Version

    >>> pparts(Version(2, 1))
    (PPart(role='major', text='2', kind='content', doc=None),
     PPart(role='dot', text='.', kind='sep', doc=None),
     PPart(role='minor', text='1', kind='content', doc=None))

    """
    return (
        PPart("major", str(obj.major)),
        PPart("dot", ".", "sep"),
        PPart("minor", str(obj.minor)),
    )


@dataclasses.dataclass(repr=False)
class Term(ReprMixin):
    """A coefficient and a symbol, joined by a product separator.

    The engine is built for values that read as ``<payload> <operator>
    <label>``, and a quantity is only the most familiar one. `Term` is that
    shape without units: it carries a ``mul`` separator, so it can demonstrate
    the ``sep`` axis, and its coefficient may be a sequence, so it can
    demonstrate a line break at a narrow width.
    """

    coeff: object
    symbol: str

    __repr_spec__ = "mul"


@pparts.dispatch  # type: ignore[misc]
def pparts_term(
    obj: Term, /, *, markup: str = "text", **kw: object
) -> tuple[PPart, ...]:
    """Decompose a term into coefficient, operator and symbol."""
    return (
        doc_part("coeff", pvalue(obj.coeff, markup=markup, **kw)),
        PPart("mul", " * ", "sep"),
        PPart("symbol", obj.symbol),
    )


@dataclasses.dataclass
class Point:
    """A plain dataclass: no `ReprMixin`, no registered `pparts`.

    `call` layout renders through an object's own ``__pdoc__``, and a
    dataclass gets a usable one from `wadler_lindig` for free. So this type
    demonstrates that the ``call`` form reconstructs a value without the type
    having told the engine anything -- which is the point of routing ``repr``
    that way rather than building it from fragments.
    """

    x: float
    y: float
