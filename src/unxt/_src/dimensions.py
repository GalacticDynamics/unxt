"""Dimensions module.

This is the private implementation of the dimensions module.

"""

__all__ = ("AbstractDimension", "dimension", "dimension_of")

import ast
import functools as ft
import operator
import re
from typing import Any, Final, NoReturn, TypeAlias

import astropy.units as apyu
from plum import dispatch

import unxt_api as uapi

AbstractDimension: TypeAlias = apyu.PhysicalType

# Regex pattern to detect PEMD operators (Parentheses, Exponentiation,
# Multiplication, Division). We match: ( ) * / ** but NOT space, +, or -
_PEMD_PATTERN = re.compile(r"[()*/]|\*\*")

# Regex pattern to match parenthesized dimension names that may contain spaces
_PAREN_DIM_PATTERN = re.compile(r"\(([^()]+)\)")

#: The binary operators dimensions support. ``**`` is handled separately: its
#: right operand is a plain number, not a dimension. ``+``/``-`` are absent by
#: design -- dimensions are invariant under addition, so those symbols belong to
#: dimension *names* ("electric-dipole moment"), not to the grammar.
_BINARY_OPS: Final = {ast.Mult: operator.mul, ast.Div: operator.truediv}


# ===================================================================
# Construct the dimensions


def _preprocess_dimension_string(expr: str, /) -> tuple[str, dict[str, str]]:
    """Preprocess dimension string to handle multi-word dimension names.

    Converts (dimension name) to _dimN for parsing, then stores the mapping.

    Parameters
    ----------
    expr : str
        The expression string that may contain parenthesized dimension names.

    Returns
    -------
    str
        The preprocessed expression with valid Python identifiers.
    dict[str, str]
        Mapping from temporary identifiers to original dimension names.

    """
    dim_mapping: dict[str, str] = {}

    def replace_paren_dim(match: re.Match[str], /) -> str:
        # Strip whitespace from the captured dimension name to handle cases like
        # "( amount of substance )" where users might include extra spaces.
        temp_id = f"_dim{len(dim_mapping)}"
        dim_mapping[temp_id] = match.group(1).strip()
        return temp_id

    return _PAREN_DIM_PATTERN.sub(replace_paren_dim, expr), dim_mapping


def _eval_exponent(node: ast.AST, /) -> int | float:
    """Evaluate a ``**`` exponent, which must be a plain (possibly negative) number.

    Unlike the left operand, the exponent is *not* a dimension -- ``length**2``
    means "length times itself", so ``2`` must reduce to a number here rather
    than route back through `_eval_dimension_node`.
    """
    # A negative exponent (``length**-1``) parses as USub over the constant.
    # Peel the sign off first so the checks below see the bare operand and can
    # report the same ``got: <type>`` detail either way.
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node, sign = node.operand, -1

    match node:
        case ast.Constant(value=int() | float() as value):
            return sign * value
        case ast.Constant(value=value):
            msg = f"Power exponent must be a number, got: {type(value).__name__}"
            raise TypeError(msg)
        case _:
            msg = "Power exponent must be a number"
            raise TypeError(msg)


def _eval_dimension_node(
    node: ast.AST, /, *, dim_mapping: dict[str, str] | None = None
) -> AbstractDimension:
    """Recursively evaluate AST nodes into dimensions or numeric values.

    Parameters
    ----------
    node : ast.AST
        AST node to evaluate.
    dim_mapping : dict[str, str] | None
        Mapping from temporary identifiers to original dimension names.

    Returns
    -------
    AbstractDimension
        Evaluated dimension, or a bare numeric constant when the node is a
        numeric factor (e.g. the ``2`` in ``"2 * length"``).

    """
    mapping = {} if dim_mapping is None else dim_mapping

    match node:
        case ast.Expression(body=body):
            return _eval_dimension_node(body, dim_mapping=mapping)

        case ast.BinOp(op=ast.Pow(), left=left, right=right):
            return _eval_dimension_node(left, dim_mapping=mapping) ** _eval_exponent(
                right
            )

        case ast.BinOp(op=op, left=left, right=right) if type(op) in _BINARY_OPS:
            return _BINARY_OPS[type(op)](
                _eval_dimension_node(left, dim_mapping=mapping),
                _eval_dimension_node(right, dim_mapping=mapping),
            )

        case ast.BinOp(op=op):
            msg = f"Unsupported operator: {type(op).__name__}"
            raise ValueError(msg)

        # A negative exponent (``length**-1``) is consumed by the ``Pow`` branch
        # above, so any ``UnaryOp`` reaching here is a standalone sign applied to
        # a dimension. Dimensions are invariant under negation, so reject it
        # rather than silently treating ``-(length)`` as ``1/length``.
        case ast.UnaryOp(op=ast.USub() | ast.UAdd()):
            msg = (
                "Unary '+'/'-' are not supported on dimensions; they are "
                "invariant under negation."
            )
            raise ValueError(msg)

        case ast.UnaryOp(op=op):
            msg = f"Unsupported unary operator: {type(op).__name__}"
            raise ValueError(msg)

        case ast.Name(id=name):
            # A temporary identifier maps back to its multi-word dimension name.
            return uapi.dimension(mapping.get(name, name))

        # A bare numeric factor, e.g. ``"2 * length"``. Returned as-is so the
        # surrounding operator applies it to the dimension.
        case ast.Constant(value=value):
            return value

        case _:
            msg = f"Unsupported AST node: {type(node).__name__}"
            raise ValueError(msg)


def _parse_dimension_string(expr: str, /) -> AbstractDimension:
    """Parse a dimension string with mathematical operations.

    Supports *, /, and ** operators following PEMDAS. Dimension names can
    be parenthesized and may contain spaces, e.g., "(amount of substance)".

    Parameters
    ----------
    expr : str
        Mathematical expression like "length / time**2" or
        "(amount of substance) / (time)"

    Returns
    -------
    AbstractDimension
        The resulting physical type from evaluating the expression.

    Examples
    --------
    >>> _parse_dimension_string("length / time**2")
    PhysicalType(...)

    """
    # Normalize whitespace
    expr = expr.strip()

    # Preprocess to handle multi-word dimension names in parentheses
    preprocessed, dim_mapping = _preprocess_dimension_string(expr)

    # Parse the expression into an AST
    try:
        tree = ast.parse(preprocessed, mode="eval")
    except SyntaxError as e:
        msg = f"Invalid dimension expression: {expr}"
        raise ValueError(msg) from e

    return _eval_dimension_node(tree, dim_mapping=dim_mapping)


@dispatch
def dimension(obj: AbstractDimension, /) -> AbstractDimension:
    """Construct dimension from a dimension object.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> length = apyu.get_physical_type("length")
    >>> length
    PhysicalType('length')

    >>> u.dimension(length) is length
    True

    """
    return obj


@dispatch
# Parsing a dimension string is pure, and astropy hands back a registry
# singleton -- ``dimension("length/time") is dimension("length/time")`` was
# already True before this cache, so memoizing changes no observable
# behaviour, it only skips redoing the parse. Bounded because the keys come
# from user input; the working vocabulary of dimension strings is tiny.
@ft.lru_cache(maxsize=256)
def dimension(obj: str, /) -> AbstractDimension:
    """Construct dimension from a string.

    The string can be:
    1. A simple dimension name (e.g., "length", "time", "mass")
    2. A multi-word dimension name (e.g., "amount of substance", "absement")
    3. A mathematical expression using *, /, and ** operators

    Mathematical Expressions:

    Expressions are evaluated using operator precedence (PEMDAS):
    - ** (exponentiation, highest precedence)
    - * and / (multiplication and division, equal precedence, left-to-right)

    Parentheses are supported for grouping and for dimension names with spaces.

    Operators Supported:
    - `*` : Multiplication (e.g., "length * time")
    - `/` : Division (e.g., "length / time")
    - `**` : Exponentiation (e.g., "length**2")

    Unsupported Operators:
    - `+` and `-` are NOT supported as operators since dimensions are invariant
      under addition and subtraction. They are treated as part of dimension names.

    Rules for Dimension Names in Expressions:
    - Single-word names don't need parentheses: "length * time"
    - Multi-word names MUST be parenthesized: "(amount of substance) * time"
    - Parenthesized single-word names are allowed: "(length) / (time)"
    - Whitespace is flexible: "length / time", "length/time", "length / time**2"

    Examples
    --------
    >>> from unxt.dims import dimension

    **Simple dimension names:**

    >>> dimension("length")
    PhysicalType('length')

    >>> dimension("time")
    PhysicalType('time')

    >>> dimension("mass")
    PhysicalType('mass')

    **Multi-word dimension names:**

    >>> dimension("amount of substance")
    PhysicalType('amount of substance')

    **Mathematical expressions with single-word names:**

    >>> dimension("length / time")
    PhysicalType({'speed', 'velocity'})

    >>> dimension("length**2")
    PhysicalType('area')

    >>> dimension("length * mass / time**2")
    PhysicalType('force')

    **Parenthesized expressions:**

    >>> dimension("(length) / (time)")
    PhysicalType({'speed', 'velocity'})

    **Expressions with multi-word dimension names:**

    >>> dimension("(amount of substance) / (time)")
    PhysicalType('catalytic activity')

    **Mixed expressions (multi-word with parentheses, single-word without):**

    >>> dimension("length * (amount of substance)")
    PhysicalType('unknown')

    >>> dimension("(absement) / (time)")
    PhysicalType('length')

    See Also
    --------
    dimension_of : Get the dimension of an object
    unxt.units : Unit specifications can also use dimension expressions

    """
    # Strip surrounding whitespace so both the operator path (which strips) and
    # the simple-name path treat e.g. " length " the same as "length".
    obj = obj.strip()

    # Check if the string contains PEMD operators using regex
    # We only consider (), *, /, ** as operators - not space, +, or -
    if _PEMD_PATTERN.search(obj):
        return _parse_dimension_string(obj)

    # Simple dimension name - use astropy directly
    return apyu.get_physical_type(obj)


# ===================================================================
# Get the dimension


@dispatch
def dimension_of(obj: Any, /) -> None:
    """Most objects have no dimension.

    Examples
    --------
    >>> from unxt.dims import dimension_of

    >>> print(dimension_of(1))
    None

    >>> print(dimension_of("length"))
    None

    """
    return None  # noqa: RET501


@dispatch
def dimension_of(obj: AbstractDimension, /) -> AbstractDimension:
    """Return the dimension of the given units.

    Examples
    --------
    >>> from unxt.dims import dimension, dimension_of

    >>> dimension_of(dimension("length"))
    PhysicalType('length')

    """
    return obj


@dispatch
def dimension_of(obj: type, /) -> NoReturn:
    """Get the dimension of a type.

    Examples
    --------
    >>> import unxt as u

    >>> try:
    ...     u.dimension_of(u.quantity.Quantity)
    ... except ValueError as e:
    ...     print(e)
    Cannot get the dimension of <class 'unxt._src.quantity.quantity.Quantity'>.

    """
    msg = f"Cannot get the dimension of {obj}."
    raise ValueError(msg)


# ===================================================================
# COMPAT


@dispatch
def name_of(dim: AbstractDimension, /) -> str:
    """Name of a dimension.

    Examples
    --------
    >>> import unxt as u

    >>> name_of(u.dimension("length"))
    'length'

    >>> name_of(u.dimension("speed"))
    'speed'

    >>> name_of(u.dimension("mass density"))
    'mass density'

    """
    if dim == "unknown":
        ptid = dim._unit._physical_type_id  # noqa: SLF001
        return " ".join(
            f"{unit}{power}" if power != 1 else unit for unit, power in ptid
        )

    return dim._physical_type[0]  # noqa: SLF001
