"""Dimensions module.

This is the private implementation of the dimensions module.

"""

__all__ = ("AbstractDimension", "dimension", "dimension_of")

import ast
import importlib.metadata
import re
from typing import Any, NoReturn, TypeAlias

import astropy.units as apyu
from packaging.version import Version, parse as parse_version
from plum import dispatch

import unxt_api as uapi

AbstractDimension: TypeAlias = apyu.PhysicalType

# Regex pattern to detect PEMD operators (Parentheses, Exponentiation,
# Multiplication, Division). We match: ( ) * / ** but NOT space, +, or -
_PEMD_PATTERN = re.compile(r"[()*/]|\*\*")


# ===================================================================
# Construct the dimensions


def _eval_dimension_node(node: ast.AST, /) -> AbstractDimension:  # noqa: C901
    """Recursively evaluate AST nodes into dimensions or numeric values.

    Parameters
    ----------
    node : ast.AST
        AST node to evaluate.

    Returns
    -------
    AbstractDimension
        Evaluated dimension.

    """
    if isinstance(node, ast.Expression):
        return _eval_dimension_node(node.body)

    if isinstance(node, ast.BinOp):
        left = _eval_dimension_node(node.left)

        if isinstance(node.op, ast.Pow):
            # For powers, right side must be a numeric constant
            if not isinstance(node.right, ast.Constant):
                msg = "Power exponent must be a number"
                raise TypeError(msg)
            right = node.right.value
            if not isinstance(right, int | float):
                msg = f"Power exponent must be a number, got: {type(right).__name__}"
                raise TypeError(msg)
            return left**right

        # For other operators, evaluate right side normally
        right = _eval_dimension_node(node.right)

        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right

        msg = f"Unsupported operator: {node.op.__class__.__name__}"
        raise ValueError(msg)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_dimension_node(node.operand)
        if isinstance(node.op, ast.USub):
            return operand**-1
        msg = f"Unsupported unary operator: {node.op.__class__.__name__}"
        raise ValueError(msg)

    if isinstance(node, ast.Name):
        # This is a dimension name - recursively call dimension()
        return uapi.dimension(node.id)

    if isinstance(node, ast.Constant):
        # Handle numeric constants (for exponents)
        return node.value

    msg = f"Unsupported AST node: {node.__class__.__name__}"
    raise ValueError(msg)


def _parse_dimension_string(expr: str, /) -> AbstractDimension:
    """Parse a dimension string with mathematical operations.

    Supports *, /, and ** operators following PEMDAS.

    Parameters
    ----------
    expr : str
        Mathematical expression like "length / time**2"

    Returns
    -------
    AbstractDimension
        The resulting physical type from evaluating the expression.

    Examples
    --------
    >>> _parse_dimension_string("length / time**2")  # doctest: +SKIP
    PhysicalType(...)

    """
    # Normalize whitespace
    expr = expr.strip()

    # Parse the expression into an AST
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        msg = f"Invalid dimension expression: {expr}"
        raise ValueError(msg) from e

    return _eval_dimension_node(tree)


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
def dimension(obj: str, /) -> AbstractDimension:
    """Construct dimension from a string.

    Supports simple dimension names and mathematical expressions using
    *, /, and ** operators. Dimension names can contain spaces, +, and -.

    Examples
    --------
    >>> from unxt.dims import dimension

    Simple dimension name:

    >>> dimension("length")
    PhysicalType('length')

    Mathematical expressions:

    >>> dimension("length / time")  # doctest: +SKIP
    PhysicalType('speed')

    >>> dimension("length / time**2")  # doctest: +SKIP
    PhysicalType('acceleration')

    """
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
    ...     u.dimension_of(u.quantity.BareQuantity)
    ... except ValueError as e:
    ...     print(e)
    Cannot get the dimension of <class 'unxt._src.quantity.unchecked.BareQuantity'>.

    """
    msg = f"Cannot get the dimension of {obj}."
    raise ValueError(msg)


# ===================================================================
# COMPAT

ASTROPY_LT_71 = parse_version(importlib.metadata.version("astropy")) < Version("7.1")


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
        name = " ".join(
            f"{unit}{power}" if power != 1 else unit for unit, power in ptid
        )

    elif ASTROPY_LT_71:
        name = dim._name_string_as_ordered_set().split("'")[1]  # noqa: SLF001
    else:
        name = dim._physical_type[0]  # noqa: SLF001

    return name
