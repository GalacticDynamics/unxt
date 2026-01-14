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

# Regex pattern to match parenthesized dimension names that may contain spaces
_PAREN_DIM_PATTERN = re.compile(r"\(([^()]+)\)")


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
    counter = 0

    def replace_paren_dim(match: re.Match[str], /) -> str:
        nonlocal counter
        # Strip whitespace from the captured dimension name to handle cases like
        # "( amount of substance )" where users might include extra spaces
        dim_name = match.group(1).strip()
        temp_id = f"_dim{counter}"
        dim_mapping[temp_id] = dim_name
        counter += 1
        return temp_id

    preprocessed = _PAREN_DIM_PATTERN.sub(replace_paren_dim, expr)
    return preprocessed, dim_mapping


def _eval_dimension_node(  # noqa: C901
    node: ast.AST,
    /,
    *,
    dim_mapping: dict[str, str] | None = None,
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
        Evaluated dimension.

    """
    if dim_mapping is None:
        dim_mapping = {}

    if isinstance(node, ast.Expression):
        return _eval_dimension_node(node.body, dim_mapping=dim_mapping)

    if isinstance(node, ast.BinOp):
        left = _eval_dimension_node(node.left, dim_mapping=dim_mapping)

        if isinstance(node.op, ast.Pow):
            # For powers, evaluate the exponent
            # It can be a Constant or a UnaryOp (for negative exponents like **-1)
            if isinstance(node.right, ast.Constant):
                right = node.right.value
            elif isinstance(node.right, ast.UnaryOp) and isinstance(
                node.right.op, ast.USub
            ):
                # Handle negative exponents like **-1
                if isinstance(node.right.operand, ast.Constant):
                    right = -node.right.operand.value
                else:
                    msg = "Power exponent must be a number"
                    raise TypeError(msg)
            else:
                msg = "Power exponent must be a number"
                raise TypeError(msg)

            if not isinstance(right, int | float):
                msg = f"Power exponent must be a number, got: {type(right).__name__}"
                raise TypeError(msg)
            return left**right

        # For other operators, evaluate right side normally
        right = _eval_dimension_node(node.right, dim_mapping=dim_mapping)

        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right

        msg = f"Unsupported operator: {node.op.__class__.__name__}"
        raise ValueError(msg)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_dimension_node(node.operand, dim_mapping=dim_mapping)
        if isinstance(node.op, ast.USub):
            return operand**-1
        msg = f"Unsupported unary operator: {node.op.__class__.__name__}"
        raise ValueError(msg)

    if isinstance(node, ast.Name):
        # Check if this is a temporary identifier that maps to a dimension name
        dim_name = dim_mapping.get(node.id, node.id)
        # This is a dimension name - recursively call dimension()
        return uapi.dimension(dim_name)

    if isinstance(node, ast.Constant):
        # Handle numeric constants (for exponents)
        return node.value

    msg = f"Unsupported AST node: {node.__class__.__name__}"
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
