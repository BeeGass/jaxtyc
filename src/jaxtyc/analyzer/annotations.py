"""Parse jaxtyping annotations from Python source using the ast module."""

from __future__ import annotations

import ast

from jaxtyc.types import DimSpec
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import ShapeSpec

# jaxtyping dtype class names -> dtype strings
_DTYPE_MAP: dict[str, str] = {
    "Float": "float32",
    "Float16": "float16",
    "Float32": "float32",
    "Float64": "float64",
    "BFloat16": "bfloat16",
    "Int": "int",
    "Int8": "int8",
    "Int16": "int16",
    "Int32": "int32",
    "Int64": "int64",
    "UInt": "uint",
    "UInt8": "uint8",
    "UInt16": "uint16",
    "UInt32": "uint32",
    "UInt64": "uint64",
    "Bool": "bool",
    "Complex": "complex",
    "Complex64": "complex64",
    "Complex128": "complex128",
    "Num": "numeric",
    "Shaped": "shaped",
    "Key": "key",
    "Scalar": "scalar",
}


def parse_shape_string(shape_str: str, dtype: str) -> ShapeSpec:
    """Parse a jaxtyping shape string into a ShapeSpec.

    Args:
        shape_str: Raw shape string from the annotation (e.g. "batch seq d_model").
        dtype: Resolved dtype string (e.g. "float32").

    Returns:
        ShapeSpec with parsed dimension specs, dtype, and scalar/any-shape flags.

    Example:
        >>> parse_shape_string("batch seq d_model", "float32")
        ShapeSpec(dims=(DimSpec(kind='named', name='batch', size=None), ...), ...)
        >>> parse_shape_string("", "float32").is_scalar
        True
        >>> parse_shape_string("...", "float32").is_any_shape
        True
    """
    stripped = shape_str.strip()

    if stripped == "":
        return ShapeSpec(dims=(), dtype=dtype, is_scalar=True)

    if stripped == "...":
        return ShapeSpec(dims=(), dtype=dtype, is_any_shape=True)

    tokens = stripped.split()
    dims: list[DimSpec] = []

    for token in tokens:
        if token == "...":
            dims.append(DimSpec(kind="ellipsis"))
        elif token == "_":
            dims.append(DimSpec(kind="anonymous"))
        elif token.startswith("*"):
            dims.append(DimSpec(kind="variadic", name=token[1:]))
        elif token.isdigit():
            dims.append(DimSpec(kind="fixed", size=int(token)))
        else:
            dims.append(DimSpec(kind="named", name=token))

    return ShapeSpec(dims=tuple(dims), dtype=dtype)


def _try_extract_jaxtyping_annotation(node: ast.expr) -> ShapeSpec | None:
    """Try to extract a ShapeSpec from an AST annotation node.

    Looks for patterns like Float[Array, "batch seq d_model"].
    """
    if not isinstance(node, ast.Subscript):
        return None

    # The value should be a Name (e.g., Float) or Attribute (e.g., jaxtyping.Float)
    dtype_name = _get_dtype_name(node.value)
    if dtype_name is None or dtype_name not in _DTYPE_MAP:
        return None

    dtype = _DTYPE_MAP[dtype_name]

    # The slice should be a Tuple with (ArrayType, shape_string)
    slc = node.slice
    if isinstance(slc, ast.Tuple) and len(slc.elts) == 2:
        shape_node = slc.elts[1]
        if isinstance(shape_node, ast.Constant) and isinstance(shape_node.value, str):
            return parse_shape_string(shape_node.value, dtype)
        # Python Ellipsis literal: Float[Array, ...]
        if isinstance(shape_node, ast.Constant) and shape_node.value is Ellipsis:
            return ShapeSpec(dims=(), dtype=dtype, is_any_shape=True)

    return None


def _get_dtype_name(node: ast.expr) -> str | None:
    """Extract the dtype class name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def extract_function_specs(source: str, file_path: str) -> list[FunctionShapeSpec]:
    """Extract FunctionShapeSpecs from Python source code.

    Parses the source with ``ast`` and finds functions whose parameters or
    return types use jaxtyping annotations. Class methods are detected
    automatically (``self``/``cls`` parameters are skipped).

    Args:
        source: Python source code as a string.
        file_path: File path used for error reporting and stored in each
            FunctionShapeSpec.

    Returns:
        List of FunctionShapeSpecs for every function with at least one
        jaxtyping annotation. Empty list if the source has syntax errors or
        contains no annotated functions.
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    results: list[FunctionShapeSpec] = []
    _visit_body(tree.body, file_path, results, class_name=None)
    return results


def _visit_body(
    body: list[ast.stmt],
    file_path: str,
    results: list[FunctionShapeSpec],
    class_name: str | None,
) -> None:
    """Walk a list of statements, extracting function specs without double-visiting."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _extract_from_function(node, file_path, results, class_name=class_name)
        elif isinstance(node, ast.ClassDef):
            _visit_body(node.body, file_path, results, class_name=node.name)


def _extract_from_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    results: list[FunctionShapeSpec],
    class_name: str | None,
) -> None:
    """Extract shape specs from a single function definition."""
    is_method = class_name is not None
    params: dict[str, ShapeSpec] = {}

    for arg in node.args.args:
        # Skip 'self' and 'cls' for methods
        if is_method and arg.arg in ("self", "cls"):
            continue
        if arg.annotation is not None:
            spec = _try_extract_jaxtyping_annotation(arg.annotation)
            if spec is not None:
                params[arg.arg] = spec

    return_spec: ShapeSpec | None = None
    if node.returns is not None:
        return_spec = _try_extract_jaxtyping_annotation(node.returns)

    # Only include functions that have at least one jaxtyping annotation
    if not params and return_spec is None:
        return

    results.append(
        FunctionShapeSpec(
            name=node.name,
            file_path=file_path,
            lineno=node.lineno,
            col_offset=node.col_offset,
            params=params,
            return_spec=return_spec,
            is_method=is_method,
            class_name=class_name,
        )
    )
