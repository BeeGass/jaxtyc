"""Parse jaxtyping annotations from Python source using the ast module."""

from __future__ import annotations

import ast
import re

from jaxtyc.types import CallSite
from jaxtyc.types import DimLocation
from jaxtyc.types import DimSpec
from jaxtyc.types import FunctionDefInfo
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


def _try_extract_tuple_return(node: ast.expr) -> list[ShapeSpec] | None:
    """Try to extract multiple ShapeSpecs from a tuple return annotation.

    Matches patterns like: tuple[Float[Array, "a b"], Float[Array, "c d"]]
    """
    if not isinstance(node, ast.Subscript):
        return None

    # Check for tuple[...] pattern
    name = _get_dtype_name(node.value)
    if name is None or name.lower() != "tuple":
        return None

    slc = node.slice
    if not isinstance(slc, ast.Tuple):
        return None

    specs: list[ShapeSpec] = []
    for elt in slc.elts:
        spec = _try_extract_jaxtyping_annotation(elt)
        if spec is None:
            return None  # All elements must be jaxtyping annotations
        specs.append(spec)

    return specs if specs else None


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

    all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    for arg in all_args:
        # Skip 'self' and 'cls' for methods
        if is_method and arg.arg in ("self", "cls"):
            continue
        if arg.annotation is not None:
            spec = _try_extract_jaxtyping_annotation(arg.annotation)
            if spec is not None:
                params[arg.arg] = spec

    return_spec: ShapeSpec | None = None
    return_specs: list[ShapeSpec] | None = None
    if node.returns is not None:
        # Try tuple return first, fallback to single return
        return_specs = _try_extract_tuple_return(node.returns)
        if return_specs is not None:
            return_spec = return_specs[0] if return_specs else None
        else:
            return_spec = _try_extract_jaxtyping_annotation(node.returns)

    # Only include functions that have at least one jaxtyping annotation
    if not params and return_spec is None:
        return

    keyword_len = 10 if isinstance(node, ast.AsyncFunctionDef) else 4
    results.append(
        FunctionShapeSpec(
            name=node.name,
            file_path=file_path,
            lineno=node.lineno,
            col_offset=node.col_offset,
            params=params,
            return_spec=return_spec,
            return_specs=return_specs,
            is_method=is_method,
            class_name=class_name,
            end_lineno=node.end_lineno or node.lineno,
            name_col_offset=node.col_offset + keyword_len,
        )
    )


def extract_call_sites(
    source: str,
    file_path: str,
    known_functions: set[str],
    include_external: bool = False,
) -> list[CallSite]:
    """Extract call sites between shape-annotated functions.

    Walks function bodies for ast.Call nodes and matches callee names
    against the set of known shape-annotated function names.

    When *include_external* is True, calls to functions **not** in
    *known_functions* are also recorded (e.g. library/external calls).
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    results: list[CallSite] = []
    _visit_body_for_calls(
        tree.body,
        file_path,
        known_functions,
        results,
        class_name=None,
        include_external=include_external,
    )
    return results


def _visit_body_for_calls(
    body: list[ast.stmt],
    file_path: str,
    known_functions: set[str],
    results: list[CallSite],
    class_name: str | None,
    include_external: bool = False,
) -> None:
    """Walk statements extracting call sites from function bodies."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            caller = f"{class_name}.{node.name}" if class_name else node.name
            _extract_calls_from_body(
                node.body,
                caller,
                file_path,
                known_functions,
                results,
                include_external=include_external,
            )
        elif isinstance(node, ast.ClassDef):
            _visit_body_for_calls(
                node.body,
                file_path,
                known_functions,
                results,
                class_name=node.name,
                include_external=include_external,
            )


def _dotted_name(node: ast.Attribute) -> str:
    """Build a dotted name from an ast.Attribute chain (e.g. jnp.lax.scan)."""
    parts = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _extract_calls_from_body(
    body: list[ast.stmt],
    caller_name: str,
    file_path: str,
    known_functions: set[str],
    results: list[CallSite],
    include_external: bool = False,
) -> None:
    """Walk a function body extracting calls to known functions."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        callee_name: str | None = None
        qualified: str | None = None
        col_start = 0
        col_end = 0
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
            col_start = node.func.col_offset
            col_end = node.func.end_col_offset or (col_start + len(callee_name))
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr
            qualified = _dotted_name(node.func)
            col_start = node.func.col_offset
            col_end = node.func.end_col_offset or (col_start + len(callee_name))
        if callee_name and (callee_name in known_functions or include_external):
            results.append(
                CallSite(
                    caller_name=caller_name,
                    callee_name=callee_name,
                    file_path=file_path,
                    lineno=node.lineno,
                    col_offset=col_start,
                    end_col_offset=col_end,
                    callee_qualified_name=qualified,
                )
            )


def extract_all_function_defs(source: str, file_path: str) -> list[FunctionDefInfo]:
    """Extract name and location for every function definition in source."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    results: list[FunctionDefInfo] = []

    def _visit(body: list[ast.stmt], class_name: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                keyword_len = 10 if isinstance(node, ast.AsyncFunctionDef) else 4
                results.append(
                    FunctionDefInfo(
                        name=node.name,
                        file_path=file_path,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        end_lineno=node.end_lineno or node.lineno,
                        name_col_offset=node.col_offset + keyword_len,
                        is_method=class_name is not None,
                        class_name=class_name,
                    )
                )
                _visit(node.body, None)
            elif isinstance(node, ast.ClassDef):
                _visit(node.body, node.name)

    _visit(tree.body, None)
    return results


def extract_dim_locations(source: str, file_path: str) -> list[DimLocation]:
    """Extract source locations for every dimension name in jaxtyping annotations."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    results: list[DimLocation] = []
    _visit_body_for_dims(tree.body, file_path, results, class_name=None)
    return results


def _visit_body_for_dims(
    body: list[ast.stmt],
    file_path: str,
    results: list[DimLocation],
    class_name: str | None,
) -> None:
    """Walk statements extracting dim locations from function annotations."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _extract_dims_from_function(node, file_path, results, class_name=class_name)
        elif isinstance(node, ast.ClassDef):
            _visit_body_for_dims(node.body, file_path, results, class_name=node.name)


def _extract_dims_from_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    results: list[DimLocation],
    class_name: str | None,
) -> None:
    """Extract DimLocation entries from a single function's annotations."""
    is_method = class_name is not None
    func_name = node.name

    all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    for arg in all_args:
        if is_method and arg.arg in ("self", "cls"):
            continue
        if arg.annotation is not None:
            _extract_dims_from_annotation(arg.annotation, arg.arg, func_name, file_path, results)

    if node.returns is not None:
        _extract_dims_from_annotation(node.returns, "__return__", func_name, file_path, results)


def _extract_dims_from_annotation(
    node: ast.expr,
    param_name: str,
    function_name: str,
    file_path: str,
    results: list[DimLocation],
) -> None:
    """Extract dim locations from a single annotation AST node."""
    if not isinstance(node, ast.Subscript):
        return

    dtype_name = _get_dtype_name(node.value)
    if dtype_name is None or dtype_name not in _DTYPE_MAP:
        return

    slc = node.slice
    if not (isinstance(slc, ast.Tuple) and len(slc.elts) == 2):
        return

    shape_node = slc.elts[1]
    if not (isinstance(shape_node, ast.Constant) and isinstance(shape_node.value, str)):
        return

    shape_str = shape_node.value
    # col_offset points to the opening quote char; content starts at +1
    string_col = shape_node.col_offset + 1
    string_line = shape_node.lineno

    for match in re.finditer(r"\S+", shape_str):
        token = match.group()
        # Skip non-named tokens
        if token == "..." or token == "_" or token.isdigit():
            continue
        # Strip leading * for variadic dims
        dim_name = token.lstrip("*")
        if not dim_name:
            continue
        # Compute column offset of the dim name (after any *)
        prefix_len = len(token) - len(dim_name)
        col_start = string_col + match.start() + prefix_len
        col_end = string_col + match.end()
        results.append(
            DimLocation(
                dim_name=dim_name,
                param_name=param_name,
                function_name=function_name,
                file_path=file_path,
                lineno=string_line,
                col_start=col_start,
                col_end=col_end,
            )
        )
