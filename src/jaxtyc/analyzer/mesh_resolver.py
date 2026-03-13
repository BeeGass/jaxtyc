"""Infer mesh shape and axis rules from Python AST."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class MeshInfo:
    """Physical mesh shape and logical-to-physical axis mapping."""

    mesh: dict[str, int] = field(default_factory=dict)
    axis_rules: dict[str, str] = field(default_factory=dict)


def resolve_mesh(tree: ast.Module) -> MeshInfo:
    """Walk AST to find mesh definitions and axis rules.

    Detects:
    - ``jax.make_mesh((sizes...), (names...))``
    - ``AbstractMesh((sizes...), (names...))``
    - ``nnx.logical_axis_rules([(logical, physical), ...])``

    Args:
        tree: Parsed AST module.

    Returns:
        MeshInfo with inferred mesh shape and axis rules.
    """
    mesh: dict[str, int] = {}
    axis_rules: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = _get_call_name(node)
        if func_name in (
            "jax.make_mesh",
            "make_mesh",
            "AbstractMesh",
            "jax.sharding.AbstractMesh",
        ):
            extracted = _extract_mesh_args(node)
            if extracted:
                mesh.update(extracted)
        elif func_name in (
            "nnx.logical_axis_rules",
            "logical_axis_rules",
            "flax.nnx.logical_axis_rules",
        ):
            extracted = _extract_axis_rules(node)
            if extracted:
                axis_rules.update(extracted)

    return MeshInfo(mesh=mesh, axis_rules=axis_rules)


def _get_call_name(node: ast.Call) -> str:
    """Extract the dotted name of a function call."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        current: ast.expr = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _extract_mesh_args(node: ast.Call) -> dict[str, int] | None:
    """Extract (sizes, names) from make_mesh/AbstractMesh call."""
    if len(node.args) < 2:
        return None
    sizes_node, names_node = node.args[0], node.args[1]
    sizes = _extract_int_tuple(sizes_node)
    names = _extract_str_tuple(names_node)
    if sizes and names and len(sizes) == len(names):
        return dict(zip(names, sizes, strict=True))
    return None


def _extract_axis_rules(node: ast.Call) -> dict[str, str] | None:
    """Extract [(logical, physical), ...] from logical_axis_rules call."""
    if not node.args:
        return None
    arg = node.args[0]
    if not isinstance(arg, (ast.List, ast.Tuple)):
        return None
    rules: dict[str, str] = {}
    for elt in arg.elts:
        if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
            logical = _extract_str(elt.elts[0])
            physical = _extract_str(elt.elts[1])
            if logical and physical:
                rules[logical] = physical
    return rules


def _extract_int_tuple(node: ast.expr) -> list[int] | None:
    """Extract a tuple of integer constants from an AST node."""
    if not isinstance(node, ast.Tuple):
        return None
    result: list[int] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
            result.append(elt.value)
        else:
            return None
    return result


def _extract_str_tuple(node: ast.expr) -> list[str] | None:
    """Extract a tuple of string constants from an AST node."""
    if not isinstance(node, ast.Tuple):
        return None
    result: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            result.append(elt.value)
        else:
            return None
    return result


def _extract_str(node: ast.expr) -> str | None:
    """Extract a string constant from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
