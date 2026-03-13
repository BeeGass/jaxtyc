# Internals

## Symbolic Dimensions

### The Problem

If two named dimensions happen to share the same concrete size, shape bugs become invisible. Consider multi-head attention where `batch=4` and `head_dim=4`. A transpose that swaps `batch` and `head_dim` produces the same shape tuple `(4, ...)` -- the checker cannot distinguish the two dimensions and reports no error.

### The Solution

Each named dimension is assigned a unique symbolic dimension object via `jax.export.symbolic_shape`:

```python
(batch_dim,) = jax.export.symbolic_shape("batch", scope=scope)
(seq_dim,)   = jax.export.symbolic_shape("seq", scope=scope)
```

Symbolic dimensions (`_DimExpr` objects) are distinct by construction. JAX propagates them through operations, and error messages use the original dimension names directly. No two distinct named dims can ever be confused.

### Implementation

`DimEnv` maintains a mapping from dimension names to symbolic `_DimExpr` objects, all sharing a single `SymbolicScope`:

```
DimEnv._dims: {"batch": <batch>, "heads": <heads>, "seq": <seq>, "head_dim": <head_dim>}
```

**Reverse mapping** is trivial: `str(dim)` returns the original name. `shape_to_names(shape)` maps a full shape tuple back to dimension names for diagnostics and LSP display.

**Special dimension kinds:**

- **Fixed dims** (e.g., `Float[Array, "batch 4 d_model"]`): the literal `4` is used directly as a plain int. Fixed dims represent known constants.
- **Variadic dims** (`*batch`): expanded to 2 symbolic dims under internal names `_var_batch_0` and `_var_batch_1`. Two dims are needed because variadic dims represent an unknown number of leading dimensions (minimum 2 for shape distinguishability).
- **Ellipsis dims** (`...`): expanded to 2 symbolic dims under `_ellipsis_0` and `_ellipsis_1`, same reasoning as variadic.
- **Anonymous dims** (`_`): each gets a unique symbolic dim under `_anon_{counter}`.

### Concrete Fallback for Modules

NNX and equinox module constructors require plain `int` arguments (symbolic `_DimExpr` objects are not accepted). `DimEnv` provides a parallel `get_concrete_size(name)` method that assigns unique odd integers starting at 101. The pipeline uses `make_concrete_shape()` for module construction and `make_shape()` (symbolic) for standalone function tracing.

### Synthetic Name Cleanup

Internal names (`_ellipsis_0`, `_var_batch_1`, `_anon_3`) leak into tracer output. The `format_named_shape()` utility in `_util.py` collapses these for user-facing display:

| Internal | Display |
|----------|---------|
| `_ellipsis_0, _ellipsis_1` | `...(0, 1)` |
| `_var_batch_0, _var_batch_1` | `*batch` |
| `_anon_N` | `_` |

A shared `DimEnv` is created per file, so the same dimension name always maps to the same symbolic dim across all functions in the file. This enables cross-function consistency checking -- if `encode` and `decode` both use a dimension named `hidden`, it maps to the same dim in both, so the checker can verify that outputs match across call boundaries.

---

## jax.eval_shape Tracing

### How It Works

`jax.eval_shape(fn, *args)` runs a function with abstract inputs (`ShapeDtypeStruct`) that carry only shape and dtype metadata -- no actual array data is allocated and no computation is performed. JAX propagates shapes through every operation in the function body and returns the output's shape/dtype.

### Input Construction

For each annotated parameter, a `ShapeDtypeStruct` is built from the `ShapeSpec`:

```python
# Annotation: q: Float[Array, "batch heads seq head_dim"]
# DimEnv creates symbolic dims for each name
ShapeDtypeStruct(shape=(batch, heads, seq, head_dim), dtype=float32)
```

When sharding annotations are present (e.g., `"batch|dp seq|None d_model|mp"`), `_build_sharded_abstract_input()` constructs inputs with `NamedSharding` and concrete sizes (divisible by the mesh partition) instead of symbolic dims. Logical axis names are resolved through `axis_rules` before building the `PartitionSpec`.

The dtype is resolved from the jaxtyping class name (`Float` -> `float32`, `BFloat16` -> `bfloat16`, etc.) via `_DTYPE_MAP`.

Parameters annotated with `Float[Array, ...]` (ellipsis literal, meaning "any shape") are skipped -- there is no shape to construct.

### Mesh Context

When any parameter has sharding annotations, `trace_function()` wraps the `eval_shape` call in a mesh context:

```python
abstract_mesh = _build_abstract_mesh(mesh_config)  # AxisType.Explicit
with jax._src.mesh.use_abstract_mesh(abstract_mesh):
    output_struct = jax.eval_shape(wrapper, **abstract_inputs)
```

This enables JAX to propagate sharding through operations. The same mesh context wraps the `make_jaxpr` call for intermediate extraction.

If sharded tracing fails (e.g., operations without explicit sharding rules like scatter or advanced indexing), the tracer retries without sharding via `_trace_fallback_unsharded()` and sets `TraceResult.sharding_fallback_reason` to indicate the fallback.

### Output Extraction

The return value of `eval_shape` may be:

- A single `ShapeDtypeStruct` -- shape and dtype are read directly.
- A PyTree of structs (tuple, dict, nested dataclass) -- `jax.tree.leaves()` extracts all leaves. For tuple return annotations (`tuple[Float[...], Float[...]]`), each leaf is checked element-by-element against its corresponding annotation. If the element count differs, a `return-count-mismatch` diagnostic is emitted.

### Error Handling

If `eval_shape` raises (e.g., incompatible matmul dimensions, unsupported Python control flow), the exception message is captured and emitted as a `trace-error` diagnostic. The function is not re-traced.

### Flax NNX and Equinox Support

For methods on NNX modules, the pipeline constructs a concrete module instance using `DimEnv.get_concrete_size()` for dimension-derived constructor args, then splits it via `nnx.split/merge` and traces the bound method with `jax.eval_shape`. When the function has sharding annotations and a mesh config, the `eval_shape` call is wrapped in a mesh context for sharding propagation.

For Equinox modules, `jax.eval_shape` traces bound methods on concrete module instances similarly, also with optional mesh context for sharded tracing.

### make_jaxpr for Intermediates

After `eval_shape` succeeds, `jax.make_jaxpr` is run separately to produce a Jaxpr (JAX's intermediate representation). Each equation in the Jaxpr represents a single primitive operation and carries:

- Output variable(s) with abstract values (shape + dtype)
- A `source_info` with traceback frames

The intermediates are extracted from equation output variables and stored as `IntermediateShape` objects for hover and the `trace` command.

If `make_jaxpr` fails (which can happen for functions that use side effects or non-functionally-pure constructs), the failure is silently swallowed -- `eval_shape` results are still valid.

---

## Source Mapping

### The Problem

Jaxpr equations correspond to low-level JAX primitives (`dot_general`, `transpose`, `broadcast_in_dim`), not user source lines. To show shapes at the right line in an editor, we need to map each equation back to the user's source code.

### Frame Filtering

Each jaxpr equation has a `source_info.traceback.frames` list -- a stack of `(file_name, line_num)` pairs captured during tracing. These frames include both user code and JAX internals:

```
frame 0: jax/_src/lax/lax.py:1234        (JAX internal)
frame 1: jax/_src/numpy/linalg.py:567    (JAX internal)
frame 2: /home/user/model.py:42          (user code)
frame 3: /home/user/model.py:38          (user code - caller)
```

The source mapper walks frames in reverse order and returns the first frame whose path does **not** contain any of the JAX-internal markers:

- `jax/_src/`
- `jaxlib/`
- `site-packages/jax`
- `site-packages/jaxlib`

The last non-internal frame (frame 2 above) gives the most specific user source location -- the line that directly caused the JAX operation.

### LSP Integration

- **Hover**: when the cursor is on a line, all `IntermediateShape` objects with a matching `source_line` are collected. The hover popup shows each operation's name, output shape (with named dimensions), and dtype. Hover also works on dimension names (showing symbolic size, all usages) and function names (showing full shape signature).
- **CodeLens**: for each function with a successful trace, a virtual annotation is rendered above the function definition showing the traced input parameter shapes and output shape.
- **trace command**: the CLI `jaxtyc trace file.py::function_name` prints all intermediates with their source lines, giving a step-by-step view of shape propagation through the function.

!!! note "Column limitation"
    JAX's traceback frames do not expose column information. The `source_col` is always 0. This means hover activates for the entire line, not a specific expression within the line.

---

## Cross-Function Shape Propagation

### How It Works

After all functions in a file are individually traced and checked, the pipeline performs cross-function analysis:

1. **Call site extraction**: `extract_call_sites(source, file_path, known_functions)` walks the AST looking for `ast.Call` nodes that reference known annotated functions. Each match produces a `CallSite` with the caller name, callee name, and source location.

2. **Consistency checking**: For each call site, `check_call_site()` compares the callee's annotated return shape against its traced output shape. If they differ, a `cross-function-mismatch` diagnostic is emitted at the call site location, showing both the annotated and actual return shapes.

This catches a common class of bugs where a function's annotation is wrong (the function works correctly but claims the wrong output shape), and downstream callers rely on the incorrect annotation.

### Shared DimEnv

Cross-function checking relies on the shared-per-file `DimEnv`. Because all functions in a file share the same symbolic dim assignments, dimension name `hidden` in function `encode` maps to the same symbolic object as `hidden` in function `decode`. This makes shape comparisons across function boundaries meaningful.

---

## Inline Suppressions

### Syntax

```python
x = fn(y)  # jaxtyc: ignore              -- suppress all rules
x = fn(y)  # jaxtyc: ignore[shape-mismatch]  -- suppress one rule
x = fn(y)  # jaxtyc: ignore[rule1, rule2]    -- suppress multiple rules
```

### Matching Logic

`extract_suppressions(source)` scans all comments in the source for the `jaxtyc: ignore` pattern and returns a list of `SuppressionComment` objects, each with a line number and a `frozenset` of rule names (empty means suppress all).

`filter_inline_suppressions(diagnostics, suppressions)` removes any diagnostic whose line matches a suppression comment on the same line or the line immediately before it. The "line before" allowance handles multi-line function signatures where the diagnostic appears on the `def` line but the suppression comment is placed above.

If a suppression specifies rule names, only diagnostics with matching `rule` values are suppressed. An empty rule set suppresses all diagnostics on that line.

---

## Sharding Validation

### Overview

jaxtyc validates sharding annotations at three levels:

1. **Annotation-level** (`check_annotation_sharding`): Static checks on the function signature
2. **Trace-level** (`check_sharding`, `check_sharding_propagation`): Checks using traced sharding info from jaxpr
3. **Mesh-level** (`check_mesh_axes`): Validates axis references against the mesh config

### Annotation Parsing

The pipe syntax (`dim|axis`) is parsed in `parse_shape_string()`. Each token is split on `|`, setting `DimSpec.mesh_axis` and `DimSpec.sharding_annotated`. The sentinel value `"None"` maps to `mesh_axis=None` (explicit replication).

### Mesh Resolution

`mesh_resolver.py` performs AST-based inference to extract mesh configuration from source code without importing it:

- `jax.make_mesh(shape, axis_names)` calls
- `AbstractMesh(sizes, names)` constructors
- `nnx.logical_axis_rules(...)` calls

The resolver returns a `MeshInfo` dataclass with `mesh` (axis -> size), `axis_rules` (logical -> physical), and `source_line`.

### Sharding Diagnostic Rules

| Rule | Check Function | Description |
|------|---------------|-------------|
| `sharding-rank-mismatch` | `check_sharding` | PartitionSpec length differs from array rank |
| `sharding-axis-unknown` | `check_sharding` | PartitionSpec references non-existent mesh axis |
| `sharding-conflict` | `check_sharding` | Conflicting PartitionSpecs on same shape at same line |
| `sharding-io-mismatch` | `check_sharding` | jit out_shardings contradict inner sharding constraint |
| `sharding-propagation-mismatch` | `check_sharding_propagation` | JAX-propagated output sharding differs from return annotation |
| `sharding-annotation-incomplete` | `check_annotation_sharding` | Piped shape with bare (unsharded) dims in strict mode |
| `sharding-dim-conflict` | `check_annotation_sharding` | Same dim name sharded on different axes across params |
| `sharding-mesh-undefined` | `check_mesh_axes` | mesh_axis references axis not in mesh or axis_rules |

### Graceful Fallback

When sharded `eval_shape` raises (operations without explicit sharding rules), the tracer catches the exception and retries via `_trace_fallback_unsharded()`:

1. Builds plain abstract inputs (no `NamedSharding`)
2. Runs `eval_shape` without mesh context
3. Returns a successful `TraceResult` with `sharding_fallback_reason` set

The pipeline emits a `trace-error` warning when fallback is used, so users know that sharding propagation was not verified for that function.

### Sharding in LSP

- **Inlay hints**: Display `dim|axis` per dimension when sharding is active (controlled by `sharding.display`)
- **Error hints**: All 8 sharding diagnostic rules are converted to `ErrorHintInfo` objects and displayed as error hints at the divergence line, not just as squiggly diagnostics
- **Completion**: Mesh axis names are suggested when typing inside pipe-syntax annotations
- **Hover**: Intermediate shapes show `P(...)` and mesh axis info inline

---

## LSP Multiplexer

### Diagnostic Filtering

The `--solo` flag (or `JAXTYC_MUX_SOLO` env var) filters diagnostics by server source:

- `jaxtyc`: Only jaxtyc shape-checking diagnostics
- `ty` / `primary` / `pyright`: Only type-checking diagnostics from the primary server

Filtering happens at the `publish_merged_diagnostics` level via `_filter_diag_sources()`, which filters the `diag_cache` dict by server key. The cache already separates diagnostics by source (`primary_name` vs `"jaxtyc"`).

All other LSP features (hover, completion, code actions, etc.) continue to merge from both servers regardless of the solo setting.
