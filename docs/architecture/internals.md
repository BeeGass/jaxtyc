# Internals

## Prime Sieve Dimensions

### The Problem

If two named dimensions happen to share the same concrete size, shape bugs become invisible. Consider multi-head attention where `batch=4` and `head_dim=4`. A transpose that swaps `batch` and `head_dim` produces the same shape tuple `(4, ...)` -- the checker cannot distinguish the two dimensions and reports no error.

### The Solution

Each named dimension is assigned a unique prime number starting at 101:

| Dimension | Prime |
|-----------|-------|
| batch | 101 |
| heads | 103 |
| seq | 107 |
| head_dim | 109 |

Primes are coprime by definition -- no product of primes equals another prime, and no permutation of primes produces the same tuple (unless the permutation is the identity). A transposed shape `(101, 103, 109, 107)` is unambiguously different from `(101, 103, 107, 109)`.

Starting at 101 (rather than 2) avoids collisions with small fixed dimension sizes commonly found in annotations (e.g., `Float[Array, "batch 4 d_model"]`).

### Implementation

`DimEnv` maintains a bidirectional mapping between dimension names and primes. Primes are generated via the Sieve of Eratosthenes with an initial limit of 1000. If all pre-computed primes are exhausted (unlikely for typical ML models), the sieve limit doubles and the sieve re-runs. Fixed literal sizes in annotations are passed as `reserved` to the constructor so primes never collide with them.

```
DimEnv._name_to_size: {"batch": 101, "heads": 103, "seq": 107, "head_dim": 109}
DimEnv._size_to_name: {101: "batch", 103: "heads", 107: "seq", 109: "head_dim"}
```

**Special dimension kinds:**

- **Fixed dims** (e.g., `Float[Array, "batch 4 d_model"]`): the literal `4` is used directly, not a prime. This is intentional -- fixed dims represent known constants.
- **Variadic dims** (`*batch`): expanded to 2 primes under internal names `_var_batch_0` and `_var_batch_1`. Two primes are needed because variadic dims represent an unknown number of leading dimensions (minimum 2 for shape distinguishability).
- **Ellipsis dims** (`...`): expanded to 2 primes under `_ellipsis_0` and `_ellipsis_1`, same reasoning as variadic.
- **Anonymous dims** (`_`): each gets a unique prime under `_anon_{counter}`.

**Reverse mapping**: `shape_to_names(shape)` maps a shape tuple back to dimension names. This powers human-readable diagnostic messages (showing `(batch, heads, head_dim, seq)` instead of `(101, 103, 109, 107)`) and LSP hover/CodeLens.

A shared `DimEnv` is created per file, so the same dimension name always maps to the same prime across all functions in the file. This enables cross-function consistency checking -- if `encode` and `decode` both use a dimension named `hidden`, it maps to the same prime in both, so the checker can verify that outputs match across call boundaries.

---

## jax.eval_shape Tracing

### How It Works

`jax.eval_shape(fn, *args)` runs a function with abstract inputs (`ShapeDtypeStruct`) that carry only shape and dtype metadata -- no actual array data is allocated and no computation is performed. JAX propagates shapes through every operation in the function body and returns the output's shape/dtype.

### Input Construction

For each annotated parameter, a `ShapeDtypeStruct` is built from the `ShapeSpec`:

```python
# Annotation: q: Float[Array, "batch heads seq head_dim"]
# DimEnv assigns: batch=101, heads=103, seq=107, head_dim=109
ShapeDtypeStruct(shape=(101, 103, 107, 109), dtype=float32)
```

The dtype is resolved from the jaxtyping class name (`Float` -> `float32`, `BFloat16` -> `bfloat16`, etc.) via `_DTYPE_MAP`.

Parameters annotated with `Float[Array, ...]` (ellipsis literal, meaning "any shape") are skipped -- there is no shape to construct.

### Output Extraction

The return value of `eval_shape` may be:

- A single `ShapeDtypeStruct` -- shape and dtype are read directly.
- A PyTree of structs (tuple, dict, nested dataclass) -- `jax.tree.leaves()` extracts all leaves. For tuple return annotations (`tuple[Float[...], Float[...]]`), each leaf is checked element-by-element against its corresponding annotation. If the element count differs, a `return-count-mismatch` diagnostic is emitted.

### Error Handling

If `eval_shape` raises (e.g., incompatible matmul dimensions, unsupported Python control flow), the exception message is captured and emitted as a `trace-error` diagnostic. The function is not re-traced.

### Flax NNX and Equinox Support

For methods on Flax NNX modules, `nnx.eval_shape` is used instead of `jax.eval_shape`. The pipeline constructs an abstract module instance (trying the constructor with a dummy RNG key), then traces the bound method.

For Equinox modules, `jax.eval_shape` traces bound methods on abstract module instances similarly.

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

- **Hover**: when the cursor is on a line, all `IntermediateShape` objects with a matching `source_line` are collected. The hover popup shows each operation's name, output shape (with named dimensions), and dtype. Hover also works on dimension names (showing symbolic prime, all usages) and function names (showing full shape signature).
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

Cross-function checking relies on the shared-per-file `DimEnv`. Because all functions in a file share the same prime assignments, dimension name `hidden` in function `encode` maps to the same prime as `hidden` in function `decode`. This makes shape comparisons across function boundaries meaningful.

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
