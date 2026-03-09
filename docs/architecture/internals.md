# Internals

## Prime Sieve Dimensions

### The Problem

If two named dimensions happen to share the same concrete size, shape bugs become invisible. Consider multi-head attention where `batch=4` and `head_dim=4`. A transpose that swaps `batch` and `head_dim` produces the same shape tuple `(4, ...)` -- the checker cannot distinguish the two dimensions and reports no error.

### The Solution

Each named dimension is assigned a unique prime number:

| Dimension | Prime |
|-----------|-------|
| batch | 2 |
| heads | 3 |
| seq | 5 |
| head_dim | 7 |

Primes are coprime by definition -- no product of primes equals another prime, and no permutation of primes produces the same tuple (unless the permutation is the identity). A transposed shape `(2, 3, 7, 5)` is unambiguously different from `(2, 3, 5, 7)`.

### Implementation

`DimEnv` maintains a bidirectional mapping between dimension names and primes. Primes are generated via the Sieve of Eratosthenes with an initial limit of 1000. If all pre-computed primes are exhausted (unlikely for typical ML models), the sieve limit doubles and the sieve re-runs.

```
DimEnv._name_to_size: {"batch": 2, "heads": 3, "seq": 5, "head_dim": 7}
DimEnv._size_to_name: {2: "batch", 3: "heads", 5: "seq", 7: "head_dim"}
```

**Special dimension kinds:**

- **Fixed dims** (e.g., `Float[Array, "batch 4 d_model"]`): the literal `4` is used directly, not a prime. This is intentional -- fixed dims represent known constants.
- **Variadic dims** (`*batch`): expanded to 2 primes under internal names `_var_batch_0` and `_var_batch_1`. Two primes are needed because variadic dims represent an unknown number of leading dimensions (minimum 2 for shape distinguishability).
- **Ellipsis dims** (`...`): expanded to 2 primes under `_ellipsis_0` and `_ellipsis_1`, same reasoning as variadic.
- **Anonymous dims** (`_`): each gets a unique prime under `_anon_{position}`.

**Reverse mapping**: `shape_to_names(shape)` maps a shape tuple back to dimension names. This powers human-readable diagnostic messages (showing `(batch, heads, head_dim, seq)` instead of `(2, 3, 7, 5)`) and LSP hover/CodeLens.

A fresh `DimEnv` is created per function to avoid cross-contamination between functions that reuse dimension names with different semantics.

---

## jax.eval_shape Tracing

### How It Works

`jax.eval_shape(fn, *args)` runs a function with abstract inputs (`ShapeDtypeStruct`) that carry only shape and dtype metadata -- no actual array data is allocated and no computation is performed. JAX propagates shapes through every operation in the function body and returns the output's shape/dtype.

### Input Construction

For each annotated parameter, a `ShapeDtypeStruct` is built from the `ShapeSpec`:

```python
# Annotation: q: Float[Array, "batch heads seq head_dim"]
# DimEnv assigns: batch=2, heads=3, seq=5, head_dim=7
ShapeDtypeStruct(shape=(2, 3, 5, 7), dtype=float32)
```

The dtype is resolved from the jaxtyping class name (`Float` -> `float32`, `BFloat16` -> `bfloat16`, etc.) via `_DTYPE_MAP`.

Parameters annotated with `Float[Array, ...]` (ellipsis literal, meaning "any shape") are skipped -- there is no shape to construct.

### Output Extraction

The return value of `eval_shape` may be:

- A single `ShapeDtypeStruct` -- shape and dtype are read directly.
- A PyTree of structs (tuple, dict, nested dataclass) -- `jax.tree.leaves()` extracts all leaves and the first leaf's shape is used for comparison against the return annotation.

### Error Handling

If `eval_shape` raises (e.g., incompatible matmul dimensions, unsupported Python control flow), the exception message is captured and emitted as a `trace-error` diagnostic. The function is not re-traced.

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

- **Hover**: when the cursor is on a line, all `IntermediateShape` objects with a matching `source_line` are collected. The hover popup shows each operation's name, output shape (with named dimensions), and dtype.
- **CodeLens**: for each function with a successful trace, a virtual annotation is rendered above the function definition showing the traced input parameter shapes and output shape.
- **trace command**: the CLI `jaxtyc trace file.py::function_name` prints all intermediates with their source lines, giving a step-by-step view of shape propagation through the function.

!!! note "Column limitation"
    JAX's traceback frames do not expose column information. The `source_col` is always 0. This means hover activates for the entire line, not a specific expression within the line.
