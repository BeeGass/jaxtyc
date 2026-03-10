# Diagnostic Rules

jaxtyc emits diagnostics with structured rule codes. Each diagnostic includes a file location, severity, human-readable message, and the rule code that triggered it.

## Rule Catalog

| Rule | Severity | Fires When | Fix |
|------|----------|-----------|-----|
| `shape-mismatch` | error | Dimension values differ between expected and actual | Check that operations preserve the annotated dimensions |
| `rank-mismatch` | error | Number of dimensions differs | Check for missing/extra dims from ops like `sum`, `expand_dims` |
| `trace-error` | error | `jax.eval_shape` raised an exception | Fix the function so it runs correctly with abstract inputs |
| `param-inconsistency` | error | Parameter annotation conflicts with resolved input shape | Ensure parameter annotations match the function signature |
| `cross-function-mismatch` | error | Callee output shape contradicts its annotation at a call site | Fix the callee's return annotation or implementation |
| `return-count-mismatch` | error | Tuple return element count differs from annotation | Match the number of elements in the return tuple annotation |
| `file-not-found` | info | File path does not exist | Check the path argument |
| `read-error` | info | File could not be read | Check file permissions |
| `import-error` | info | Module could not be imported | Ensure all dependencies are installed |
| `resolve-error` | info | Function object could not be resolved from module | Check function name and class membership |

---

## Error Rules (with examples)

### `shape-mismatch`

Fires when the traced output shape has the correct rank but one or more dimension values differ from the annotation. Because jaxtyc assigns unique primes to each named dimension, a swapped axis produces a different prime in that position.

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def attention(
    q: Float[Array, "batch heads seq head_dim"],
    k: Float[Array, "batch heads seq head_dim"],
) -> Float[Array, "batch heads seq seq"]:
    # Bug: transposes q instead of k.
    # matmul produces (batch, heads, head_dim, seq)
    # but annotation expects (batch, heads, seq, seq).
    return jnp.matmul(jnp.swapaxes(q, -1, -2), k)
```

```
$ jaxtyc check wrong_transpose.py
wrong_transpose.py:8:0: error[shape-mismatch]
  Shape mismatch in return of `attention`
    Expected: (batch, heads, seq, seq)
    Got:      (batch, heads, head_dim, seq)
```

---

### `rank-mismatch`

Fires when the traced output has a different number of dimensions than annotated. Common causes: reducing ops without `keepdims=True`, missing `expand_dims`, or reshape errors.

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def project(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    # Bug: sum over last axis collapses it.
    # Output is (batch, seq), not (batch, seq, d_model).
    return jnp.sum(x, axis=-1)
```

```
$ jaxtyc check wrong_rank.py
wrong_rank.py:8:0: error[rank-mismatch]
  Rank mismatch in return of `project`
    Expected: (batch, seq, d_model) (rank 3)
    Got:      (batch, seq) (rank 2)
```

---

### `trace-error`

Fires when `jax.eval_shape` itself raises an exception during tracing. This means the function body cannot be symbolically evaluated with the given input shapes -- typically due to incompatible operands, unsupported Python constructs inside JIT, or missing imports at module scope.

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def broken_matmul(
    a: Float[Array, "batch m k"],
    b: Float[Array, "batch n k"],  # inner dims don't align
) -> Float[Array, "batch m n"]:
    return jnp.matmul(a, b)  # k != n, JAX raises
```

```
$ jaxtyc check broken_matmul.py
broken_matmul.py:5:0: error[trace-error]
  Trace error in `broken_matmul`: ...matmul requires compatible inner dimensions...
```

!!! tip
    `trace-error` often surfaces real bugs that would also fail at runtime. Fix the function body first, then re-run jaxtyc.

---

### `param-inconsistency`

Fires when a parameter's annotated shape does not match the shape that JAX resolved during tracing. This can happen when the annotation and actual function signature disagree about parameter order or shape.

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def linear(
    x: Float[Array, "batch seq d_in"],
    w: Float[Array, "d_in d_out"],
) -> Float[Array, "batch seq d_out"]:
    # If w is actually used as (d_out, d_in) internally
    return jnp.matmul(x, w)
```

```
$ jaxtyc check param_bug.py
param_bug.py:5:0: error[param-inconsistency]
  Parameter `w` shape inconsistency in `linear`
    Annotated: (d_in, d_out)
    Resolved:  (d_out, d_in)
```

---

### `cross-function-mismatch`

Fires when a function calls another annotated function and the callee's traced output shape does not match its return annotation. This catches bugs where the annotation is wrong but the callee is used downstream.

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def encode(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq hidden"]:
    # Bug: returns (batch, seq, d_model) not (batch, seq, hidden)
    return x

def pipeline(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq hidden"]:
    return encode(x)
```

```
$ jaxtyc check cross_bug.py
cross_bug.py:5:0: error[shape-mismatch]
  Shape mismatch in return of `encode`
    Expected: (batch, seq, hidden)
    Got:      (batch, seq, d_model)
cross_bug.py:12:11: error[cross-function-mismatch]
  Cross-function shape mismatch: `encode` called from `pipeline`
    Annotated return: (batch, seq, hidden)
    Actual return:    (batch, seq, d_model)
```

---

### `return-count-mismatch`

Fires when a function annotated with a tuple return type produces a different number of output elements than expected.

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def split_heads(
    x: Float[Array, "batch seq d_model"],
) -> tuple[Float[Array, "batch seq half"], Float[Array, "batch seq half"]]:
    # Bug: returns 3 elements instead of 2
    third = x.shape[-1] // 3
    return x[..., :third], x[..., third:2*third], x[..., 2*third:]
```

```
$ jaxtyc check tuple_bug.py
tuple_bug.py:5:0: error[return-count-mismatch]
  Return count mismatch in `split_heads`
    Expected: 2 elements
    Got:      3 elements
```

---

## Info Rules

The four info-severity rules (`file-not-found`, `read-error`, `import-error`, `resolve-error`) are non-fatal. They indicate that jaxtyc could not analyze a file or function but do not represent shape bugs. By default, info diagnostics are shown; set `severity = "error"` in `[tool.jaxtyc]` to suppress them.

---

## Inline Suppressions

Add a comment to suppress diagnostics on a specific line:

```python
# Suppress all rules on this line
result = buggy_function(x)  # jaxtyc: ignore

# Suppress a specific rule
result = buggy_function(x)  # jaxtyc: ignore[shape-mismatch]

# Suppress multiple rules
result = buggy_function(x)  # jaxtyc: ignore[shape-mismatch, rank-mismatch]
```

**Placement rules:**

- The comment can be on the same line as the diagnostic, or on the line immediately before it.
- An empty rule list (`# jaxtyc: ignore`) suppresses all rules on that line.
- Rule names must match exactly (e.g., `shape-mismatch`, not `shape_mismatch`).
