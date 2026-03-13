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
| `sharding-rank-mismatch` | error | PartitionSpec length differs from array rank | Match PartitionSpec entries to array dimensions |
| `sharding-axis-unknown` | error | PartitionSpec references a non-existent mesh axis | Use axis names defined in the Mesh |
| `sharding-conflict` | error | Conflicting PartitionSpecs on same shape at same line | Use a single consistent PartitionSpec |
| `sharding-io-mismatch` | warning | jit out_shardings contradict an inner sharding_constraint | Align jit output sharding with inner constraints |
| `sharding-propagation-mismatch` | error | JAX-propagated output sharding differs from return annotation | Fix the return annotation sharding or the function body |
| `sharding-annotation-incomplete` | warning | Piped shape with bare (unsharded) dims in strict mode | Add `\|axis` or `\|None` to all dims in piped annotations |
| `sharding-dim-conflict` | error | Same dim name sharded on different axes across params | Use consistent sharding for the same dimension name |
| `sharding-mesh-undefined` | error | mesh_axis references axis not in mesh config or axis_rules | Use an axis name defined in the mesh or axis_rules |
| `file-not-found` | info | File path does not exist | Check the path argument |
| `read-error` | info | File could not be read | Check file permissions |
| `import-error` | info | Module could not be imported | Ensure all dependencies are installed |
| `resolve-error` | info | Function object could not be resolved from module | Check function name and class membership |

---

## Error Rules (with examples)

### `shape-mismatch`

Fires when the traced output shape has the correct rank but one or more dimension values differ from the annotation. Because jaxtyc assigns unique symbolic dimensions to each named dimension, a swapped axis produces a different symbolic value in that position.

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

## Sharding Rules (with examples)

Sharding rules validate JAX sharding constraints, mesh axes, and pipe-syntax annotations. They fire from both trace-level checks (sharding metadata on intermediates from `jax.lax.with_sharding_constraint`, `shard_map`, `jit` out_shardings) and annotation-level checks (pipe syntax `dim|axis` in jaxtyping annotations).

### `sharding-rank-mismatch`

Fires when a `PartitionSpec` has a different number of entries than the array's rank. Each entry in `PartitionSpec` corresponds to one array dimension.

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float

devices = np.array(jax.devices()).reshape(1, 1)
mesh = Mesh(devices, ("data", "model"))

def sharded_fn(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    # Bug: P("data", None) has 2 entries but array has rank 3
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None)))
```

```
$ jaxtyc check sharded.py
sharded.py:14:0: error[sharding-rank-mismatch]
  Sharding rank mismatch in `sharded_fn`:
    PartitionSpec has 2 entries but array has rank 3
```

---

### `sharding-axis-unknown`

Fires when a `PartitionSpec` references an axis name that does not exist in the mesh.

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float

devices = np.array(jax.devices()).reshape(1)
mesh = Mesh(devices, ("data",))

def sharded_fn(
    x: Float[Array, "batch seq"],
) -> Float[Array, "batch seq"]:
    # Bug: "model" is not an axis name in this mesh
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("model", None)))
```

```
$ jaxtyc check sharded.py
sharded.py:14:0: error[sharding-axis-unknown]
  Unknown mesh axis `model` in `sharded_fn`:
    mesh has axes ('data',)
```

---

### `sharding-conflict`

Fires when multiple intermediates at the same source line and with the same shape have different `PartitionSpec` values. This indicates contradictory sharding intent.

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float

devices = np.array(jax.devices()).reshape(1, 1)
mesh = Mesh(devices, ("data", "model"))

def conflicting(
    x: Float[Array, "batch seq"],
) -> Float[Array, "batch seq"]:
    # Bug: two constraints on the same shape at the same line with different specs
    a = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None)))
    return jax.lax.with_sharding_constraint(a, NamedSharding(mesh, P(None, "model")))
```

```
$ jaxtyc check sharded.py
sharded.py:14:0: error[sharding-conflict]
  Conflicting sharding specs in `conflicting` at line 14:
    P('data', None), P(None, 'model')
```

---

### `sharding-io-mismatch`

Fires (as a warning) when a `jit` function's `out_shardings` contradict an inner `sharding_constraint` applied to the same shape. This can cause silent resharding at the jit boundary.

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float

devices = np.array(jax.devices()).reshape(1, 1)
mesh = Mesh(devices, ("data", "model"))

@jax.jit
def mismatch(
    x: Float[Array, "batch seq"],
) -> Float[Array, "batch seq"]:
    # Inner constraint says P("data", None)
    y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None)))
    return y
# But jit out_shardings says P(None, "model")
```

```
$ jaxtyc check sharded.py
sharded.py:15:0: warning[sharding-io-mismatch]
  Sharding I/O mismatch in `mismatch`:
    jit specifies P(None, 'model') but inner constraint specifies P('data', None)
```

---

### `sharding-propagation-mismatch`

Fires when the sharding that JAX propagates through operations differs from the sharding declared in the return annotation.

```python
from jaxtyping import Array, Float

def forward(
    x: Float[Array, "batch|dp seq|None d_model|mp"],
) -> Float[Array, "batch|dp seq|None d_model|None"]:
    # Bug: d_model keeps its |mp sharding through ops,
    # but return annotation claims |None
    return x * 2.0
```

```
$ jaxtyc check forward.py
forward.py:5:0: error[sharding-propagation-mismatch]
  Sharding propagation mismatch in `forward`:
    Propagated: P('dp', None, 'mp')
    Annotated:  P('dp', None, None)
```

---

### `sharding-annotation-incomplete`

Fires (as a warning) when a shape annotation uses pipe syntax but some dimensions lack sharding specification. Only active when `strict_annotation = true`.

```python
from jaxtyping import Array, Float

def forward(
    x: Float[Array, "batch|dp seq d_model|mp"],
    #                         ^^^ missing |axis
) -> Float[Array, "batch|dp seq|None d_model|mp"]:
    return x
```

```
$ jaxtyc check forward.py
forward.py:4:0: warning[sharding-annotation-incomplete]
  Incomplete sharding annotation in `forward`:
    dim `seq` has no |axis in a piped shape
```

---

### `sharding-dim-conflict`

Fires when the same dimension name is sharded on different mesh axes across parameters.

```python
from jaxtyping import Array, Float

def matmul(
    x: Float[Array, "batch|dp d_model|mp"],
    w: Float[Array, "d_model|dp d_out|None"],
    #                ^^^^^^^^^ d_model is |mp in x but |dp in w
) -> Float[Array, "batch|dp d_out|None"]:
    return x @ w
```

```
$ jaxtyc check matmul.py
matmul.py:4:0: error[sharding-dim-conflict]
  Sharding conflict for dim `d_model` in `matmul`:
    parameter `x` shards on axis `mp`
    parameter `w` shards on axis `dp`
```

---

### `sharding-mesh-undefined`

Fires when a `|axis` annotation references an axis name not defined in the mesh config or axis_rules.

```python
from jaxtyping import Array, Float

# With mesh = { data = 4 }, no axis_rules
def forward(
    x: Float[Array, "batch|dp seq|None"],
    #                      ^^^ dp is not in mesh or axis_rules
) -> Float[Array, "batch|dp seq|None"]:
    return x
```

```
$ jaxtyc check forward.py
forward.py:5:0: error[sharding-mesh-undefined]
  Undefined mesh axis `dp` on dim `batch` in `forward`:
    not found in mesh axes ['data'] or axis_rules []
```

---

## Info Rules

The four info-severity rules (`file-not-found`, `read-error`, `import-error`, `resolve-error`) are non-fatal. They indicate that jaxtyc could not analyze a file or function but do not represent shape bugs. By default (with `severity = "error"`), info diagnostics are suppressed; set `severity = "info"` in `[tool.jaxtyc]` to see them.

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
