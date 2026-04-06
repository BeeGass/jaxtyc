# Your First Shape Check

This walkthrough takes you from a correct function through introducing a bug to tracing intermediate shapes. Assumes jaxtyc is installed and `jaxtyc version` works.

---

## Step 1: Write an Annotated Function

Create `attention.py` with standard multi-head attention:

```python
# attention.py
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

def attention(
    q: Float[Array, "batch heads seq head_dim"],
    k: Float[Array, "batch heads seq head_dim"],
    v: Float[Array, "batch heads seq head_dim"],
) -> Float[Array, "batch heads seq head_dim"]:
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2))  # (batch, heads, seq, seq)
    weights = jax.nn.softmax(scores, axis=-1)           # (batch, heads, seq, seq)
    return jnp.matmul(weights, v)                       # (batch, heads, seq, head_dim)
```

Every parameter and the return type carry jaxtyping shape annotations. The four named dimensions -- `batch`, `heads`, `seq`, `head_dim` -- are all jaxtyc needs.

---

## Step 2: Check It

```bash
$ jaxtyc check attention.py
All checks passed: 1 function(s) checked (0.04s)
```

No errors. The output shape `(batch, heads, seq, head_dim)` matches the return annotation.

---

## Step 3: Introduce a Bug

Replace the function with a buggy version that transposes `q` instead of `k`:

```python
def attention(
    q: Float[Array, "batch heads seq head_dim"],
    k: Float[Array, "batch heads seq head_dim"],
) -> Float[Array, "batch heads seq seq"]:
    return jnp.matmul(jnp.swapaxes(q, -1, -2), k)  # Bug!
```

The intent is `(batch, heads, seq, seq)`, but `swapaxes(q, -1, -2)` produces `(batch, heads, head_dim, seq)` and `matmul(..., k)` contracts over `seq`, yielding `(batch, heads, head_dim, seq)` -- not `(batch, heads, seq, seq)`.

```bash
$ jaxtyc check attention.py
attention.py:6:0: error[shape-mismatch]
  Shape mismatch in return of `attention`
    Expected: (batch, heads, seq, seq)
    Got:      (batch, heads, head_dim, seq)

Found 1 error(s) in 1 function(s) checked (0.03s)
```

!!! warning "Why this is subtle"
    If `seq` and `head_dim` happen to be equal at runtime (e.g., both 64), a unit test with concrete values would pass. jaxtyc uses distinct symbolic dimensions for each name, so `seq` and `head_dim` can never collide regardless of their concrete sizes.

---

## Step 4: Trace Intermediate Shapes

Go back to the correct version from Step 1 and run `jaxtyc trace`:

```bash
$ jaxtyc trace attention.py::attention
attention(q: float32[batch, heads, seq, head_dim], k: float32[batch, heads, seq, head_dim], v: float32[batch, heads, seq, head_dim]) -> float32[batch, heads, seq, head_dim]

  Line 11: transpose -> (batch, heads, head_dim, seq)  [float32]
  Line 11: dot_general -> (batch, heads, seq, seq)  [float32]
  Line 12: sub -> (batch, heads, seq, seq)  [float32]
  Line 12: exp -> (batch, heads, seq, seq)  [float32]
  Line 12: reduce_sum -> (batch, heads, seq, 1)  [float32]
  Line 12: div -> (batch, heads, seq, seq)  [float32]
  Line 13: dot_general -> (batch, heads, seq, head_dim)  [float32]

  Output: (batch, heads, seq, head_dim) [matches]
```

!!! info "Reading the trace"
    Each line shows a JAX primitive (`transpose`, `dot_general`, `exp`, etc.), its output shape with dimension names resolved, and the source line it maps to. The final `[matches]` confirms the traced output equals the annotated return shape.

### How symbolic dimension tracing works

When jaxtyc builds abstract inputs, it assigns each unique dimension name a distinct symbolic dimension via `jax.export.symbolic_shape`. For example, the four dimensions in the attention function become four symbolic values:

- `batch` -> a unique symbolic `_DimExpr` for "batch"
- `heads` -> a unique symbolic `_DimExpr` for "heads"
- `seq` -> a unique symbolic `_DimExpr` for "seq"
- `head_dim` -> a unique symbolic `_DimExpr` for "head_dim"

These symbolic dimensions are abstract values that JAX traces through computations without allocating real arrays or executing any FLOPs. At the output, jaxtyc compares the symbolic values against the return annotation. If the output contains the symbolic value for `head_dim` where the annotation says `seq`, jaxtyc detects the mismatch because these are distinct symbolic objects -- reported with exact dimension names rather than opaque integers.

No arrays are allocated and no computation is executed. Shapes are checked purely symbolically.

!!! tip "When to use trace"
    `jaxtyc trace` is most useful for debugging *why* a shape mismatch occurs. The intermediate shapes show exactly which operation introduced the wrong dimension.

---

## Next Steps

- [CLI reference](guide/cli.md) -- all commands, flags, and output formats
- [Annotation guide](guide/annotations.md) -- supported jaxtyping patterns (variadics, ellipsis, fixed dims)
- [CI integration](guide/ci.md) -- `jaxtyc check --format github` in GitHub Actions
