# Annotations

jaxtyc reads standard [jaxtyping](https://github.com/patrick-kidger/jaxtyping) annotations directly from your source AST. No decorators, no runtime imports, no configuration needed -- if your function has `Float[Array, "batch seq d_model"]`, jaxtyc will find it.

---

## Supported dtype classes

jaxtyc recognizes every jaxtyping dtype class and maps it to a concrete JAX dtype for `eval_shape` tracing:

| jaxtyping class | jaxtyc dtype | JAX dtype used for tracing |
|-----------------|-------------|---------------------------|
| `Float` | `float32` | `jnp.float32` |
| `Float16` | `float16` | `jnp.float16` |
| `Float32` | `float32` | `jnp.float32` |
| `Float64` | `float64` | `jnp.float64` |
| `BFloat16` | `bfloat16` | `jnp.bfloat16` |
| `Int` | `int` | `jnp.int32` |
| `Int8` | `int8` | `jnp.int8` |
| `Int16` | `int16` | `jnp.int16` |
| `Int32` | `int32` | `jnp.int32` |
| `Int64` | `int64` | `jnp.int64` |
| `UInt` | `uint` | `jnp.uint32` |
| `UInt8` | `uint8` | `jnp.uint8` |
| `UInt16` | `uint16` | `jnp.uint16` |
| `UInt32` | `uint32` | `jnp.uint32` |
| `UInt64` | `uint64` | `jnp.uint64` |
| `Bool` | `bool` | `jnp.bool_` |
| `Complex` | `complex` | `jnp.complex64` |
| `Complex64` | `complex64` | `jnp.complex64` |
| `Complex128` | `complex128` | `jnp.complex128` |
| `Num` | `numeric` | `jnp.float32` |
| `Shaped` | `shaped` | `jnp.float32` |
| `Key` | `key` | `jnp.uint32` |
| `Scalar` | `scalar` | `jnp.float32` |

!!! note "Abstract dtype classes"
    `Float`, `Int`, `UInt`, `Num`, `Shaped`, `Key`, and `Scalar` are abstract -- they accept multiple concrete dtypes at runtime. For tracing purposes, jaxtyc picks a single representative dtype (shown in the third column). This does not affect shape checking.

---

## Shape string syntax

The second argument to a jaxtyping annotation is a shape string. jaxtyc parses each token and assigns it a symbolic size for tracing.

### Named dimensions

Each unique name gets a distinct prime number as its size. Two parameters sharing a dimension name share the same prime, so shape mismatches propagate correctly through arithmetic.

```python
def attention(
    q: Float[Array, "batch seq d_model"],
    k: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    ...
```

`batch`, `seq`, and `d_model` each get one prime (e.g., 2, 3, 5). Both `q` and `k` get shape `(2, 3, 5)`.

### Fixed dimensions

Literal integers are used as-is. Useful for known constants like attention head count or patch size.

```python
def to_heads(
    x: Float[Array, "batch seq 512"],
) -> Float[Array, "batch seq 8 64"]:
    ...
```

`512`, `8`, and `64` appear verbatim in the traced shape.

### Variadic dimensions

A `*` prefix expands to 2 placeholder dimensions. This models batch dimensions of unknown rank.

```python
def layer_norm(
    x: Float[Array, "*batch d_model"],
) -> Float[Array, "*batch d_model"]:
    ...
```

`*batch` expands to two internal dims (`_var_batch_0`, `_var_batch_1`), giving a rank-3 shape for tracing.

### Anonymous dimensions

`_` is an unnamed placeholder. Each `_` gets its own unique prime, so two `_` tokens do not constrain each other.

```python
def pool(
    x: Float[Array, "batch _ d_model"],
) -> Float[Array, "batch _ d_model"]:
    ...
```

### Ellipsis (any shape)

`"..."` as the entire shape string skips shape checking entirely. The function is still traced for other annotations.

```python
def identity(
    x: Float[Array, "..."],
) -> Float[Array, "..."]:
    return x
```

### Ellipsis prefix

`"..."` as a leading token matches any number of leading dimensions. Only the suffix dimensions are checked.

```python
def final_linear(
    x: Float[Array, "... d_model"],
    w: Float[Array, "d_model d_out"],
) -> Float[Array, "... d_out"]:
    ...
```

The `...` prefix expands to 2 internal dims (`_ellipsis_0`, `_ellipsis_1`), making the traced input rank 3.

### Scalar

An empty string `""` means a zero-rank (scalar) array.

```python
def mse_loss(
    pred: Float[Array, "batch d"],
    target: Float[Array, "batch d"],
) -> Float[Array, ""]:
    return jnp.mean((pred - target) ** 2)
```

---

## Common ML patterns

### Attention (q/k/v with matmul and transpose)

```python
def scaled_dot_product_attention(
    q: Float[Array, "batch heads seq d_head"],
    k: Float[Array, "batch heads seq d_head"],
    v: Float[Array, "batch heads seq d_head"],
) -> Float[Array, "batch heads seq d_head"]:
    scale = jnp.sqrt(q.shape[-1]).astype(q.dtype)
    # (batch, heads, seq, d_head) @ (batch, heads, d_head, seq) -> (batch, heads, seq, seq)
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale
    weights = jax.nn.softmax(attn, axis=-1)
    # (batch, heads, seq, seq) @ (batch, heads, seq, d_head) -> (batch, heads, seq, d_head)
    return jnp.matmul(weights, v)
```

jaxtyc traces the matmul chain and verifies the output shape `(batch, heads, seq, d_head)` matches the return annotation.

### Linear projection (matmul changes inner dim)

```python
def linear(
    x: Float[Array, "batch seq d_in"],
    w: Float[Array, "d_in d_out"],
    b: Float[Array, "d_out"],
) -> Float[Array, "batch seq d_out"]:
    return x @ w + b
```

The checker confirms that `d_in` contracts and `d_out` appears in the output.

### Residual connections (add requires matching shapes)

```python
def residual_block(
    x: Float[Array, "batch seq d_model"],
    sublayer_out: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    return x + sublayer_out
```

Both operands share all three dimension names, so the add is shape-compatible.

### Flax NNX methods

jaxtyc automatically skips `self` and `cls` parameters when parsing class methods. Annotate the remaining parameters and return type as usual.

```python
class MLP(nnx.Module):
    def __call__(
        self,
        x: Float[Array, "batch seq d_model"],
    ) -> Float[Array, "batch seq d_model"]:
        ...
```

### Scalar returns

Loss functions returning a scalar should annotate with an empty shape string:

```python
def cross_entropy(
    logits: Float[Array, "batch vocab"],
    labels: Int[Array, "batch"],
) -> Float[Array, ""]:
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(labels, logits.shape[-1]), axis=-1))
```

---

## Limitations

- **No tuple/pytree returns.** If a function returns a tuple or nested pytree of arrays, jaxtyc only checks the first leaf. Multi-output annotations are not yet supported.
- **No runtime-dependent shapes.** Shapes that depend on array *values* (e.g., `jnp.where` producing variable-length output) cannot be statically determined. jaxtyc reports the shape that `jax.eval_shape` computes, which may differ from the actual runtime shape in pathological cases.
- **No cross-function inference.** Each function is traced independently. If function `A` calls function `B`, the shapes inside `B` are not checked when analyzing `A` -- you need annotations on `B` as well.
- **Variadic and ellipsis expand to fixed rank.** `*batch` always expands to exactly 2 dimensions and `...` prefix also expands to 2. If your actual batch rank differs, the rank check may produce a false positive. This is a deliberate simplification for static analysis.
- **Import side effects.** jaxtyc imports your module to get live function objects for `jax.eval_shape`. If your module has import-time side effects (GPU initialization, file I/O), those will execute during analysis.
- **Abstract dtype resolution.** `Num`, `Shaped`, `Key`, and `Scalar` are mapped to a single representative dtype. If your function's behavior depends on the specific dtype within an abstract class, jaxtyc may not catch dtype-related shape differences.
