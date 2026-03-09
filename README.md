# jaxtyc

Static array shape checking for JAX powered by `jax.eval_shape`.

Reads [jaxtyping](https://docs.kidger.site/jaxtyping/) annotations and verifies shapes at analysis time — no runtime cost, no FLOPs.

## Installation

```bash
uv add jaxtyc
```

## Quick Start

```bash
jaxtyc check my_model.py
```
