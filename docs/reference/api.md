# Python API

## Public API

::: jaxtyc.analyze_file

::: jaxtyc.Diagnostic

::: jaxtyc.FileResult

::: jaxtyc.TraceResult

## Types

::: jaxtyc.types.DimSpec

::: jaxtyc.types.ShapeSpec

::: jaxtyc.types.FunctionShapeSpec

::: jaxtyc.types.IntermediateShape

::: jaxtyc.types.ShardingInfo

::: jaxtyc.types.ErrorHintInfo

::: jaxtyc.types.DimLocation

::: jaxtyc.types.CallSite

::: jaxtyc.types.DiagnosticData

::: jaxtyc.types.SuppressionComment

### Type Aliases

- **`DimSize`**: `TypeAlias = Any` -- represents either a plain `int` or a symbolic `jax.export._DimExpr` object. Used in shape fields of `DiagnosticData`, `IntermediateShape`, and `TraceResult`.
- **`NamedShape`**: `TypeAlias = tuple[str | None, ...]` -- dimension names resolved from tracing, with `None` for unrecognised sizes.
- **`Severity`**: `TypeAlias = Literal["error", "warning", "info"]`

## Configuration

::: jaxtyc.config.JaxtycConfig

::: jaxtyc.config.HintsConfig

::: jaxtyc.config.ShardingConfig

::: jaxtyc.config.NavigationConfig

::: jaxtyc.config.load_config

::: jaxtyc.config.filter_diagnostics
