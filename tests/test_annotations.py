"""Tests for jaxtyc.analyzer.annotations — AST-based jaxtyping annotation parser."""

from __future__ import annotations

import textwrap

from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.annotations import parse_shape_string
from jaxtyc.types import DimSpec

# ---------------------------------------------------------------------------
# parse_shape_string
# ---------------------------------------------------------------------------


class TestParseShapeString:
    def test_named_dims(self):
        spec = parse_shape_string("batch seq d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="named", name="batch"),
            DimSpec(kind="named", name="seq"),
            DimSpec(kind="named", name="d_model"),
        )
        assert spec.dtype == "float32"

    def test_fixed_dim(self):
        spec = parse_shape_string("batch 4 d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="named", name="batch"),
            DimSpec(kind="fixed", size=4),
            DimSpec(kind="named", name="d_model"),
        )

    def test_variadic_dim(self):
        spec = parse_shape_string("*batch seq", "float32")
        assert spec.dims == (
            DimSpec(kind="variadic", name="batch"),
            DimSpec(kind="named", name="seq"),
        )

    def test_anonymous_dim(self):
        spec = parse_shape_string("batch _ d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="named", name="batch"),
            DimSpec(kind="anonymous"),
            DimSpec(kind="named", name="d_model"),
        )

    def test_ellipsis_in_string(self):
        spec = parse_shape_string("... d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="ellipsis"),
            DimSpec(kind="named", name="d_model"),
        )

    def test_empty_string_is_scalar(self):
        spec = parse_shape_string("", "float32")
        assert spec.is_scalar
        assert spec.dims == ()

    def test_single_dim(self):
        spec = parse_shape_string("features", "float32")
        assert spec.dims == (DimSpec(kind="named", name="features"),)

    def test_mixed(self):
        spec = parse_shape_string("*batch 4 seq head_dim", "float32")
        assert len(spec.dims) == 4
        assert spec.dims[0] == DimSpec(kind="variadic", name="batch")
        assert spec.dims[1] == DimSpec(kind="fixed", size=4)
        assert spec.dims[2] == DimSpec(kind="named", name="seq")
        assert spec.dims[3] == DimSpec(kind="named", name="head_dim")


# ---------------------------------------------------------------------------
# extract_function_specs
# ---------------------------------------------------------------------------


class TestExtractFunctionSpecs:
    def test_simple_function(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def attention(
                q: Float[Array, "batch heads seq head_dim"],
                k: Float[Array, "batch heads seq head_dim"],
            ) -> Float[Array, "batch heads seq seq"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        func = specs[0]
        assert func.name == "attention"
        assert "q" in func.params
        assert "k" in func.params
        assert func.return_spec is not None
        assert len(func.return_spec.dims) == 4
        assert func.return_spec.dims[3] == DimSpec(kind="named", name="seq")

    def test_no_return_annotation(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def process(x: Float[Array, "batch dim"]):
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].return_spec is None
        assert "x" in specs[0].params

    def test_int_annotation(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Int

            def embed(ids: Int[Array, "batch seq"]) -> Int[Array, "batch seq dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].params["ids"].dtype == "int"
        assert specs[0].return_spec.dtype == "int"

    def test_bool_annotation(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Bool

            def mask(x: Bool[Array, "batch seq"]) -> Bool[Array, "batch seq"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].params["x"].dtype == "bool"

    def test_class_method(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            class MyModel:
                def __call__(self, x: Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
                    pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        func = specs[0]
        assert func.is_method
        assert func.class_name == "MyModel"
        assert "self" not in func.params
        assert "x" in func.params

    def test_no_jaxtyping_annotations_skipped(self):
        source = textwrap.dedent("""\
            def add(x: int, y: int) -> int:
                return x + y
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 0

    def test_mixed_annotations(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def forward(x: Float[Array, "batch dim"], training: bool) -> Float[Array, "batch dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert "x" in specs[0].params
        assert "training" not in specs[0].params

    def test_multiple_functions(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def encode(x: Float[Array, "batch dim"]) -> Float[Array, "batch hidden"]:
                pass

            def decode(x: Float[Array, "batch hidden"]) -> Float[Array, "batch dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 2
        assert specs[0].name == "encode"
        assert specs[1].name == "decode"

    def test_ellipsis_any_shape(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def flexible(x: Float[Array, "..."]) -> Float[Array, "batch dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].params["x"].is_any_shape

    def test_scalar_annotation(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def loss(x: Float[Array, "batch dim"]) -> Float[Array, ""]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].return_spec.is_scalar

    def test_lineno_tracking(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            # some comment

            def attention(
                q: Float[Array, "batch heads seq head_dim"],
            ) -> Float[Array, "batch heads seq seq"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].lineno == 5
        assert specs[0].file_path == "test.py"
