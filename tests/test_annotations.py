"""Tests for jaxtyc.analyzer.annotations — AST-based jaxtyping annotation parser."""

from __future__ import annotations

import textwrap

from jaxtyc.analyzer.annotations import extract_call_sites
from jaxtyc.analyzer.annotations import extract_dim_locations
from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.annotations import parse_shape_string
from jaxtyc.types import DimSpec

# ---------------------------------------------------------------------------
# parse_shape_string
# ---------------------------------------------------------------------------


class TestParseShapeString:
    def test_named_dims(self) -> None:
        spec = parse_shape_string("batch seq d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="named", name="batch"),
            DimSpec(kind="named", name="seq"),
            DimSpec(kind="named", name="d_model"),
        )
        assert spec.dtype == "float32"

    def test_fixed_dim(self) -> None:
        spec = parse_shape_string("batch 4 d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="named", name="batch"),
            DimSpec(kind="fixed", size=4),
            DimSpec(kind="named", name="d_model"),
        )

    def test_variadic_dim(self) -> None:
        spec = parse_shape_string("*batch seq", "float32")
        assert spec.dims == (
            DimSpec(kind="variadic", name="batch"),
            DimSpec(kind="named", name="seq"),
        )

    def test_anonymous_dim(self) -> None:
        spec = parse_shape_string("batch _ d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="named", name="batch"),
            DimSpec(kind="anonymous"),
            DimSpec(kind="named", name="d_model"),
        )

    def test_ellipsis_in_string(self) -> None:
        spec = parse_shape_string("... d_model", "float32")
        assert spec.dims == (
            DimSpec(kind="ellipsis"),
            DimSpec(kind="named", name="d_model"),
        )

    def test_empty_string_is_scalar(self) -> None:
        spec = parse_shape_string("", "float32")
        assert spec.is_scalar
        assert spec.dims == ()

    def test_single_dim(self) -> None:
        spec = parse_shape_string("features", "float32")
        assert spec.dims == (DimSpec(kind="named", name="features"),)

    def test_mixed(self) -> None:
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
    def test_simple_function(self) -> None:
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

    def test_no_return_annotation(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def process(x: Float[Array, "batch dim"]):
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].return_spec is None
        assert "x" in specs[0].params

    def test_int_annotation(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Int

            def embed(ids: Int[Array, "batch seq"]) -> Int[Array, "batch seq dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].params["ids"].dtype == "int"
        assert specs[0].return_spec.dtype == "int"

    def test_bool_annotation(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Bool

            def mask(x: Bool[Array, "batch seq"]) -> Bool[Array, "batch seq"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].params["x"].dtype == "bool"

    def test_class_method(self) -> None:
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

    def test_no_jaxtyping_annotations_skipped(self) -> None:
        source = textwrap.dedent("""\
            def add(x: int, y: int) -> int:
                return x + y
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 0

    def test_mixed_annotations(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def forward(x: Float[Array, "batch dim"], training: bool) -> Float[Array, "batch dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert "x" in specs[0].params
        assert "training" not in specs[0].params

    def test_multiple_functions(self) -> None:
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

    def test_ellipsis_any_shape(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def flexible(x: Float[Array, "..."]) -> Float[Array, "batch dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].params["x"].is_any_shape

    def test_scalar_annotation(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def loss(x: Float[Array, "batch dim"]) -> Float[Array, ""]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].return_spec.is_scalar

    def test_posonly_args(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def f(x: Float[Array, "a b"], /) -> Float[Array, "a b"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert "x" in specs[0].params

    def test_kwonly_args(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def f(*, x: Float[Array, "a b"]) -> Float[Array, "a b"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert "x" in specs[0].params

    def test_mixed_posonly_kwonly_args(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def f(x: Float[Array, "a b"], /, *, y: Float[Array, "c d"]) -> Float[Array, "a b"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert "x" in specs[0].params
        assert "y" in specs[0].params

    def test_async_function_name_col_offset(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            async def attention(x: Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        # "async def " is 10 chars; col_offset is 0 for top-level
        assert specs[0].name_col_offset == 10

    def test_sync_function_name_col_offset(self) -> None:
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def attention(x: Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        # "def " is 4 chars; col_offset is 0 for top-level
        assert specs[0].name_col_offset == 4

    def test_typing_tuple_return(self) -> None:
        source = textwrap.dedent("""\
            from typing import Tuple
            from jaxtyping import Array, Float

            def f(x: Float[Array, "a b"]) -> Tuple[Float[Array, "a b"], Float[Array, "c d"]]:
                pass
        """)
        specs = extract_function_specs(source, "test.py")
        assert len(specs) == 1
        assert specs[0].return_specs is not None
        assert len(specs[0].return_specs) == 2

    def test_lineno_tracking(self) -> None:
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


# ---------------------------------------------------------------------------
# extract_dim_locations
# ---------------------------------------------------------------------------


class TestExtractDimLocations:
    def test_basic_shape_string(self) -> None:
        source = 'def f(x: Float[Array, "batch seq d_model"]): pass\n'
        locs = extract_dim_locations(source, "test.py")
        names = [loc.dim_name for loc in locs]
        assert names == ["batch", "seq", "d_model"]
        # All on line 1, param "x", function "f"
        for loc in locs:
            assert loc.lineno == 1
            assert loc.param_name == "x"
            assert loc.function_name == "f"

    def test_column_offsets(self) -> None:
        # "batch seq" starts after the opening quote
        # def f(x: Float[Array, "batch seq"]): pass
        # 0123456789...
        source = 'def f(x: Float[Array, "batch seq"]): pass\n'
        locs = extract_dim_locations(source, "test.py")
        assert len(locs) == 2
        # "batch" starts at col 23 (col 22 is the quote, content starts at 23)
        assert locs[0].dim_name == "batch"
        assert locs[0].col_start == 23
        assert locs[0].col_end == 28
        # "seq" starts at 29
        assert locs[1].dim_name == "seq"
        assert locs[1].col_start == 29
        assert locs[1].col_end == 32

    def test_multiline_function(self) -> None:
        source = textwrap.dedent("""\
            def attention(
                q: Float[Array, "batch seq"],
                k: Float[Array, "batch seq"],
            ) -> Float[Array, "batch seq"]:
                pass
        """)
        locs = extract_dim_locations(source, "test.py")
        # 2 dims per param (q, k) + 2 dims in return = 6 total
        assert len(locs) == 6
        names = [loc.dim_name for loc in locs]
        assert names == ["batch", "seq", "batch", "seq", "batch", "seq"]
        # Check param names
        assert locs[0].param_name == "q"
        assert locs[1].param_name == "q"
        assert locs[2].param_name == "k"
        assert locs[3].param_name == "k"
        assert locs[4].param_name == "__return__"
        assert locs[5].param_name == "__return__"

    def test_variadic_dim(self) -> None:
        source = 'def f(x: Float[Array, "*batch seq"]): pass\n'
        locs = extract_dim_locations(source, "test.py")
        assert len(locs) == 2
        # The dim name should be "batch" (without *), but col_start points after the *
        assert locs[0].dim_name == "batch"
        # *batch starts at col 23, the 'b' of batch is at col 24
        assert locs[0].col_start == 24
        assert locs[0].col_end == 29  # end of "batch" token in "*batch"
        assert locs[1].dim_name == "seq"

    def test_fixed_dims_skipped(self) -> None:
        source = 'def f(x: Float[Array, "batch 4 seq"]): pass\n'
        locs = extract_dim_locations(source, "test.py")
        names = [loc.dim_name for loc in locs]
        assert names == ["batch", "seq"]

    def test_anonymous_and_ellipsis_skipped(self) -> None:
        source = 'def f(x: Float[Array, "... _ batch"]): pass\n'
        locs = extract_dim_locations(source, "test.py")
        assert len(locs) == 1
        assert locs[0].dim_name == "batch"

    def test_return_annotation(self) -> None:
        source = 'def f(x: Float[Array, "a"]) -> Float[Array, "b c"]: pass\n'
        locs = extract_dim_locations(source, "test.py")
        assert len(locs) == 3
        assert locs[0].param_name == "x"
        assert locs[0].dim_name == "a"
        assert locs[1].param_name == "__return__"
        assert locs[1].dim_name == "b"
        assert locs[2].param_name == "__return__"
        assert locs[2].dim_name == "c"

    def test_class_method(self) -> None:
        source = textwrap.dedent("""\
            class Model:
                def __call__(self, x: Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
                    pass
        """)
        locs = extract_dim_locations(source, "test.py")
        assert len(locs) == 4
        # self should be skipped
        for loc in locs:
            assert loc.function_name == "__call__"
            assert loc.param_name in ("x", "__return__")

    def test_no_jaxtyping_returns_empty(self) -> None:
        source = "def f(x: int) -> int: pass\n"
        locs = extract_dim_locations(source, "test.py")
        assert locs == []

    def test_syntax_error_returns_empty(self) -> None:
        source = "def f(x: Float[Array, :"
        locs = extract_dim_locations(source, "test.py")
        assert locs == []


# ---------------------------------------------------------------------------
# extract_call_sites
# ---------------------------------------------------------------------------


class TestExtractCallSites:
    def test_basic_call(self) -> None:
        source = textwrap.dedent("""\
            def encode(x: Float[Array, "batch dim"]): pass
            def pipeline(x: Float[Array, "batch dim"]):
                return encode(x)
        """)
        sites = extract_call_sites(source, "test.py", {"encode"})
        assert len(sites) == 1
        assert sites[0].caller_name == "pipeline"
        assert sites[0].callee_name == "encode"
        assert sites[0].lineno == 3

    def test_multiple_calls(self) -> None:
        source = textwrap.dedent("""\
            def encode(x): pass
            def decode(x): pass
            def autoencoder(x):
                h = encode(x)
                return decode(h)
        """)
        sites = extract_call_sites(source, "test.py", {"encode", "decode"})
        assert len(sites) == 2
        callees = {s.callee_name for s in sites}
        assert callees == {"encode", "decode"}
        for s in sites:
            assert s.caller_name == "autoencoder"

    def test_unknown_function_ignored(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                return unknown(x)
        """)
        sites = extract_call_sites(source, "test.py", {"encode"})
        assert len(sites) == 0

    def test_attribute_call(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                return self.encode(x)
        """)
        # ast.Attribute callee: attr is "encode"
        sites = extract_call_sites(source, "test.py", {"encode"})
        assert len(sites) == 1
        assert sites[0].callee_name == "encode"

    def test_class_method_caller(self) -> None:
        source = textwrap.dedent("""\
            class Model:
                def forward(self, x):
                    return encode(x)
        """)
        sites = extract_call_sites(source, "test.py", {"encode"})
        assert len(sites) == 1
        assert sites[0].caller_name == "Model.forward"

    def test_col_offsets(self) -> None:
        source = "def f(x):\n    return encode(x)\n"
        sites = extract_call_sites(source, "test.py", {"encode"})
        assert len(sites) == 1
        assert sites[0].col_offset == 11
        assert sites[0].end_col_offset == 17  # "encode" is 6 chars

    def test_syntax_error_returns_empty(self) -> None:
        sites = extract_call_sites("def f(:", "test.py", {"encode"})
        assert sites == []

    def test_empty_known_functions(self) -> None:
        source = "def f(x): return encode(x)\n"
        sites = extract_call_sites(source, "test.py", set())
        assert sites == []

    def test_nested_call_in_expression(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                y = 1 + encode(x) * 2
                return y
        """)
        sites = extract_call_sites(source, "test.py", {"encode"})
        assert len(sites) == 1
        assert sites[0].callee_name == "encode"


# ---------------------------------------------------------------------------
# extract_all_function_defs
# ---------------------------------------------------------------------------


def test_extract_all_function_defs_includes_non_annotated() -> None:
    from jaxtyc.analyzer.annotations import extract_all_function_defs

    source = textwrap.dedent("""\
        def helper(x):
            return x + 1

        def encode(x: Float[Array, "batch d"]) -> Float[Array, "batch d"]:
            return helper(x)

        class Model:
            def forward(self, x):
                return x
    """)
    defs = extract_all_function_defs(source, "/test.py")
    names = [d.name for d in defs]
    assert "helper" in names
    assert "encode" in names
    assert "forward" in names
    fwd = next(d for d in defs if d.name == "forward")
    assert fwd.is_method is True
    assert fwd.class_name == "Model"


def test_extract_all_function_defs_nested_not_method() -> None:
    """Nested functions inside class methods should NOT be marked as methods."""
    from jaxtyc.analyzer.annotations import extract_all_function_defs

    source = textwrap.dedent("""\
        class Model:
            def forward(self, x):
                def _local_helper(y):
                    return y + 1
                return _local_helper(x)
    """)
    defs = extract_all_function_defs(source, "/test.py")
    fwd = next(d for d in defs if d.name == "forward")
    assert fwd.is_method is True
    assert fwd.class_name == "Model"
    helper = next(d for d in defs if d.name == "_local_helper")
    assert helper.is_method is False
    assert helper.class_name is None


def test_extract_call_sites_include_external() -> None:
    """include_external=True captures calls to non-workspace functions."""
    from jaxtyc.analyzer.annotations import extract_call_sites

    source = textwrap.dedent("""\
        import jax.numpy as jnp

        def transform(x):
            y = jnp.dot(x, x)
            return y
    """)
    known = {"transform"}

    # Without include_external: dot is not in known_functions, not captured
    calls = extract_call_sites(source, "/test.py", known)
    assert not any(c.callee_name == "dot" for c in calls)

    # With include_external: dot IS captured
    calls_ext = extract_call_sites(source, "/test.py", known, include_external=True)
    assert any(c.callee_name == "dot" for c in calls_ext)
    dot_call = next(c for c in calls_ext if c.callee_name == "dot")
    assert dot_call.caller_name == "transform"


def test_extract_call_sites_include_external_bare_name() -> None:
    """include_external captures bare function calls too."""
    from jaxtyc.analyzer.annotations import extract_call_sites

    source = textwrap.dedent("""\
        def process(x):
            y = len(x)
            return print(y)
    """)
    known = {"process"}
    calls = extract_call_sites(source, "/test.py", known, include_external=True)
    callee_names = {c.callee_name for c in calls}
    assert "len" in callee_names
    assert "print" in callee_names


def test_extract_call_sites_qualified_name() -> None:
    """Attribute calls store the full dotted path in callee_qualified_name."""
    from jaxtyc.analyzer.annotations import extract_call_sites

    source = textwrap.dedent("""\
        import jax.numpy as jnp

        def transform(x):
            y = jnp.matmul(x, x)
            z = jnp.lax.scan(lambda c, x: (c, x), y, y)
            w = encode(z)
            return w
    """)
    known = {"encode"}
    calls = extract_call_sites(source, "/test.py", known, include_external=True)

    matmul_call = next(c for c in calls if c.callee_name == "matmul")
    assert matmul_call.callee_qualified_name == "jnp.matmul"

    scan_call = next(c for c in calls if c.callee_name == "scan")
    assert scan_call.callee_qualified_name == "jnp.lax.scan"

    # Bare name calls have no qualified name
    encode_call = next(c for c in calls if c.callee_name == "encode")
    assert encode_call.callee_qualified_name is None
