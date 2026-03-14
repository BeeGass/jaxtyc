"""Tests for jaxtyc.lsp.index — FileIndex, WorkspaceIndex, build_file_index."""

from __future__ import annotations

import textwrap

from jaxtyc.lsp.index import FileIndex
from jaxtyc.lsp.index import WorkspaceIndex
from jaxtyc.lsp.index import build_file_index
from jaxtyc.types import CallSite
from jaxtyc.types import DimLocation
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import ShapeSpec


def _make_dim(
    dim_name: str,
    param_name: str,
    function_name: str,
    lineno: int,
    col_start: int,
    col_end: int,
) -> DimLocation:
    return DimLocation(
        dim_name=dim_name,
        param_name=param_name,
        function_name=function_name,
        file_path="test.py",
        lineno=lineno,
        col_start=col_start,
        col_end=col_end,
    )


class TestBuildFileIndex:
    def test_builds_from_source(self):
        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def f(x: Float[Array, "batch seq"]) -> Float[Array, "batch seq"]:
                pass
        """)
        idx = build_file_index(source, "test.py", "file:///test.py")
        assert len(idx.function_specs) == 1
        assert idx.function_specs[0].name == "f"
        assert len(idx.dim_locations) == 4  # batch, seq in param + batch, seq in return

    def test_reuses_func_specs(self):
        source = 'def f(x: Float[Array, "a b"]): pass\n'
        specs = [
            FunctionShapeSpec(
                name="f",
                file_path="test.py",
                lineno=1,
                col_offset=0,
                params={},
                return_spec=None,
            )
        ]
        idx = build_file_index(source, "test.py", "file:///test.py", func_specs=specs)
        assert list(idx.function_specs) == specs
        assert len(idx.dim_locations) == 2  # still extracts dims from source


class TestWorkspaceIndex:
    def _make_index(self) -> tuple[WorkspaceIndex, FileIndex]:
        ws = WorkspaceIndex()
        dims = [
            _make_dim("batch", "q", "attention", 3, 22, 27),
            _make_dim("seq", "q", "attention", 3, 28, 31),
            _make_dim("batch", "k", "attention", 4, 22, 27),
            _make_dim("seq", "k", "attention", 4, 28, 31),
            _make_dim("batch", "__return__", "attention", 5, 22, 27),
            _make_dim("seq", "__return__", "attention", 5, 28, 31),
        ]
        spec = FunctionShapeSpec(
            name="attention",
            file_path="test.py",
            lineno=2,
            col_offset=0,
            params={
                "q": ShapeSpec(dims=(), dtype="float32"),
                "k": ShapeSpec(dims=(), dtype="float32"),
            },
            return_spec=ShapeSpec(dims=(), dtype="float32"),
        )
        fi = FileIndex(
            file_path="test.py",
            uri="file:///test.py",
            function_specs=[spec],
            dim_locations=dims,
            call_sites=[],
        )
        ws.update_file(fi)
        return ws, fi

    def test_update_and_get_file(self):
        ws, fi = self._make_index()
        assert ws.get_file("file:///test.py") is fi
        assert ws.get_file("file:///other.py") is None

    def test_remove_file(self):
        ws, _ = self._make_index()
        ws.remove_file("file:///test.py")
        assert ws.get_file("file:///test.py") is None

    def test_find_function_at(self):
        ws, _ = self._make_index()
        spec = ws.find_function_at("file:///test.py", 2, 0)
        assert spec is not None
        assert spec.name == "attention"
        assert ws.find_function_at("file:///test.py", 99, 0) is None

    def test_find_dim_at(self):
        ws, _ = self._make_index()
        # "batch" at line 3, cols [22, 27) — col_end is exclusive
        dim = ws.find_dim_at("file:///test.py", 3, 23)
        assert dim is not None
        assert dim.dim_name == "batch"
        # Outside range
        assert ws.find_dim_at("file:///test.py", 3, 50) is None
        # Exact start
        dim = ws.find_dim_at("file:///test.py", 3, 22)
        assert dim is not None
        assert dim.dim_name == "batch"
        # Gap between batch and seq (col 27 is past batch, before seq at 28)
        assert ws.find_dim_at("file:///test.py", 3, 27) is None

    def test_find_dim_at_boundary(self):
        ws, _ = self._make_index()
        # batch: col_start=22, col_end=27 -> valid range is [22, 26]
        assert ws.find_dim_at("file:///test.py", 3, 22) is not None
        assert ws.find_dim_at("file:///test.py", 3, 26) is not None
        assert ws.find_dim_at("file:///test.py", 3, 27) is None  # one past end
        # seq: col_start=28, col_end=31
        assert ws.find_dim_at("file:///test.py", 3, 28) is not None
        assert ws.find_dim_at("file:///test.py", 3, 30) is not None
        assert ws.find_dim_at("file:///test.py", 3, 31) is None

    def test_find_dim_definition(self):
        ws, _ = self._make_index()
        defn = ws.find_dim_definition("batch", "attention", "file:///test.py")
        assert defn is not None
        assert defn.param_name == "q"
        assert defn.lineno == 3

    def test_find_dim_definition_missing(self):
        ws, _ = self._make_index()
        assert ws.find_dim_definition("nonexistent", "attention", "file:///test.py") is None

    def test_find_dim_references_in_function(self):
        ws, _ = self._make_index()
        refs = ws.find_dim_references_in_function("batch", "attention", "file:///test.py")
        assert len(refs) == 3  # q, k, return
        params = [r.param_name for r in refs]
        assert "q" in params
        assert "k" in params
        assert "__return__" in params

    def test_find_all_dim_references_scoped(self):
        ws, _ = self._make_index()
        refs = ws.find_all_dim_references("seq", uri="file:///test.py")
        assert len(refs) == 3

    def test_find_all_dim_references_global(self):
        ws, _ = self._make_index()
        refs = ws.find_all_dim_references("batch")
        assert len(refs) == 3

    def test_find_function_by_name(self):
        ws, _ = self._make_index()
        funcs = ws.find_function_by_name("attention")
        assert len(funcs) == 1
        assert funcs[0].name == "attention"
        assert ws.find_function_by_name("nonexistent") == []

    def test_search_symbols(self):
        ws, _ = self._make_index()
        results = ws.search_symbols("att")
        assert len(results) == 1
        assert results[0].name == "attention"
        # Case insensitive
        results = ws.search_symbols("ATT")
        assert len(results) == 1
        # No match
        assert ws.search_symbols("zzz") == []

    def test_search_symbols_cap(self):
        ws = WorkspaceIndex()
        for i in range(100):
            fi = FileIndex(
                file_path=f"f{i}.py",
                uri=f"file:///f{i}.py",
                function_specs=[
                    FunctionShapeSpec(
                        name=f"func_{i}",
                        file_path=f"f{i}.py",
                        lineno=1,
                        col_offset=0,
                        params={},
                        return_spec=None,
                    )
                ],
                dim_locations=[],
                call_sites=[],
            )
            ws.update_file(fi)
        results = ws.search_symbols("func")
        assert len(results) == 50

    def test_get_callers_of(self):
        ws = WorkspaceIndex()
        fi = FileIndex(
            file_path="test.py",
            uri="file:///test.py",
            function_specs=[],
            dim_locations=[],
            call_sites=[
                CallSite("autoencoder", "encode", "test.py", 10, 4, 10),
                CallSite("autoencoder", "decode", "test.py", 11, 4, 10),
            ],
        )
        ws.update_file(fi)
        callers = ws.get_callers_of("encode", "file:///test.py")
        assert len(callers) == 1
        assert callers[0].caller_name == "autoencoder"

    def test_get_callees_of(self):
        ws = WorkspaceIndex()
        fi = FileIndex(
            file_path="test.py",
            uri="file:///test.py",
            function_specs=[],
            dim_locations=[],
            call_sites=[
                CallSite("autoencoder", "encode", "test.py", 10, 4, 10),
                CallSite("autoencoder", "decode", "test.py", 11, 4, 10),
            ],
        )
        ws.update_file(fi)
        callees = ws.get_callees_of("autoencoder", "file:///test.py")
        assert len(callees) == 2
        names = {c.callee_name for c in callees}
        assert names == {"encode", "decode"}


def test_find_call_site_at() -> None:
    """find_call_site_at should match position within a call site range."""
    idx = WorkspaceIndex()
    cs = CallSite(
        caller_name="main",
        callee_name="encode",
        file_path="/test.py",
        lineno=10,
        col_offset=4,
        end_col_offset=10,
    )
    fi = FileIndex(
        file_path="/test.py",
        uri="file:///test.py",
        function_specs=[],
        dim_locations=[],
        call_sites=[cs],
    )
    idx.update_file(fi)

    # Inside range
    found = idx.find_call_site_at("file:///test.py", 10, 6)
    assert found is not None
    assert found.callee_name == "encode"

    # At start boundary
    found = idx.find_call_site_at("file:///test.py", 10, 4)
    assert found is not None

    # At end boundary (exclusive)
    not_found = idx.find_call_site_at("file:///test.py", 10, 10)
    assert not_found is None

    # Wrong line
    not_found = idx.find_call_site_at("file:///test.py", 11, 6)
    assert not_found is None

    # Unknown URI
    not_found = idx.find_call_site_at("file:///other.py", 10, 6)
    assert not_found is None


def test_find_function_def_by_name() -> None:
    from jaxtyc.types import FunctionDefInfo

    idx = WorkspaceIndex()
    fdef = FunctionDefInfo(
        name="helper", file_path="/a.py", lineno=3, col_offset=0, name_col_offset=4
    )
    idx.update_file(
        FileIndex(
            file_path="/a.py",
            uri="file:///a.py",
            function_specs=[],
            dim_locations=[],
            call_sites=[],
            function_defs=[fdef],
        )
    )
    results = idx.find_function_def_by_name("helper")
    assert len(results) == 1
    assert results[0].name == "helper"


def test_find_function_by_name_preferred_uri() -> None:
    """preferred_uri should sort matching specs so the preferred file comes first."""
    idx = WorkspaceIndex()
    spec_a = FunctionShapeSpec(
        name="encode",
        file_path="/a.py",
        lineno=5,
        col_offset=0,
        params={},
        return_spec=None,
    )
    spec_b = FunctionShapeSpec(
        name="encode",
        file_path="/b.py",
        lineno=10,
        col_offset=0,
        params={},
        return_spec=None,
    )
    idx.update_file(
        FileIndex(
            file_path="/a.py",
            uri="file:///a.py",
            function_specs=[spec_a],
            dim_locations=[],
            call_sites=[],
        )
    )
    idx.update_file(
        FileIndex(
            file_path="/b.py",
            uri="file:///b.py",
            function_specs=[spec_b],
            dim_locations=[],
            call_sites=[],
        )
    )
    # Without preferred_uri, returns both
    assert len(idx.find_function_by_name("encode")) == 2
    # With preferred_uri, preferred file's spec comes first
    results_a = idx.find_function_by_name("encode", preferred_uri="file:///a.py")
    assert results_a[0].file_path == "/a.py"
    results_b = idx.find_function_by_name("encode", preferred_uri="file:///b.py")
    assert results_b[0].file_path == "/b.py"
    # Unknown preferred_uri still returns all
    assert len(idx.find_function_by_name("encode", preferred_uri="file:///c.py")) == 2


def test_search_symbols_includes_non_annotated() -> None:
    """search_symbols should find non-annotated functions via function_defs."""
    from jaxtyc.types import FunctionDefInfo

    idx = WorkspaceIndex()
    helper_def = FunctionDefInfo(
        name="normalize",
        file_path="/a.py",
        lineno=3,
        col_offset=0,
        name_col_offset=4,
    )
    spec = FunctionShapeSpec(
        name="encode",
        file_path="/a.py",
        lineno=10,
        col_offset=0,
        params={},
        return_spec=None,
    )
    idx.update_file(
        FileIndex(
            file_path="/a.py",
            uri="file:///a.py",
            function_specs=[spec],
            dim_locations=[],
            call_sites=[],
            function_defs=[
                helper_def,
                FunctionDefInfo(
                    name="encode",
                    file_path="/a.py",
                    lineno=10,
                    col_offset=0,
                    name_col_offset=4,
                ),
            ],
        )
    )
    # Annotated function found
    assert len(idx.search_symbols("encode")) == 1
    # Non-annotated function should also be found
    results = idx.search_symbols("normalize")
    assert len(results) == 1
    assert results[0].name == "normalize"
    # Empty query returns all
    all_results = idx.search_symbols("")
    names = {r.name for r in all_results}
    assert "encode" in names
    assert "normalize" in names


def test_find_dim_definition_file_scoped() -> None:
    """find_dim_definition without function_name should search the entire file."""
    ws = WorkspaceIndex()
    dims = [
        _make_dim("batch", "x", "encode", 5, 22, 27),
        _make_dim("batch", "__return__", "encode", 6, 20, 25),
        _make_dim("batch", "h", "decode", 10, 22, 27),
        _make_dim("batch", "__return__", "decode", 11, 20, 25),
    ]
    fi = FileIndex(
        file_path="test.py",
        uri="file:///test.py",
        function_specs=[],
        dim_locations=dims,
        call_sites=[],
    )
    ws.update_file(fi)
    # Function-scoped: first batch in decode is line 10
    defn = ws.find_dim_definition("batch", "decode", "file:///test.py")
    assert defn is not None
    assert defn.lineno == 10
    # File-scoped: first batch in the entire file is line 5
    defn_file = ws.find_dim_definition_in_file("batch", "file:///test.py")
    assert defn_file is not None
    assert defn_file.lineno == 5


def test_get_callers_of_scoped_to_file() -> None:
    """get_callers_of should only return call sites from the specified file."""
    ws = WorkspaceIndex()
    cs_a = CallSite("main_a", "decode", "/a.py", 10, 4, 10)
    cs_b = CallSite("main_b", "decode", "/b.py", 20, 4, 10)
    ws.update_file(
        FileIndex(
            file_path="/a.py",
            uri="file:///a.py",
            function_specs=[],
            dim_locations=[],
            call_sites=[cs_a],
        )
    )
    ws.update_file(
        FileIndex(
            file_path="/b.py",
            uri="file:///b.py",
            function_specs=[],
            dim_locations=[],
            call_sites=[cs_b],
        )
    )
    # File-scoped: only call sites in file A
    callers_a = ws.get_callers_of("decode", "file:///a.py")
    assert len(callers_a) == 1
    assert callers_a[0].caller_name == "main_a"
    # File-scoped: only call sites in file B
    callers_b = ws.get_callers_of("decode", "file:///b.py")
    assert len(callers_b) == 1
    assert callers_b[0].caller_name == "main_b"
    # Unknown URI returns empty
    assert ws.get_callers_of("decode", "file:///unknown.py") == []


def test_get_all_callers_of() -> None:
    """get_all_callers_of returns call sites from all files."""
    ws = WorkspaceIndex()
    cs_a = CallSite("main_a", "decode", "/a.py", 10, 4, 10)
    cs_b = CallSite("main_b", "decode", "/b.py", 20, 4, 10)
    ws.update_file(
        FileIndex(
            file_path="/a.py",
            uri="file:///a.py",
            function_specs=[],
            dim_locations=[],
            call_sites=[cs_a],
        )
    )
    ws.update_file(
        FileIndex(
            file_path="/b.py",
            uri="file:///b.py",
            function_specs=[],
            dim_locations=[],
            call_sites=[cs_b],
        )
    )
    all_callers = ws.get_all_callers_of("decode")
    assert len(all_callers) == 2
    names = {c.caller_name for c in all_callers}
    assert names == {"main_a", "main_b"}
