"""Per-file and workspace-level index for LSP navigation features."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from dataclasses import field

from jaxtyc.analyzer.annotations import extract_call_sites
from jaxtyc.analyzer.annotations import extract_dim_locations
from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.types import CallSite
from jaxtyc.types import DimLocation
from jaxtyc.types import FunctionShapeSpec


@dataclass
class FileIndex:
    """Per-file index of shape-annotated symbols for LSP navigation."""

    file_path: str
    uri: str
    function_specs: list[FunctionShapeSpec]
    dim_locations: list[DimLocation]
    call_sites: list[CallSite] = field(default_factory=list)


def build_file_index(
    source: str,
    file_path: str,
    uri: str,
    func_specs: list[FunctionShapeSpec] | None = None,
) -> FileIndex:
    """Build a complete FileIndex from source code.

    If func_specs is provided, reuses them instead of re-parsing.
    """
    if func_specs is None:
        func_specs = extract_function_specs(source, file_path)
    dim_locs = extract_dim_locations(source, file_path)
    known_names = {s.name for s in func_specs}
    call_sites = extract_call_sites(source, file_path, known_names)
    return FileIndex(
        file_path=file_path,
        uri=uri,
        function_specs=func_specs,
        dim_locations=dim_locs,
        call_sites=call_sites,
    )


class WorkspaceIndex:
    """Thread-safe workspace-level index for cross-file navigation queries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._files: dict[str, FileIndex] = {}

    def update_file(self, file_index: FileIndex) -> None:
        with self._lock:
            self._files[file_index.uri] = file_index

    def remove_file(self, uri: str) -> None:
        with self._lock:
            self._files.pop(uri, None)

    def get_file(self, uri: str) -> FileIndex | None:
        with self._lock:
            return self._files.get(uri)

    def all_files(self) -> list[FileIndex]:
        with self._lock:
            return list(self._files.values())

    def find_function_at(self, uri: str, line: int, col: int) -> FunctionShapeSpec | None:
        """Find the function whose def line contains the given 1-based line."""
        idx = self.get_file(uri)
        if idx is None:
            return None
        for spec in idx.function_specs:
            if spec.lineno == line:
                return spec
        return None

    def find_dim_at(self, uri: str, line: int, col: int) -> DimLocation | None:
        """Find a DimLocation at the given 1-based line, 0-based col."""
        idx = self.get_file(uri)
        if idx is None:
            return None
        for dim in idx.dim_locations:
            if dim.lineno == line and dim.col_start <= col < dim.col_end:
                return dim
        return None

    def find_dim_definition(
        self, dim_name: str, function_name: str, uri: str
    ) -> DimLocation | None:
        """Find the first occurrence of a dim name in a function (the 'definition')."""
        idx = self.get_file(uri)
        if idx is None:
            return None
        for dim in idx.dim_locations:
            if dim.dim_name == dim_name and dim.function_name == function_name:
                return dim
        return None

    def find_dim_references_in_function(
        self, dim_name: str, function_name: str, uri: str
    ) -> list[DimLocation]:
        """Find all occurrences of a dim name within a specific function."""
        idx = self.get_file(uri)
        if idx is None:
            return []
        return [
            d
            for d in idx.dim_locations
            if d.dim_name == dim_name and d.function_name == function_name
        ]

    def find_all_dim_references(self, dim_name: str, uri: str | None = None) -> list[DimLocation]:
        """Find all occurrences of a dim name, optionally scoped to a URI."""
        with self._lock:
            files = [self._files[uri]] if uri and uri in self._files else list(self._files.values())
        results: list[DimLocation] = []
        for idx in files:
            results.extend(d for d in idx.dim_locations if d.dim_name == dim_name)
        return results

    def find_function_by_name(self, name: str) -> list[FunctionShapeSpec]:
        """Find all functions with the given name across the workspace."""
        with self._lock:
            files = list(self._files.values())
        results: list[FunctionShapeSpec] = []
        for idx in files:
            results.extend(s for s in idx.function_specs if s.name == name)
        return results

    def search_symbols(self, query: str) -> list[FunctionShapeSpec]:
        """Search for functions whose names contain the query (case-insensitive)."""
        q = query.lower()
        with self._lock:
            files = list(self._files.values())
        results: list[FunctionShapeSpec] = []
        for idx in files:
            for spec in idx.function_specs:
                if q in spec.name.lower():
                    results.append(spec)
                    if len(results) >= 50:
                        return results
        return results

    def get_callers_of(self, function_name: str, uri: str) -> list[CallSite]:
        """Find all call sites where the given function is called."""
        with self._lock:
            files = list(self._files.values())
        results: list[CallSite] = []
        for idx in files:
            results.extend(c for c in idx.call_sites if c.callee_name == function_name)
        return results

    def get_callees_of(self, function_name: str, uri: str) -> list[CallSite]:
        """Find all functions called by the given function."""
        idx = self.get_file(uri)
        if idx is None:
            return []
        return [c for c in idx.call_sites if c.caller_name == function_name]
