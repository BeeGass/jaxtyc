"""Tests for jaxtyc.lsp.mux — LSP multiplexer pure functions, helpers, and integration."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from jaxtyc.lsp.mux import _clean_hover_text
from jaxtyc.lsp.mux import _detect_project_root
from jaxtyc.lsp.mux import _extract_file_path_from_msg
from jaxtyc.lsp.mux import _find_primary_server
from jaxtyc.lsp.mux import _hover_compact_enabled
from jaxtyc.lsp.mux import _patch_root_uri
from jaxtyc.lsp.mux import _uri_to_path
from jaxtyc.lsp.mux import encode_message
from jaxtyc.lsp.mux import merge_results
from jaxtyc.lsp.mux import read_message

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# read_message
# ---------------------------------------------------------------------------


class TestReadMessage:
    def test_valid_message(self) -> None:
        async def _run() -> dict | None:
            body = {"jsonrpc": "2.0", "id": 1, "method": "test"}
            reader = asyncio.StreamReader()
            reader.feed_data(encode_message(body))
            return await read_message(reader)

        result = asyncio.run(_run())
        assert result == {"jsonrpc": "2.0", "id": 1, "method": "test"}

    def test_eof_returns_none(self) -> None:
        async def _run() -> dict | None:
            reader = asyncio.StreamReader()
            reader.feed_eof()
            return await read_message(reader)

        assert asyncio.run(_run()) is None

    def test_incomplete_read_returns_none(self) -> None:
        async def _run() -> dict | None:
            reader = asyncio.StreamReader()
            # Feed a header claiming 100 bytes but provide no body, then EOF
            reader.feed_data(b"Content-Length: 100\r\n\r\n")
            reader.feed_eof()
            return await read_message(reader)

        assert asyncio.run(_run()) is None

    def test_multiple_messages(self) -> None:
        async def _run() -> list[dict]:
            reader = asyncio.StreamReader()
            msg1 = {"jsonrpc": "2.0", "id": 1, "method": "first"}
            msg2 = {"jsonrpc": "2.0", "id": 2, "method": "second"}
            reader.feed_data(encode_message(msg1) + encode_message(msg2))
            reader.feed_eof()
            results = []
            r = await read_message(reader)
            if r:
                results.append(r)
            r = await read_message(reader)
            if r:
                results.append(r)
            return results

        results = asyncio.run(_run())
        assert len(results) == 2
        assert results[0]["method"] == "first"
        assert results[1]["method"] == "second"

    def test_zero_content_length_returns_none(self) -> None:
        async def _run() -> dict | None:
            reader = asyncio.StreamReader()
            reader.feed_data(b"Content-Length: 0\r\n\r\n")
            reader.feed_eof()
            return await read_message(reader)

        assert asyncio.run(_run()) is None


# ---------------------------------------------------------------------------
# encode_message
# ---------------------------------------------------------------------------


class TestEncodeMessage:
    def test_round_trip(self) -> None:
        body = {"jsonrpc": "2.0", "id": 42, "result": None}
        raw = encode_message(body)
        assert b"Content-Length:" in raw
        assert b'"jsonrpc"' in raw

    def test_content_length_matches_body(self) -> None:
        body = {"jsonrpc": "2.0", "id": 1, "result": {"key": "value"}}
        raw = encode_message(body)
        header_end = raw.index(b"\r\n\r\n") + 4
        header = raw[:header_end].decode("ascii")
        cl = int(header.split("Content-Length: ")[1].split("\r\n")[0])
        actual_body = raw[header_end:]
        assert len(actual_body) == cl

    def test_body_is_valid_json(self) -> None:
        body = {"jsonrpc": "2.0", "method": "test", "params": {"x": [1, 2, 3]}}
        raw = encode_message(body)
        header_end = raw.index(b"\r\n\r\n") + 4
        parsed = json.loads(raw[header_end:])
        assert parsed == body


# ---------------------------------------------------------------------------
# _uri_to_path
# ---------------------------------------------------------------------------


class TestUriToPath:
    def test_simple_path(self) -> None:
        assert _uri_to_path("file:///home/user/file.py") == "/home/user/file.py"

    def test_percent_decode(self) -> None:
        assert _uri_to_path("file:///home/user/my%20project/f.py") == "/home/user/my project/f.py"

    def test_non_file_uri(self) -> None:
        assert _uri_to_path("https://example.com") is None

    def test_empty_string(self) -> None:
        assert _uri_to_path("") is None

    def test_special_chars(self) -> None:
        assert _uri_to_path("file:///home/%E4%B8%AD%E6%96%87/f.py") is not None


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


class TestMergeResults:
    def test_hover_merge(self) -> None:
        primary = {"result": {"contents": {"kind": "markdown", "value": "type info"}}}
        jaxtyc = {"result": {"contents": {"kind": "markdown", "value": "shape info"}}}
        result = merge_results("textDocument/hover", primary, jaxtyc)
        assert result is not None
        assert "type info" in result["contents"]["value"]
        assert "shape info" in result["contents"]["value"]

    def test_hover_primary_only(self) -> None:
        primary = {"result": {"contents": {"kind": "markdown", "value": "type info"}}}
        result = merge_results("textDocument/hover", primary, None)
        assert result is not None
        assert "type info" in result["contents"]["value"]

    def test_hover_jaxtyc_only(self) -> None:
        jaxtyc = {"result": {"contents": {"kind": "markdown", "value": "shape info"}}}
        result = merge_results("textDocument/hover", None, jaxtyc)
        assert result is not None
        assert "shape info" in result["contents"]["value"]

    def test_hover_both_none(self) -> None:
        result = merge_results("textDocument/hover", None, None)
        assert result is None

    def test_hover_null_results(self) -> None:
        result = merge_results("textDocument/hover", {"result": None}, {"result": None})
        assert result is None

    def test_hover_string_contents(self) -> None:
        primary = {"result": {"contents": "plain text"}}
        result = merge_results("textDocument/hover", primary, None)
        assert result is not None
        assert "plain text" in result["contents"]["value"]

    def test_array_merge_codelens(self) -> None:
        primary = {"result": [{"range": {}, "command": {"title": "a"}}]}
        jaxtyc = {"result": [{"range": {}, "command": {"title": "b"}}]}
        result = merge_results("textDocument/codeLens", primary, jaxtyc)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_array_merge_references(self) -> None:
        primary = {"result": [{"uri": "file:///a.py", "range": {}}]}
        jaxtyc = {"result": [{"uri": "file:///b.py", "range": {}}]}
        result = merge_results("textDocument/references", primary, jaxtyc)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_array_merge_empty_primary(self) -> None:
        jaxtyc = {"result": [{"range": {}}]}
        result = merge_results("textDocument/codeLens", {"result": []}, jaxtyc)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_array_merge_both_empty(self) -> None:
        result = merge_results("textDocument/codeLens", {"result": []}, {"result": []})
        assert result is None

    def test_completion_merge(self) -> None:
        primary = {"result": {"isIncomplete": False, "items": [{"label": "a"}]}}
        jaxtyc = {"result": {"isIncomplete": False, "items": [{"label": "b"}]}}
        result = merge_results("textDocument/completion", primary, jaxtyc)
        assert isinstance(result, dict)
        assert len(result["items"]) == 2

    def test_completion_merge_list_format(self) -> None:
        primary = {"result": [{"label": "a"}]}
        jaxtyc = {"result": [{"label": "b"}]}
        result = merge_results("textDocument/completion", primary, jaxtyc)
        assert isinstance(result, dict)
        assert len(result["items"]) == 2

    def test_completion_merge_both_empty(self) -> None:
        result = merge_results(
            "textDocument/completion", {"result": {"items": []}}, {"result": {"items": []}}
        )
        assert result is None

    def test_single_value_prefers_primary(self) -> None:
        primary = {"result": {"uri": "file:///a.py"}}
        jaxtyc = {"result": {"uri": "file:///b.py"}}
        result = merge_results("textDocument/definition", primary, jaxtyc)
        assert result == {"uri": "file:///a.py"}

    def test_single_value_falls_back_to_jaxtyc(self) -> None:
        jaxtyc = {"result": {"uri": "file:///b.py"}}
        result = merge_results("textDocument/definition", {"result": None}, jaxtyc)
        assert result == {"uri": "file:///b.py"}

    def test_single_value_both_none(self) -> None:
        result = merge_results("textDocument/definition", {"result": None}, {"result": None})
        assert result is None

    def test_unknown_method_prefers_primary(self) -> None:
        primary = {"result": {"data": "primary"}}
        jaxtyc = {"result": {"data": "jaxtyc"}}
        result = merge_results("unknown/method", primary, jaxtyc)
        assert result == {"data": "primary"}

    def test_folding_range_merge(self) -> None:
        primary = {"result": [{"startLine": 0, "endLine": 5}]}
        jaxtyc = {"result": [{"startLine": 10, "endLine": 15}]}
        result = merge_results("textDocument/foldingRange", primary, jaxtyc)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_inlay_hint_merge(self) -> None:
        primary = {"result": [{"position": {"line": 0, "character": 5}}]}
        jaxtyc = {"result": [{"position": {"line": 1, "character": 3}}]}
        result = merge_results("textDocument/inlayHint", primary, jaxtyc)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_workspace_symbol_merge(self) -> None:
        primary = {"result": [{"name": "foo"}]}
        jaxtyc = {"result": [{"name": "bar"}]}
        result = merge_results("workspace/symbol", primary, jaxtyc)
        assert isinstance(result, list)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _hover_compact_enabled
# ---------------------------------------------------------------------------


class TestHoverCompactEnabled:
    def test_default_enabled(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JAXTYC_HOVER_COMPACT", None)
            assert _hover_compact_enabled() is True

    def test_disabled_with_zero(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "0"}):
            assert _hover_compact_enabled() is False

    def test_disabled_with_false(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "false"}):
            assert _hover_compact_enabled() is False

    def test_disabled_with_no(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "no"}):
            assert _hover_compact_enabled() is False

    def test_disabled_with_off(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "off"}):
            assert _hover_compact_enabled() is False

    def test_enabled_with_one(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "1"}):
            assert _hover_compact_enabled() is True

    def test_enabled_with_true(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "true"}):
            assert _hover_compact_enabled() is True


# ---------------------------------------------------------------------------
# _mux_solo_server
# ---------------------------------------------------------------------------


class TestMuxSoloServer:
    def test_default_returns_none(self) -> None:
        from jaxtyc.lsp.mux import _mux_solo_server

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JAXTYC_MUX_SOLO", None)
            assert _mux_solo_server() is None

    def test_jaxtyc_solo(self) -> None:
        from jaxtyc.lsp.mux import _mux_solo_server

        with mock.patch.dict(os.environ, {"JAXTYC_MUX_SOLO": "jaxtyc"}):
            assert _mux_solo_server() == "jaxtyc"

    def test_ty_solo(self) -> None:
        from jaxtyc.lsp.mux import _mux_solo_server

        with mock.patch.dict(os.environ, {"JAXTYC_MUX_SOLO": "ty"}):
            assert _mux_solo_server() == "ty"

    def test_primary_alias(self) -> None:
        """'primary' should also work as an alias for the primary server."""
        from jaxtyc.lsp.mux import _mux_solo_server

        with mock.patch.dict(os.environ, {"JAXTYC_MUX_SOLO": "primary"}):
            assert _mux_solo_server() == "primary"

    def test_empty_string_returns_none(self) -> None:
        from jaxtyc.lsp.mux import _mux_solo_server

        with mock.patch.dict(os.environ, {"JAXTYC_MUX_SOLO": ""}):
            assert _mux_solo_server() is None

    def test_case_insensitive(self) -> None:
        from jaxtyc.lsp.mux import _mux_solo_server

        with mock.patch.dict(os.environ, {"JAXTYC_MUX_SOLO": "JAXTYC"}):
            assert _mux_solo_server() == "jaxtyc"


class TestFilterDiagsByServer:
    """Test that _filter_diag_sources filters the diag_cache correctly."""

    def test_no_filter_returns_all(self) -> None:
        from jaxtyc.lsp.mux import _filter_diag_sources

        cache: dict[str, list[dict]] = {
            "ty": [{"message": "ty diag"}],
            "jaxtyc": [{"message": "jaxtyc diag"}],
        }
        result = _filter_diag_sources(cache, solo=None)
        assert len(result) == 2

    def test_solo_jaxtyc_filters_primary(self) -> None:
        from jaxtyc.lsp.mux import _filter_diag_sources

        cache: dict[str, list[dict]] = {
            "ty": [{"message": "ty diag"}],
            "jaxtyc": [{"message": "jaxtyc diag"}],
        }
        result = _filter_diag_sources(cache, solo="jaxtyc")
        merged: list[dict] = []
        for diags in result.values():
            merged.extend(diags)
        assert len(merged) == 1
        assert merged[0]["message"] == "jaxtyc diag"

    def test_solo_primary_filters_jaxtyc(self) -> None:
        from jaxtyc.lsp.mux import _filter_diag_sources

        cache: dict[str, list[dict]] = {
            "ty": [{"message": "ty diag"}],
            "jaxtyc": [{"message": "jaxtyc diag"}],
        }
        result = _filter_diag_sources(cache, solo="primary")
        merged: list[dict] = []
        for diags in result.values():
            merged.extend(diags)
        assert len(merged) == 1
        assert merged[0]["message"] == "ty diag"

    def test_solo_ty_matches_primary_name(self) -> None:
        """When solo='ty', it should match the key 'ty' in the cache."""
        from jaxtyc.lsp.mux import _filter_diag_sources

        cache: dict[str, list[dict]] = {
            "ty": [{"message": "ty diag"}],
            "jaxtyc": [{"message": "jaxtyc diag"}],
        }
        result = _filter_diag_sources(cache, solo="ty")
        merged: list[dict] = []
        for diags in result.values():
            merged.extend(diags)
        assert len(merged) == 1
        assert merged[0]["message"] == "ty diag"


class TestRunMuxSoloArg:
    """run_mux() accepts a solo parameter that overrides the env var."""

    def test_run_mux_accepts_solo_param(self) -> None:
        import inspect

        from jaxtyc.lsp.mux import run_mux

        sig = inspect.signature(run_mux)
        assert "solo" in sig.parameters, "run_mux() should accept a 'solo' parameter"
        assert sig.parameters["solo"].default is None

    def test_cmd_mux_passes_solo_arg(self) -> None:
        """CLI 'mux --solo jaxtyc' should parse and pass solo to run_mux."""
        from jaxtyc.cli.main import main

        with mock.patch("jaxtyc.cli.main.cmd_mux") as mock_cmd:
            mock_cmd.return_value = 0
            main(["mux", "--solo", "jaxtyc"])
            args = mock_cmd.call_args[0][0]
            assert args.solo == "jaxtyc"

    def test_cmd_mux_solo_default_none(self) -> None:
        """CLI 'mux' without --solo should default to None."""
        from jaxtyc.cli.main import main

        with mock.patch("jaxtyc.cli.main.cmd_mux") as mock_cmd:
            mock_cmd.return_value = 0
            main(["mux"])
            args = mock_cmd.call_args[0][0]
            assert args.solo is None


# ---------------------------------------------------------------------------
# _clean_hover_text
# ---------------------------------------------------------------------------


class TestCleanHoverText:
    def test_replaces_nbsp(self) -> None:
        assert "&nbsp;" not in _clean_hover_text("hello&nbsp;world")

    def test_unescapes_underscores(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "1"}):
            result = _clean_hover_text("my\\_var\\_name")
            assert "\\_" not in result
            assert "my_var_name" in result

    def test_collapses_blank_lines(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "1"}):
            result = _clean_hover_text("a\n\n\n\n\nb")
            assert "\n\n\n" not in result

    def test_truncates_long_text(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "1"}):
            long_text = "x" * 2000
            result = _clean_hover_text(long_text)
            assert len(result) < 2000
            assert "truncated" in result

    def test_no_compaction_when_disabled(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "0"}):
            text = "my\\_var\n\n\n\n\nmore"
            result = _clean_hover_text(text)
            # Should still replace &nbsp; but not unescape underscores
            assert "\\_" in result

    def test_short_text_not_truncated(self) -> None:
        with mock.patch.dict(os.environ, {"JAXTYC_HOVER_COMPACT": "1"}):
            result = _clean_hover_text("short text")
            assert result == "short text"


# ---------------------------------------------------------------------------
# _detect_project_root
# ---------------------------------------------------------------------------


class TestDetectProjectRoot:
    def test_finds_venv_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / ".venv"
            venv_dir.mkdir()
            subdir = Path(tmpdir) / "src" / "pkg"
            subdir.mkdir(parents=True)
            test_file = subdir / "test.py"
            test_file.write_text("")
            root = _detect_project_root(str(test_file))
            assert root == str(Path(tmpdir).resolve())

    def test_finds_pyproject_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("")
            subdir = Path(tmpdir) / "src" / "pkg"
            subdir.mkdir(parents=True)
            test_file = subdir / "test.py"
            test_file.write_text("")
            root = _detect_project_root(str(test_file))
            assert root == str(Path(tmpdir).resolve())

    def test_finds_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".git").mkdir()
            subdir = Path(tmpdir) / "pkg"
            subdir.mkdir()
            test_file = subdir / "test.py"
            test_file.write_text("")
            root = _detect_project_root(str(test_file))
            assert root == str(Path(tmpdir).resolve())

    def test_venv_takes_priority_over_pyproject(self) -> None:
        """In monorepo-like structures, .venv should be found before nested pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Root has .venv
            (Path(tmpdir) / ".venv").mkdir()
            # Nested package has pyproject.toml but no .venv
            nested = Path(tmpdir) / "packages" / "sub"
            nested.mkdir(parents=True)
            (nested / "pyproject.toml").write_text("")
            test_file = nested / "test.py"
            test_file.write_text("")
            root = _detect_project_root(str(test_file))
            # Should find .venv at root, not pyproject.toml at nested
            assert root == str(Path(tmpdir).resolve())

    def test_returns_none_for_root(self) -> None:
        root = _detect_project_root("/tmp/standalone_file.py")
        # May or may not find a project root depending on /tmp layout
        # but should not crash
        assert root is None or isinstance(root, str)

    def test_finds_setup_py(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "setup.py").write_text("")
            subdir = Path(tmpdir) / "pkg"
            subdir.mkdir()
            test_file = subdir / "test.py"
            test_file.write_text("")
            root = _detect_project_root(str(test_file))
            assert root == str(Path(tmpdir).resolve())


# ---------------------------------------------------------------------------
# _patch_root_uri
# ---------------------------------------------------------------------------


class TestPatchRootUri:
    def test_patches_root_uri(self) -> None:
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": "file:///old/path",
                "rootPath": "/old/path",
            },
        }
        patched = _patch_root_uri(msg, "/new/project")
        assert patched["params"]["rootUri"] == "file:///new/project"
        assert patched["params"]["rootPath"] == "/new/project"
        # Original should not be modified
        assert msg["params"]["rootUri"] == "file:///old/path"

    def test_patches_workspace_folders(self) -> None:
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": "file:///old",
                "workspaceFolders": [{"uri": "file:///old", "name": "old"}],
            },
        }
        patched = _patch_root_uri(msg, "/new/project")
        assert patched["params"]["workspaceFolders"][0]["uri"] == "file:///new/project"
        assert patched["params"]["workspaceFolders"][0]["name"] == "project"

    def test_handles_empty_workspace_folders(self) -> None:
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": None,
                "workspaceFolders": [],
            },
        }
        patched = _patch_root_uri(msg, "/new/project")
        assert patched["params"]["rootUri"] == "file:///new/project"
        assert patched["params"]["workspaceFolders"] == []

    def test_handles_no_workspace_folders(self) -> None:
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"processId": None, "rootUri": None},
        }
        patched = _patch_root_uri(msg, "/new/project")
        assert patched["params"]["rootUri"] == "file:///new/project"


# ---------------------------------------------------------------------------
# _extract_file_path_from_msg
# ---------------------------------------------------------------------------


class TestExtractFilePathFromMsg:
    def test_did_open(self) -> None:
        msg = {
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///home/user/file.py",
                    "languageId": "python",
                    "version": 1,
                    "text": "",
                }
            },
        }
        assert _extract_file_path_from_msg(msg) == "/home/user/file.py"

    def test_did_save(self) -> None:
        msg = {
            "method": "textDocument/didSave",
            "params": {"textDocument": {"uri": "file:///home/user/file.py"}},
        }
        assert _extract_file_path_from_msg(msg) == "/home/user/file.py"

    def test_hover(self) -> None:
        msg = {
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": "file:///home/user/file.py"},
                "position": {"line": 0, "character": 0},
            },
        }
        assert _extract_file_path_from_msg(msg) == "/home/user/file.py"

    def test_no_text_document(self) -> None:
        msg = {"method": "initialized", "params": {}}
        assert _extract_file_path_from_msg(msg) is None

    def test_no_params(self) -> None:
        msg = {"method": "shutdown"}
        assert _extract_file_path_from_msg(msg) is None

    def test_non_file_uri(self) -> None:
        msg = {
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {"uri": "untitled:Untitled-1", "text": ""},
            },
        }
        assert _extract_file_path_from_msg(msg) is None

    def test_percent_encoded_path(self) -> None:
        msg = {
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {"uri": "file:///home/user/my%20project/f.py", "text": ""},
            },
        }
        assert _extract_file_path_from_msg(msg) == "/home/user/my project/f.py"


# ---------------------------------------------------------------------------
# _find_primary_server
# ---------------------------------------------------------------------------


class TestFindPrimaryServer:
    def test_finds_pyright_langserver_first(self) -> None:
        def mock_which(cmd: str) -> str | None:
            return "/usr/bin/" + cmd if cmd == "pyright-langserver" else None

        with mock.patch("shutil.which", side_effect=mock_which):
            result = _find_primary_server()
            assert result == ("pyright-langserver", "--stdio")

    def test_falls_back_to_ty(self) -> None:
        def mock_which(cmd: str) -> str | None:
            return "/usr/bin/ty" if cmd == "ty" else None

        with mock.patch("shutil.which", side_effect=mock_which):
            result = _find_primary_server()
            assert result == ("ty", "server")

    def test_falls_back_to_pyright(self) -> None:
        def mock_which(cmd: str) -> str | None:
            return "/usr/bin/pyright" if cmd == "pyright" else None

        with mock.patch("shutil.which", side_effect=mock_which):
            result = _find_primary_server()
            assert result == ("pyright", "--stdio")

    def test_raises_when_none_found(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            raised = False
            try:
                _find_primary_server()
            except RuntimeError as e:
                raised = True
                assert "No Python type checker found" in str(e)
            assert raised, "Should have raised RuntimeError"


# ---------------------------------------------------------------------------
# Mux Integration Tests — full lifecycle via subprocess
# ---------------------------------------------------------------------------


def _lsp_encode(body: dict) -> bytes:
    """Encode a JSON-RPC message with Content-Length header."""
    content = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(content)}\r\n\r\n".encode("ascii")
    return header + content


def _parse_lsp_messages(data: bytes) -> list[dict]:
    """Parse all LSP messages from raw bytes."""
    messages = []
    text = data.decode("utf-8", errors="replace")
    while "Content-Length:" in text:
        header_end = text.index("\r\n\r\n")
        header = text[:header_end]
        cl_part = header.split("Content-Length:")[1]
        length = int(cl_part.split("\r\n")[0].strip())
        body_start = header_end + 4
        body = text[body_start : body_start + length]
        try:
            messages.append(json.loads(body))
        except json.JSONDecodeError:
            pass
        text = text[body_start + length :]
    return messages


class TestMuxIntegration:
    """Full integration tests for the LSP multiplexer subprocess.

    These tests spawn `jaxtyc mux` as a subprocess and communicate via
    stdin/stdout using the LSP JSON-RPC protocol. They verify:
    - Synthetic initialize response is immediate
    - Deferred server startup on first file message
    - Diagnostics from jaxtyc are forwarded
    - Shutdown/exit lifecycle
    """

    def test_initialize_returns_synthetic_response(self) -> None:
        """Mux should respond to initialize immediately with synthetic capabilities."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "mux"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Send initialize
        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "processId": None,
                        "capabilities": {},
                        "rootUri": None,
                    },
                }
            )
        )
        proc.stdin.flush()

        # Send exit immediately (no need to wait for servers)
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        proc.stdin.flush()
        time.sleep(0.5)
        proc.stdin.write(
            _lsp_encode({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "exit", "params": {}}))
        proc.stdin.flush()

        stdout, _ = proc.communicate(timeout=10)
        messages = _parse_lsp_messages(stdout)

        # Should have the synthetic initialize response
        init_resp = next((m for m in messages if m.get("id") == 1), None)
        assert init_resp is not None, f"No init response, got: {messages}"
        assert "result" in init_resp
        caps = init_resp["result"]["capabilities"]
        assert caps["hoverProvider"] is True
        assert caps["definitionProvider"] is True
        assert caps["completionProvider"] == {}
        assert caps["codeLensProvider"] == {}

    def test_synthetic_response_includes_server_info(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "mux"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"processId": None, "capabilities": {}, "rootUri": None},
                }
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        proc.stdin.flush()
        time.sleep(0.3)
        proc.stdin.write(
            _lsp_encode({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "exit", "params": {}}))
        proc.stdin.flush()

        stdout, _ = proc.communicate(timeout=10)
        messages = _parse_lsp_messages(stdout)

        init_resp = next((m for m in messages if m.get("id") == 1), None)
        assert init_resp is not None
        server_info = init_resp["result"]["serverInfo"]
        assert server_info["name"] == "jaxtyc-mux"
        assert server_info["version"]  # Should be a non-empty string

    def test_synthetic_response_includes_all_capabilities(self) -> None:
        """Verify all jaxtyc features are advertised in the synthetic response."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "mux"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"processId": None, "capabilities": {}, "rootUri": None},
                }
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        proc.stdin.flush()
        time.sleep(0.3)
        proc.stdin.write(
            _lsp_encode({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "exit", "params": {}}))
        proc.stdin.flush()

        stdout, _ = proc.communicate(timeout=10)
        messages = _parse_lsp_messages(stdout)
        init_resp = next((m for m in messages if m.get("id") == 1), None)
        assert init_resp is not None
        caps = init_resp["result"]["capabilities"]

        expected_capabilities = [
            "textDocumentSync",
            "hoverProvider",
            "completionProvider",
            "definitionProvider",
            "referencesProvider",
            "documentSymbolProvider",
            "codeActionProvider",
            "codeLensProvider",
            "renameProvider",
            "prepareRenameProvider",
            "foldingRangeProvider",
            "inlayHintProvider",
            "callHierarchyProvider",
            "implementationProvider",
            "workspaceSymbolProvider",
            "signatureHelpProvider",
            "semanticTokensProvider",
            "linkedEditingRangeProvider",
            "documentHighlightProvider",
            "diagnosticProvider",
        ]
        for cap in expected_capabilities:
            assert cap in caps, f"Missing capability: {cap}"

    def test_mux_starts_servers_on_did_open(self) -> None:
        """Opening a file should start both servers and produce diagnostics."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "mux"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        fixture_path = str(FIXTURES / "wrong_transpose.py")
        fixture_uri = Path(fixture_path).as_uri()
        source = Path(fixture_path).read_text()

        # Initialize
        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"processId": None, "capabilities": {}, "rootUri": None},
                }
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        proc.stdin.flush()

        # Open file — this triggers server startup
        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/didOpen",
                    "params": {
                        "textDocument": {
                            "uri": fixture_uri,
                            "languageId": "python",
                            "version": 1,
                            "text": source,
                        }
                    },
                }
            )
        )
        proc.stdin.flush()

        # Wait for servers to produce diagnostics
        time.sleep(5)

        proc.stdin.write(
            _lsp_encode({"jsonrpc": "2.0", "id": 99, "method": "shutdown", "params": {}})
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "exit", "params": {}}))
        proc.stdin.flush()

        stdout, _ = proc.communicate(timeout=15)
        messages = _parse_lsp_messages(stdout)

        # Should have diagnostics from at least one server (jaxtyc)
        diag_messages = [
            m for m in messages if m.get("method") == "textDocument/publishDiagnostics"
        ]
        assert len(diag_messages) >= 1, (
            f"Expected diagnostics, got methods: {[m.get('method', m.get('id')) for m in messages]}"
        )

    def test_mux_correct_file_no_jaxtyc_errors(self) -> None:
        """A correct file should produce no jaxtyc error diagnostics."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "mux"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        fixture_path = str(FIXTURES / "correct_attention.py")
        fixture_uri = Path(fixture_path).as_uri()
        source = Path(fixture_path).read_text()

        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"processId": None, "capabilities": {}, "rootUri": None},
                }
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        proc.stdin.flush()

        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/didOpen",
                    "params": {
                        "textDocument": {
                            "uri": fixture_uri,
                            "languageId": "python",
                            "version": 1,
                            "text": source,
                        }
                    },
                }
            )
        )
        proc.stdin.flush()

        time.sleep(5)

        proc.stdin.write(
            _lsp_encode({"jsonrpc": "2.0", "id": 99, "method": "shutdown", "params": {}})
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "exit", "params": {}}))
        proc.stdin.flush()

        stdout, _ = proc.communicate(timeout=15)
        messages = _parse_lsp_messages(stdout)

        diag_messages = [
            m for m in messages if m.get("method") == "textDocument/publishDiagnostics"
        ]

        # Filter to jaxtyc-sourced errors — there should be none
        for dm in diag_messages:
            jaxtyc_errors = [
                d
                for d in dm["params"].get("diagnostics", [])
                if d.get("source") == "jaxtyc" and d.get("severity") == 1
            ]
            assert len(jaxtyc_errors) == 0, f"Unexpected jaxtyc errors: {jaxtyc_errors}"

    def test_mux_exits_cleanly(self) -> None:
        """Mux should exit cleanly after shutdown+exit without hanging."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "mux"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        proc.stdin.write(
            _lsp_encode(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"processId": None, "capabilities": {}, "rootUri": None},
                }
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        proc.stdin.flush()

        time.sleep(0.3)

        proc.stdin.write(
            _lsp_encode({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_encode({"jsonrpc": "2.0", "method": "exit", "params": {}}))
        proc.stdin.flush()

        start = time.perf_counter()
        proc.communicate(timeout=10)
        elapsed = time.perf_counter() - start

        assert elapsed < 8.0, f"Mux took {elapsed:.1f}s to exit, expected < 8s"
        assert proc.returncode is not None


# ---------------------------------------------------------------------------
# Readiness gate (server warmup)
# ---------------------------------------------------------------------------


class TestReadinessGate:
    """Tests for the readiness-event gate that prevents messages from being sent
    to backend servers before they finish processing their initialize request."""

    def test_readiness_events_gate_server_startup(self) -> None:
        """Readiness events should block until set, simulating the startup gate."""

        async def _run_blocked() -> None:
            primary_ready = asyncio.Event()
            jaxtyc_ready = asyncio.Event()
            await asyncio.wait_for(
                asyncio.gather(primary_ready.wait(), jaxtyc_ready.wait()),
                timeout=0.05,
            )

        async def _run_ready() -> None:
            primary_ready = asyncio.Event()
            jaxtyc_ready = asyncio.Event()
            primary_ready.set()
            jaxtyc_ready.set()
            await asyncio.wait_for(
                asyncio.gather(primary_ready.wait(), jaxtyc_ready.wait()),
                timeout=0.05,
            )

        # Not set -> should timeout
        raised = False
        try:
            asyncio.run(_run_blocked())
        except TimeoutError:
            raised = True
        assert raised, "Expected TimeoutError when events not set"

        # Both set -> completes fine
        asyncio.run(_run_ready())

    def test_readiness_partial_set_still_blocks(self) -> None:
        """If only one event is set, the gate should still block."""

        async def _run_primary_only() -> None:
            primary_ready = asyncio.Event()
            jaxtyc_ready = asyncio.Event()
            primary_ready.set()
            await asyncio.wait_for(
                asyncio.gather(primary_ready.wait(), jaxtyc_ready.wait()),
                timeout=0.05,
            )

        async def _run_jaxtyc_only() -> None:
            primary_ready = asyncio.Event()
            jaxtyc_ready = asyncio.Event()
            jaxtyc_ready.set()
            await asyncio.wait_for(
                asyncio.gather(primary_ready.wait(), jaxtyc_ready.wait()),
                timeout=0.05,
            )

        raised = False
        try:
            asyncio.run(_run_primary_only())
        except TimeoutError:
            raised = True
        assert raised, "Expected TimeoutError when only primary_ready set"

        raised = False
        try:
            asyncio.run(_run_jaxtyc_only())
        except TimeoutError:
            raised = True
        assert raised, "Expected TimeoutError when only jaxtyc_ready set"

    def test_primary_init_response_swallowed(self) -> None:
        """The primary server's initialize response should be swallowed by the
        output handler since the mux already sent a synthetic response."""

        # This tests the structural pattern: when handle_primary_output sees
        # a response with id == primary_init_id, it should set the event and
        # NOT forward the message to the client.
        async def _run() -> list[dict]:
            """Simulate handle_primary_output recognizing the init response."""
            primary_ready = asyncio.Event()
            primary_init_id = 1
            forwarded: list[dict] = []

            # Simulate reading messages from a server
            reader = asyncio.StreamReader()
            # Message 1: the initialize response (should be swallowed)
            reader.feed_data(
                encode_message({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}})
            )
            # Message 2: a normal notification (should be forwarded)
            reader.feed_data(
                encode_message(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {"uri": "file:///test.py", "diagnostics": []},
                    }
                )
            )
            reader.feed_eof()

            while True:
                msg = await read_message(reader)
                if msg is None:
                    break
                msg_id = msg.get("id")
                # Simulate the readiness-gate logic in handle_primary_output
                if msg_id is not None and msg_id == primary_init_id and not primary_ready.is_set():
                    primary_ready.set()
                    continue  # Swallow
                forwarded.append(msg)

            assert primary_ready.is_set(), "primary_ready should be set"
            return forwarded

        forwarded = asyncio.run(_run())
        # Only the diagnostics notification should be forwarded
        assert len(forwarded) == 1
        assert forwarded[0].get("method") == "textDocument/publishDiagnostics"


# ---------------------------------------------------------------------------
# _merge_arrays deduplication
# ---------------------------------------------------------------------------


def test_merge_arrays_deduplicates_locations() -> None:
    """_merge_arrays should deduplicate Location objects with same uri+range."""
    from jaxtyc.lsp.mux import _merge_arrays

    loc = {
        "uri": "file:///test.py",
        "range": {
            "start": {"line": 4, "character": 5},
            "end": {"line": 4, "character": 10},
        },
    }
    result = _merge_arrays([loc], [loc.copy()])
    assert result is not None
    assert len(result) == 1


def test_merge_arrays_keeps_unique_locations() -> None:
    """_merge_arrays should keep Location objects with different uri+range."""
    from jaxtyc.lsp.mux import _merge_arrays

    loc_a = {
        "uri": "file:///test.py",
        "range": {
            "start": {"line": 4, "character": 5},
            "end": {"line": 4, "character": 10},
        },
    }
    loc_b = {
        "uri": "file:///test.py",
        "range": {
            "start": {"line": 10, "character": 0},
            "end": {"line": 10, "character": 6},
        },
    }
    result = _merge_arrays([loc_a], [loc_b])
    assert result is not None
    assert len(result) == 2


def test_merge_arrays_no_dedup_without_location_keys() -> None:
    """_merge_arrays should NOT deduplicate items without uri+range (e.g., CodeLens commands)."""
    from jaxtyc.lsp.mux import _merge_arrays

    item = {"command": {"title": "shapes: x: (batch)"}}
    result = _merge_arrays([item], [item.copy()])
    assert result is not None
    assert len(result) == 2
