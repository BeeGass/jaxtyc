"""jaxtyc CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _enforce_cpu_backend() -> None:
    """Force JAX to use CPU backend unless overridden.

    jaxtyc uses only jax.eval_shape and jax.make_jaxpr, which work
    identically on CPU. Setting JAX_PLATFORMS=cpu prevents GPU VRAM
    pre-allocation that wastes memory in a static analysis tool.

    Override with JAXTYC_BACKEND=gpu or by setting JAX_PLATFORMS directly.
    """
    backend = os.environ.get("JAXTYC_BACKEND", "cpu").strip().lower()
    if backend != "gpu" and "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"


_enforce_cpu_backend()

import jaxtyc  # noqa: E402
from jaxtyc.analyzer._errors import truncate_error  # noqa: E402
from jaxtyc.analyzer.pipeline import analyze_file  # noqa: E402
from jaxtyc.cli.formatters import FORMATTERS  # noqa: E402
from jaxtyc.config import filter_diagnostics  # noqa: E402
from jaxtyc.config import load_config  # noqa: E402
from jaxtyc.types import FileResult  # noqa: E402


def _apply_exclude(files: list[str], patterns: list[str]) -> list[str]:
    """Filter out files matching any of the exclude glob patterns."""
    import fnmatch

    result = []
    for f in files:
        if not any(fnmatch.fnmatch(f, pat) for pat in patterns):
            result.append(f)
    return result


def _collect_python_files(paths: list[str]) -> list[str]:
    """Expand paths into a list of .py files (recurse directories)."""
    files: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".py":
            files.append(str(path))
        elif path.is_dir():
            files.extend(str(f) for f in sorted(path.rglob("*.py")))
        else:
            files.append(str(path))  # Let analyze_file handle the error
    return files


def cmd_check(args: argparse.Namespace) -> int:
    """Run shape checks on files/directories."""
    config = load_config(Path.cwd())

    files = _collect_python_files(args.paths)
    if config.exclude:
        files = _apply_exclude(files, config.exclude)
    if not files:
        print("No Python files found.", file=sys.stderr)
        return 0

    formatter = FORMATTERS[args.format]
    start = time.monotonic()

    results: list[FileResult] = []
    for f in files:
        result = analyze_file(f)
        filtered = filter_diagnostics(result.diagnostics, config)
        results.append(
            FileResult(
                file_path=result.file_path,
                functions_checked=result.functions_checked,
                diagnostics=filtered,
                trace_results=result.trace_results,
            )
        )

    elapsed = time.monotonic() - start
    output = formatter(results, elapsed)
    if output.strip():
        print(output)

    import jax

    jax.clear_caches()

    # Exit nonzero if any errors
    has_errors = any(d.severity == "error" for r in results for d in r.diagnostics)
    return 1 if has_errors else 0


def cmd_trace(args: argparse.Namespace) -> int:
    """Trace intermediate shapes through a function."""
    # Parse file::function syntax
    if "::" not in args.target:
        print(f"Error: expected file.py::function_name, got: {args.target}", file=sys.stderr)
        return 1

    file_path, func_name = args.target.rsplit("::", 1)

    from jaxtyc.analyzer.annotations import extract_function_specs
    from jaxtyc.analyzer.dim_env import DimEnv
    from jaxtyc.analyzer.importer import import_module_from_path
    from jaxtyc.analyzer.source_map import extract_source_mapped_intermediates
    from jaxtyc.analyzer.tracer import _build_abstract_input
    from jaxtyc.analyzer.tracer import trace_function

    # Read and parse
    path = Path(file_path)
    if not path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    source = path.read_text(encoding="utf-8")
    func_specs = extract_function_specs(source, file_path)

    # Find the target function
    target_spec = None
    for spec in func_specs:
        if spec.name == func_name:
            target_spec = spec
            break

    if target_spec is None:
        print(
            f"Error: no jaxtyping-annotated function `{func_name}` in {file_path}", file=sys.stderr
        )
        return 1

    # Import and trace
    try:
        module = import_module_from_path(file_path)
    except Exception as e:
        print(f"Error: could not import {file_path}: {truncate_error(e)}", file=sys.stderr)
        return 1

    fn = getattr(module, func_name, None)
    if target_spec.class_name:
        cls = getattr(module, target_spec.class_name, None)
        if cls:
            fn = getattr(cls, func_name, None)

    if fn is None:
        print(f"Error: could not resolve function `{func_name}`", file=sys.stderr)
        return 1

    env = DimEnv()
    trace = trace_function(fn, target_spec.params, env)

    # Print header
    param_strs = []
    for pname, pspec in target_spec.params.items():
        dim_names = ", ".join(d.name or str(d.size) or d.kind for d in pspec.dims)
        param_strs.append(f"{pname}: {pspec.dtype}[{dim_names}]")

    ret_str = ""
    if target_spec.return_spec:
        ret_dims = ", ".join(d.name or str(d.size) or d.kind for d in target_spec.return_spec.dims)
        ret_str = f" -> {target_spec.return_spec.dtype}[{ret_dims}]"

    print(f"{func_name}({', '.join(param_strs)}){ret_str}")
    print()

    if not trace.success:
        print(f"  Trace error: {trace.error}")
        return 1

    # Print intermediates

    abstract_inputs = {}
    for pname, pspec in target_spec.params.items():
        if not pspec.is_any_shape:
            abstract_inputs[pname] = _build_abstract_input(pspec, env)

    intermediates = extract_source_mapped_intermediates(fn, abstract_inputs, env)

    for inter in intermediates:
        named = ", ".join(n or str(s) for n, s in zip(inter.named_shape, inter.shape, strict=True))
        loc = ""
        if inter.source_line > 0:
            loc = f"Line {inter.source_line}: "
        print(f"  {loc}{inter.op_name} -> ({named})  [{inter.dtype}]")

    # Print output
    if trace.output_shape is not None:
        out_named = ", ".join(
            n or str(s)
            for n, s in zip(env.shape_to_names(trace.output_shape), trace.output_shape, strict=True)
        )
        match_str = ""
        if target_spec.return_spec and not target_spec.return_spec.is_any_shape:
            expected = env.make_shape(target_spec.return_spec)
            match_str = " [matches]" if expected == trace.output_shape else " [MISMATCH]"
        print(f"\n  Output: ({out_named}){match_str}")

    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Watch files/directories and re-check on change."""
    try:
        from watchfiles import watch
    except ImportError:
        print(
            "Error: watchfiles not installed. Install with: pip install jaxtyc[watch]",
            file=sys.stderr,
        )
        return 1

    files = _collect_python_files(args.paths)
    if not files:
        print("No Python files found.", file=sys.stderr)
        return 0

    # Determine directories to watch
    watch_dirs: set[str] = set()
    for f in files:
        watch_dirs.add(str(Path(f).parent))
    # Also watch any directories passed directly
    for p in args.paths:
        if Path(p).is_dir():
            watch_dirs.add(str(Path(p).resolve()))

    formatter = FORMATTERS[args.format]

    print(f"Watching {len(watch_dirs)} directory(ies) for changes...")
    sys.stdout.flush()

    # Run initial check
    start = time.monotonic()
    results: list[FileResult] = []
    for f in files:
        results.append(analyze_file(f))
    elapsed = time.monotonic() - start
    output = formatter(results, elapsed)
    if output.strip():
        print(output)
        sys.stdout.flush()

    # Watch for changes
    for changes in watch(*watch_dirs):
        changed_py = [path for _, path in changes if path.endswith(".py")]
        if not changed_py:
            continue

        start = time.monotonic()
        results = []
        for f in changed_py:
            results.append(analyze_file(f))
        elapsed = time.monotonic() - start
        output = formatter(results, elapsed)
        if output.strip():
            print(output)
            sys.stdout.flush()

        import jax

        jax.clear_caches()

    return 0


def cmd_lsp(args: argparse.Namespace) -> int:
    """Start the LSP server."""
    from jaxtyc.lsp.server import start_lsp

    start_lsp()
    return 0


def cmd_mux(args: argparse.Namespace) -> int:
    """Start the LSP multiplexer (ty + jaxtyc)."""
    import asyncio

    from jaxtyc.lsp.mux import run_mux

    asyncio.run(run_mux(solo=args.solo))
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print version."""
    print(f"jaxtyc {jaxtyc.__version__}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="jaxtyc",
        description="Static array shape checking for JAX",
    )
    subparsers = parser.add_subparsers(dest="command")

    # check
    check_parser = subparsers.add_parser("check", help="Check files for shape errors")
    check_parser.add_argument("paths", nargs="+", help="Files or directories to check")
    check_parser.add_argument(
        "--format",
        choices=["full", "concise", "json", "github"],
        default="full",
        help="Output format (default: full)",
    )
    check_parser.set_defaults(func=cmd_check)

    # trace
    trace_parser = subparsers.add_parser("trace", help="Trace intermediate shapes")
    trace_parser.add_argument("target", help="file.py::function_name")
    trace_parser.set_defaults(func=cmd_trace)

    # watch
    watch_parser = subparsers.add_parser("watch", help="Watch and re-check on file change")
    watch_parser.add_argument("paths", nargs="+", help="Files or directories to watch")
    watch_parser.add_argument(
        "--format",
        choices=["full", "concise", "json", "github"],
        default="full",
        help="Output format (default: full)",
    )
    watch_parser.set_defaults(func=cmd_watch)

    # lsp
    lsp_parser = subparsers.add_parser("lsp", help="Start LSP server (stdio)")
    lsp_parser.set_defaults(func=cmd_lsp)

    # mux
    mux_parser = subparsers.add_parser("mux", help="Start LSP multiplexer (ty + jaxtyc)")
    mux_parser.add_argument(
        "--solo",
        choices=["jaxtyc", "ty", "primary", "pyright"],
        default=None,
        help="Show diagnostics from only this server (default: both)",
    )
    mux_parser.set_defaults(func=cmd_mux)

    # version
    version_parser = subparsers.add_parser("version", help="Print version")
    version_parser.set_defaults(func=cmd_version)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
