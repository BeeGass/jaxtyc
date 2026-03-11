import { describe, it, expect } from "vitest";
import {
  buildCandidates,
  pickServer,
  buildServerCommand,
  buildCheckCommand,
  statusText,
  folderForPath,
  ServerCommand,
  Candidate,
  FolderStatus,
} from "../src/discovery";

// -- buildCandidates ----------------------------------------------------------

describe("buildCandidates", () => {
  const baseCtx = {
    virtualEnv: undefined as string | undefined,
    workspaceFolders: [] as string[],
    pythonExtInterpreter: undefined as string | undefined,
    pathExists: (_p: string) => false,
    readdir: (_p: string) => [] as string[],
    isDirectory: (_p: string) => false,
  };

  it("adds VIRTUAL_ENV when set and bin exists", () => {
    const ctx = {
      ...baseCtx,
      virtualEnv: "/home/user/venvs/myenv",
      pathExists: (p: string) => p === "/home/user/venvs/myenv/bin/python3",
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const venvCandidate = candidates.find(
      (c) => c.command === "/home/user/venvs/myenv/bin/python3"
    );
    expect(venvCandidate).toBeDefined();
    expect(venvCandidate!.testArgs).toEqual(["-c", "import jaxtyc"]);
    expect(venvCandidate!.serverArgs).toEqual([
      "-m",
      "jaxtyc.cli.main",
      "lsp",
    ]);
  });

  it("skips VIRTUAL_ENV when bin does not exist", () => {
    const ctx = {
      ...baseCtx,
      virtualEnv: "/home/user/venvs/missing",
      pathExists: () => false,
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const venvCandidate = candidates.find((c) =>
      c.command.includes("missing")
    );
    expect(venvCandidate).toBeUndefined();
  });

  it("adds .venv from workspace folder root", () => {
    const ctx = {
      ...baseCtx,
      workspaceFolders: ["/home/user/project"],
      pathExists: (p: string) =>
        p === "/home/user/project/.venv/bin/python3",
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const wsCandidate = candidates.find(
      (c) => c.command === "/home/user/project/.venv/bin/python3"
    );
    expect(wsCandidate).toBeDefined();
    expect(wsCandidate!.serverArgs).toEqual([
      "-m",
      "jaxtyc.cli.main",
      "lsp",
    ]);
  });

  it("finds .venv in subdir with pyproject.toml (worktree layout)", () => {
    const existing = new Set([
      "/home/user/parent/child/pyproject.toml",
      "/home/user/parent/child/.venv/bin/python3",
    ]);
    const ctx = {
      ...baseCtx,
      workspaceFolders: ["/home/user/parent"],
      pathExists: (p: string) => existing.has(p),
      readdir: (p: string) =>
        p === "/home/user/parent" ? ["child", "other"] : [],
      isDirectory: (p: string) =>
        p === "/home/user/parent/child" || p === "/home/user/parent/other",
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const subCandidate = candidates.find(
      (c) => c.command === "/home/user/parent/child/.venv/bin/python3"
    );
    expect(subCandidate).toBeDefined();
  });

  it("skips subdir without pyproject.toml", () => {
    const existing = new Set([
      "/home/user/parent/child/.venv/bin/python3",
      // no pyproject.toml
    ]);
    const ctx = {
      ...baseCtx,
      workspaceFolders: ["/home/user/parent"],
      pathExists: (p: string) => existing.has(p),
      readdir: (p: string) =>
        p === "/home/user/parent" ? ["child"] : [],
      isDirectory: (p: string) => p === "/home/user/parent/child",
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const subCandidate = candidates.find(
      (c) => c.command === "/home/user/parent/child/.venv/bin/python3"
    );
    expect(subCandidate).toBeUndefined();
  });

  it("always includes jaxtyc executable candidate", () => {
    const candidates = buildCandidates(baseCtx, "mux", []);
    const jaxtycCandidate = candidates.find((c) => c.command === "jaxtyc");
    expect(jaxtycCandidate).toBeDefined();
    expect(jaxtycCandidate!.testArgs).toEqual(["version"]);
    expect(jaxtycCandidate!.serverArgs).toEqual(["mux"]);
  });

  it("adds VS Code Python extension interpreter when set", () => {
    const ctx = {
      ...baseCtx,
      pythonExtInterpreter: "/usr/bin/python3.11",
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const pyCandidate = candidates.find(
      (c) => c.command === "/usr/bin/python3.11"
    );
    expect(pyCandidate).toBeDefined();
  });

  it("skips Python extension interpreter when set to 'python'", () => {
    const ctx = {
      ...baseCtx,
      pythonExtInterpreter: "python",
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    // Should not have a candidate with command "python" from the Python ext
    // (python3 fallback is separate)
    const pyCandidate = candidates.find(
      (c) => c.command === "python" && c.testArgs[0] === "-c"
    );
    expect(pyCandidate).toBeUndefined();
  });

  it("always includes python3 fallback", () => {
    const candidates = buildCandidates(baseCtx, "lsp", []);
    const py3 = candidates.find((c) => c.command === "python3");
    expect(py3).toBeDefined();
    expect(py3!.testArgs).toEqual(["-c", "import jaxtyc"]);
  });

  it("passes extra args through to serverArgs", () => {
    const candidates = buildCandidates(baseCtx, "lsp", ["--verbose"]);
    const py3 = candidates.find((c) => c.command === "python3");
    expect(py3!.serverArgs).toEqual([
      "-m",
      "jaxtyc.cli.main",
      "lsp",
      "--verbose",
    ]);
  });

  it("jaxtyc executable passes extra args without -m prefix", () => {
    const candidates = buildCandidates(baseCtx, "mux", ["--debug"]);
    const jaxtyc = candidates.find((c) => c.command === "jaxtyc");
    expect(jaxtyc!.serverArgs).toEqual(["mux", "--debug"]);
  });

  it("scans multiple workspace folders", () => {
    const existing = new Set([
      "/home/user/projectA/.venv/bin/python3",
      "/home/user/projectB/.venv/bin/python3",
    ]);
    const ctx = {
      ...baseCtx,
      workspaceFolders: ["/home/user/projectA", "/home/user/projectB"],
      pathExists: (p: string) => existing.has(p),
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const venvCandidates = candidates.filter((c) =>
      c.command.includes(".venv")
    );
    expect(venvCandidates).toHaveLength(2);
  });

  it("handles readdir errors gracefully", () => {
    const ctx = {
      ...baseCtx,
      workspaceFolders: ["/home/user/project"],
      pathExists: () => false,
      readdir: () => {
        throw new Error("EACCES");
      },
    };
    // Should not throw
    const candidates = buildCandidates(ctx, "lsp", []);
    expect(candidates.length).toBeGreaterThan(0);
  });
});

// -- pickServer ---------------------------------------------------------------

describe("pickServer", () => {
  it("returns the first candidate that passes canRun", () => {
    const candidates: Candidate[] = [
      {
        command: "/bad/python3",
        testArgs: ["-c", "import jaxtyc"],
        serverArgs: ["-m", "jaxtyc.cli.main", "lsp"],
      },
      {
        command: "/good/python3",
        testArgs: ["-c", "import jaxtyc"],
        serverArgs: ["-m", "jaxtyc.cli.main", "lsp"],
      },
    ];
    const canRun = (cmd: string, _args: string[]) =>
      cmd === "/good/python3";
    const result = pickServer(candidates, canRun);
    expect(result).toEqual({
      command: "/good/python3",
      args: ["-m", "jaxtyc.cli.main", "lsp"],
    });
  });

  it("returns undefined when no candidate passes", () => {
    const candidates: Candidate[] = [
      {
        command: "/bad/python3",
        testArgs: ["-c", "import jaxtyc"],
        serverArgs: ["-m", "jaxtyc.cli.main", "lsp"],
      },
    ];
    const canRun = () => false;
    expect(pickServer(candidates, canRun)).toBeUndefined();
  });

  it("returns undefined for empty candidates list", () => {
    expect(pickServer([], () => true)).toBeUndefined();
  });

  it("stops at the first passing candidate", () => {
    let callCount = 0;
    const candidates: Candidate[] = [
      {
        command: "first",
        testArgs: [],
        serverArgs: ["lsp"],
      },
      {
        command: "second",
        testArgs: [],
        serverArgs: ["lsp"],
      },
    ];
    const canRun = () => {
      callCount++;
      return true;
    };
    pickServer(candidates, canRun);
    expect(callCount).toBe(1);
  });
});

// -- buildServerCommand -------------------------------------------------------

describe("buildServerCommand", () => {
  it("uses explicit pythonPath when provided", () => {
    const cmd = buildServerCommand({
      pythonPath: "/usr/bin/python3.11",
      mode: "lsp",
      extraArgs: [],
    });
    expect(cmd).toEqual({
      command: "/usr/bin/python3.11",
      args: ["-m", "jaxtyc.cli.main", "lsp"],
    });
  });

  it("passes extra args with explicit pythonPath", () => {
    const cmd = buildServerCommand({
      pythonPath: "/usr/bin/python3.11",
      mode: "mux",
      extraArgs: ["--verbose"],
    });
    expect(cmd).toEqual({
      command: "/usr/bin/python3.11",
      args: ["-m", "jaxtyc.cli.main", "mux", "--verbose"],
    });
  });

  it("returns discovered server when no explicit path", () => {
    const cmd = buildServerCommand({
      pythonPath: "",
      mode: "lsp",
      extraArgs: [],
      discovered: { command: "jaxtyc", args: ["lsp"] },
    });
    expect(cmd).toEqual({ command: "jaxtyc", args: ["lsp"] });
  });

  it("returns undefined when no explicit path and no discovery", () => {
    const cmd = buildServerCommand({
      pythonPath: "",
      mode: "lsp",
      extraArgs: [],
      discovered: undefined,
    });
    expect(cmd).toBeUndefined();
  });
});

// -- buildCheckCommand --------------------------------------------------------

describe("buildCheckCommand", () => {
  it("builds check command with explicit pythonPath", () => {
    const cmd = buildCheckCommand({
      pythonPath: "/usr/bin/python3",
      filePath: "/home/user/test.py",
      extraArgs: [],
    });
    expect(cmd).toEqual({
      command: "/usr/bin/python3",
      args: ["-m", "jaxtyc.cli.main", "check", "/home/user/test.py"],
    });
  });

  it("builds check command for discovered Python interpreter", () => {
    const cmd = buildCheckCommand({
      pythonPath: "",
      filePath: "/home/user/test.py",
      extraArgs: [],
      discovered: {
        command: "/home/user/.venv/bin/python3",
        args: ["-m", "jaxtyc.cli.main", "lsp"],
      },
    });
    expect(cmd).toEqual({
      command: "/home/user/.venv/bin/python3",
      args: ["-m", "jaxtyc.cli.main", "check", "/home/user/test.py"],
    });
  });

  it("builds check command for discovered jaxtyc executable", () => {
    const cmd = buildCheckCommand({
      pythonPath: "",
      filePath: "/home/user/test.py",
      extraArgs: [],
      discovered: { command: "jaxtyc", args: ["lsp"] },
    });
    expect(cmd).toEqual({
      command: "jaxtyc",
      args: ["check", "/home/user/test.py"],
    });
  });

  it("passes extra args to check command", () => {
    const cmd = buildCheckCommand({
      pythonPath: "/usr/bin/python3",
      filePath: "/home/user/test.py",
      extraArgs: ["--format", "json"],
    });
    expect(cmd).toEqual({
      command: "/usr/bin/python3",
      args: [
        "-m",
        "jaxtyc.cli.main",
        "check",
        "/home/user/test.py",
        "--format",
        "json",
      ],
    });
  });

  it("returns undefined when no pythonPath and no discovery", () => {
    const cmd = buildCheckCommand({
      pythonPath: "",
      filePath: "/home/user/test.py",
      extraArgs: [],
      discovered: undefined,
    });
    expect(cmd).toBeUndefined();
  });
});

// -- candidate ordering -------------------------------------------------------

describe("candidate ordering", () => {
  it("prefers VIRTUAL_ENV over workspace .venv", () => {
    const existing = new Set([
      "/activated/venv/bin/python3",
      "/home/user/project/.venv/bin/python3",
    ]);
    const ctx = {
      virtualEnv: "/activated/venv",
      workspaceFolders: ["/home/user/project"],
      pythonExtInterpreter: undefined,
      pathExists: (p: string) => existing.has(p),
      readdir: () => [] as string[],
      isDirectory: () => false,
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const venvIdx = candidates.findIndex(
      (c) => c.command === "/activated/venv/bin/python3"
    );
    const wsIdx = candidates.findIndex(
      (c) => c.command === "/home/user/project/.venv/bin/python3"
    );
    expect(venvIdx).toBeLessThan(wsIdx);
  });

  it("prefers workspace .venv over jaxtyc on PATH", () => {
    const ctx = {
      virtualEnv: undefined,
      workspaceFolders: ["/home/user/project"],
      pythonExtInterpreter: undefined,
      pathExists: (p: string) =>
        p === "/home/user/project/.venv/bin/python3",
      readdir: () => [] as string[],
      isDirectory: () => false,
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const wsIdx = candidates.findIndex(
      (c) => c.command === "/home/user/project/.venv/bin/python3"
    );
    const jaxtycIdx = candidates.findIndex((c) => c.command === "jaxtyc");
    expect(wsIdx).toBeLessThan(jaxtycIdx);
  });

  it("prefers jaxtyc on PATH over python3 fallback", () => {
    const candidates = buildCandidates(
      {
        virtualEnv: undefined,
        workspaceFolders: [],
        pythonExtInterpreter: undefined,
        pathExists: () => false,
        readdir: () => [] as string[],
        isDirectory: () => false,
      },
      "lsp",
      []
    );
    const jaxtycIdx = candidates.findIndex((c) => c.command === "jaxtyc");
    const py3Idx = candidates.findIndex((c) => c.command === "python3");
    expect(jaxtycIdx).toBeLessThan(py3Idx);
  });

  it("prefers workspace subdir .venv over jaxtyc on PATH", () => {
    const existing = new Set([
      "/home/user/parent/child/pyproject.toml",
      "/home/user/parent/child/.venv/bin/python3",
    ]);
    const ctx = {
      virtualEnv: undefined,
      workspaceFolders: ["/home/user/parent"],
      pythonExtInterpreter: undefined,
      pathExists: (p: string) => existing.has(p),
      readdir: (p: string) =>
        p === "/home/user/parent" ? ["child"] : [],
      isDirectory: (p: string) => p === "/home/user/parent/child",
    };
    const candidates = buildCandidates(ctx, "lsp", []);
    const subIdx = candidates.findIndex(
      (c) => c.command === "/home/user/parent/child/.venv/bin/python3"
    );
    const jaxtycIdx = candidates.findIndex((c) => c.command === "jaxtyc");
    expect(subIdx).toBeLessThan(jaxtycIdx);
  });
});

// -- statusText ---------------------------------------------------------------

describe("statusText", () => {
  it("shows mode for single running folder", () => {
    const m = new Map<string, FolderStatus>([["a", "running"]]);
    expect(statusText("lsp", m)).toBe("jaxtyc [lsp]");
  });

  it("shows folder count for multiple running folders", () => {
    const m = new Map<string, FolderStatus>([["a", "running"], ["b", "running"]]);
    expect(statusText("mux", m)).toBe("jaxtyc [mux] (2 folders)");
  });

  it("shows partial count when some fail", () => {
    const m = new Map<string, FolderStatus>([
      ["a", "running"], ["b", "error"], ["c", "not-found"],
    ]);
    expect(statusText("lsp", m)).toBe("jaxtyc [lsp] (1/3)");
  });

  it("shows not-found when none running", () => {
    const m = new Map<string, FolderStatus>([["a", "error"]]);
    expect(statusText("lsp", m)).toBe("jaxtyc (not found)");
  });

  it("handles empty map", () => {
    expect(statusText("lsp", new Map())).toBe("jaxtyc (no folders)");
  });
});

// -- folderForPath ------------------------------------------------------------

describe("folderForPath", () => {
  it("matches file to containing folder", () => {
    expect(folderForPath("/a/b/c.py", ["/a/b", "/d"])).toBe("/a/b");
  });

  it("prefers longest match (most specific folder)", () => {
    expect(folderForPath("/a/b/c/d.py", ["/a", "/a/b/c"])).toBe("/a/b/c");
  });

  it("returns undefined for no match", () => {
    expect(folderForPath("/x/y.py", ["/a", "/b"])).toBeUndefined();
  });

  it("does not match partial directory names", () => {
    expect(folderForPath("/abc/d.py", ["/ab"])).toBeUndefined();
  });
});
