import { join, sep } from "path";
import { platform } from "os";

function venvPython(venvDir: string): string {
  return platform() === "win32"
    ? join(venvDir, "Scripts", "python.exe")
    : join(venvDir, "bin", "python3");
}

export type FolderStatus = "running" | "error" | "not-found";

export function statusText(
  mode: string,
  folders: Map<string, FolderStatus>
): string {
  if (folders.size === 0) return "jaxtyc (no folders)";
  const running = [...folders.values()].filter((s) => s === "running").length;
  const total = folders.size;
  if (running === total) {
    return total === 1
      ? `jaxtyc [${mode}]`
      : `jaxtyc [${mode}] (${total} folders)`;
  }
  if (running === 0) return "jaxtyc (not found)";
  return `jaxtyc [${mode}] (${running}/${total})`;
}

export function folderForPath(
  filePath: string,
  folderPaths: string[]
): string | undefined {
  let best: string | undefined;
  let bestLen = 0;
  for (const fp of folderPaths) {
    if (filePath.startsWith(fp + sep) && fp.length > bestLen) {
      best = fp;
      bestLen = fp.length;
    }
  }
  return best;
}

export interface ServerCommand {
  command: string;
  args: string[];
}

export interface Candidate {
  command: string;
  testArgs: string[];
  serverArgs: string[];
}

export interface DiscoveryContext {
  virtualEnv: string | undefined;
  workspaceFolders: string[];
  pythonExtInterpreter: string | undefined;
  pathExists: (path: string) => boolean;
  readdir: (path: string) => string[];
  isDirectory: (path: string) => boolean;
}

export function buildCandidates(
  ctx: DiscoveryContext,
  mode: string,
  extraArgs: string[]
): Candidate[] {
  const candidates: Candidate[] = [];

  const pyServerArgs = (m: string, extra: string[]) => [
    "-m",
    "jaxtyc.cli.main",
    m,
    ...extra,
  ];

  // 1. VIRTUAL_ENV env var
  if (ctx.virtualEnv) {
    const bin = venvPython(ctx.virtualEnv);
    if (ctx.pathExists(bin)) {
      candidates.push({
        command: bin,
        testArgs: ["-c", "import jaxtyc"],
        serverArgs: pyServerArgs(mode, extraArgs),
      });
    }
  }

  // 2. .venv in workspace folders (direct + one-level subdir with pyproject.toml)
  for (const folder of ctx.workspaceFolders) {
    const venvBin = venvPython(join(folder, ".venv"));
    if (ctx.pathExists(venvBin)) {
      candidates.push({
        command: venvBin,
        testArgs: ["-c", "import jaxtyc"],
        serverArgs: pyServerArgs(mode, extraArgs),
      });
    }
    try {
      for (const entry of ctx.readdir(folder)) {
        const sub = join(folder, entry);
        if (
          ctx.isDirectory(sub) &&
          ctx.pathExists(join(sub, "pyproject.toml")) &&
          ctx.pathExists(venvPython(join(sub, ".venv")))
        ) {
          candidates.push({
            command: venvPython(join(sub, ".venv")),
            testArgs: ["-c", "import jaxtyc"],
            serverArgs: pyServerArgs(mode, extraArgs),
          });
        }
      }
    } catch {
      // Permission errors or unreadable dirs
    }
  }

  // 3. jaxtyc executable on PATH
  candidates.push({
    command: "jaxtyc",
    testArgs: ["version"],
    serverArgs: [mode, ...extraArgs],
  });

  // 4. VS Code Python extension's interpreter
  if (ctx.pythonExtInterpreter && ctx.pythonExtInterpreter !== "python") {
    candidates.push({
      command: ctx.pythonExtInterpreter,
      testArgs: ["-c", "import jaxtyc"],
      serverArgs: pyServerArgs(mode, extraArgs),
    });
  }

  // 5. python3 on PATH
  candidates.push({
    command: "python3",
    testArgs: ["-c", "import jaxtyc"],
    serverArgs: pyServerArgs(mode, extraArgs),
  });

  return candidates;
}

export function pickServer(
  candidates: Candidate[],
  canRun: (command: string, args: string[]) => boolean
): ServerCommand | undefined {
  for (const candidate of candidates) {
    if (canRun(candidate.command, candidate.testArgs)) {
      return { command: candidate.command, args: candidate.serverArgs };
    }
  }
  return undefined;
}

export function buildServerCommand(opts: {
  pythonPath: string;
  mode: string;
  extraArgs: string[];
  discovered?: ServerCommand;
}): ServerCommand | undefined {
  if (opts.pythonPath) {
    return {
      command: opts.pythonPath,
      args: ["-m", "jaxtyc.cli.main", opts.mode, ...opts.extraArgs],
    };
  }
  return opts.discovered;
}

export function buildCheckCommand(opts: {
  pythonPath: string;
  filePath: string;
  extraArgs: string[];
  discovered?: ServerCommand;
}): ServerCommand | undefined {
  if (opts.pythonPath) {
    return {
      command: opts.pythonPath,
      args: [
        "-m",
        "jaxtyc.cli.main",
        "check",
        opts.filePath,
        ...opts.extraArgs,
      ],
    };
  }
  if (!opts.discovered) {
    return undefined;
  }
  const isPython = opts.discovered.args[0] === "-m";
  return {
    command: opts.discovered.command,
    args: isPython
      ? [
          "-m",
          "jaxtyc.cli.main",
          "check",
          opts.filePath,
          ...opts.extraArgs,
        ]
      : ["check", opts.filePath, ...opts.extraArgs],
  };
}
