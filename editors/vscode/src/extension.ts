import { execFile, execFileSync } from "child_process";
import { existsSync, readdirSync, statSync } from "fs";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";
import {
  buildCandidates,
  buildCheckCommand,
  buildServerCommand,
  FolderStatus,
  folderForPath,
  pickServer,
  ServerCommand,
  statusText,
} from "./discovery";
import {
  dispose as disposeDecorations,
  handleInlayHints,
  initDecorationTypes,
  onDocumentClose,
  onEditorChange,
} from "./shape-decorations";
import { parseTraceOutput, renderTraceHtml } from "./trace-panel";

const clients = new Map<string, LanguageClient>();
const folderStatuses = new Map<string, FolderStatus>();

let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

function canRun(command: string, args: string[]): boolean {
  try {
    execFileSync(command, args, { timeout: 5000, stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function discoverServerForFolder(
  folderPath: string,
  mode: string,
  extraArgs: string[]
): ServerCommand | undefined {
  const pyConfig = vscode.workspace.getConfiguration("python");
  const candidates = buildCandidates(
    {
      virtualEnv: process.env["VIRTUAL_ENV"],
      workspaceFolders: [folderPath],
      pythonExtInterpreter: pyConfig.get<string>("defaultInterpreterPath"),
      pathExists: existsSync,
      readdir: readdirSync,
      isDirectory: (p) => {
        try {
          return statSync(p).isDirectory();
        } catch {
          return false;
        }
      },
    },
    mode,
    extraArgs
  );
  return pickServer(candidates, canRun);
}

function getConfig(): { mode: string; pythonPath: string; args: string[] } {
  const cfg = vscode.workspace.getConfiguration("jaxtyc");
  return {
    mode: cfg.get<string>("mode", "lsp"),
    pythonPath: cfg.get<string>("pythonPath", ""),
    args: cfg.get<string[]>("args", []),
  };
}

async function startClientForFolder(
  folder: vscode.WorkspaceFolder
): Promise<void> {
  const folderUri = folder.uri.toString();
  const { mode, pythonPath, args } = getConfig();

  const server = buildServerCommand({
    pythonPath,
    mode,
    extraArgs: args,
    discovered: discoverServerForFolder(folder.uri.fsPath, mode, args),
  });

  if (!server) {
    outputChannel.appendLine(`[${folder.name}] jaxtyc not found`);
    folderStatuses.set(folderUri, "not-found");
    updateStatusBar();
    return;
  }

  outputChannel.appendLine(
    `[${folder.name}] Server: ${server.command} ${server.args.join(" ")}`
  );

  const serverOptions: ServerOptions = {
    command: server.command,
    args: server.args,
    options: { cwd: folder.uri.fsPath },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [
      {
        scheme: "file",
        language: "python",
        pattern: `${folder.uri.fsPath}/**`,
      },
    ],
    outputChannel,
    workspaceFolder: folder,
    middleware: {
      provideInlayHints: async (document, range, token, next) => {
        const hints = await next(document, range, token);
        if (hints && hints.length > 0) {
          // Convert to colored decorations, suppress default rendering
          return handleInlayHints(document, hints as any);
        }
        return hints;
      },
    },
  };

  const client = new LanguageClient(
    `jaxtyc-${folder.name}`,
    `jaxtyc (${folder.name})`,
    serverOptions,
    clientOptions
  );

  try {
    await client.start();
    clients.set(folderUri, client);
    folderStatuses.set(folderUri, "running");
    const serverVersion = client.initializeResult?.serverInfo?.version;
    if (serverVersion) {
      statusBarItem.tooltip = `jaxtyc ${serverVersion} — shape checker`;
    }
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    outputChannel.appendLine(`[${folder.name}] Failed: ${detail}`);
    folderStatuses.set(folderUri, "error");
  }
  updateStatusBar();
}

async function stopClientForFolder(folderUri: string): Promise<void> {
  const client = clients.get(folderUri);
  if (client) {
    try {
      await client.stop();
    } catch {
      /* already stopped */
    }
    clients.delete(folderUri);
    folderStatuses.delete(folderUri);
  }
}

async function stopAllClients(): Promise<void> {
  await Promise.all([...clients.keys()].map(stopClientForFolder));
}

function updateStatusBar(): void {
  const { mode } = getConfig();
  statusBarItem.text = statusText(mode, folderStatuses);
  const hasError = [...folderStatuses.values()].some(
    (s) => s === "error" || s === "not-found"
  );
  statusBarItem.backgroundColor = hasError
    ? new vscode.ThemeColor("statusBarItem.warningBackground")
    : undefined;
  statusBarItem.show();
}

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  outputChannel = vscode.window.createOutputChannel("jaxtyc");
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left
  );
  statusBarItem.tooltip = "jaxtyc shape checker";
  statusBarItem.command = "jaxtyc.showMenu";

  initDecorationTypes();

  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    await startClientForFolder(folder);
  }

  context.subscriptions.push(
    vscode.workspace.onDidChangeWorkspaceFolders(async (e) => {
      for (const removed of e.removed) {
        await stopClientForFolder(removed.uri.toString());
      }
      for (const added of e.added) {
        await startClientForFolder(added);
      }
      updateStatusBar();
    })
  );

  // Reinitialize decoration colors when the user switches themes
  context.subscriptions.push(
    vscode.window.onDidChangeActiveColorTheme(() => {
      initDecorationTypes();
    })
  );

  // Re-apply decorations when the active editor changes
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      onEditorChange(editor);
    })
  );

  // Clear decoration cache when a document is closed
  context.subscriptions.push(
    vscode.workspace.onDidCloseTextDocument((doc) => {
      onDocumentClose(doc.uri.toString());
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("jaxtyc.showMenu", async () => {
      const items: { label: string; action: string }[] = [
        { label: "$(refresh) Restart Server", action: "restart" },
        { label: "$(file-code) Check Current File", action: "check" },
        { label: "$(output) Show Output", action: "output" },
        { label: "$(beaker) Trace Function", action: "trace" },
        { label: "$(gear) Open Settings", action: "settings" },
      ];
      const pick = await vscode.window.showQuickPick(items, {
        placeHolder: "jaxtyc",
      });
      if (!pick) return;
      if (pick.action === "output") {
        outputChannel.show();
      } else if (pick.action === "settings") {
        vscode.commands.executeCommand(
          "workbench.action.openSettings",
          "jaxtyc"
        );
      } else if (pick.action === "restart") {
        vscode.commands.executeCommand("jaxtyc.restartServer");
      } else if (pick.action === "check") {
        vscode.commands.executeCommand("jaxtyc.checkFile");
      } else if (pick.action === "trace") {
        vscode.commands.executeCommand("jaxtyc.traceFunction");
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("jaxtyc.restartServer", async () => {
      outputChannel.appendLine("Restarting all jaxtyc servers...");
      await stopAllClients();
      for (const folder of vscode.workspace.workspaceFolders ?? []) {
        await startClientForFolder(folder);
      }
      outputChannel.appendLine("Servers restarted.");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("jaxtyc.checkFile", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("No active Python file to check.");
        return;
      }
      const filePath = editor.document.uri.fsPath;
      const folderPath = folderForPath(
        filePath,
        (vscode.workspace.workspaceFolders ?? []).map((f) => f.uri.fsPath)
      );
      const { pythonPath, args } = getConfig();
      const cmd = buildCheckCommand({
        pythonPath,
        filePath,
        extraArgs: args,
        discovered: folderPath
          ? discoverServerForFolder(folderPath, "lsp", [])
          : undefined,
      });
      if (!cmd) {
        vscode.window.showErrorMessage(
          "jaxtyc not found for this workspace folder."
        );
        return;
      }
      outputChannel.show(true);
      outputChannel.appendLine(`Checking ${filePath}...`);
      execFile(
        cmd.command,
        cmd.args,
        { cwd: folderPath },
        (error, stdout, stderr) => {
          if (stdout) outputChannel.appendLine(stdout);
          if (stderr) outputChannel.appendLine(stderr);
          if (error && !stdout && !stderr)
            outputChannel.appendLine(`Error: ${error.message}`);
        }
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("jaxtyc.traceFunction", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("No active Python file.");
        return;
      }
      const filePath = editor.document.uri.fsPath;
      const funcName = await vscode.window.showInputBox({
        prompt: "Function name to trace",
        placeHolder: "e.g. linear",
      });
      if (!funcName) return;

      const target = `${filePath}::${funcName}`;
      const folderPath = folderForPath(
        filePath,
        (vscode.workspace.workspaceFolders ?? []).map((f) => f.uri.fsPath)
      );
      const { pythonPath } = getConfig();
      const discovered = folderPath
        ? discoverServerForFolder(folderPath, "lsp", [])
        : undefined;

      let cmd: string;
      let args: string[];
      if (pythonPath) {
        cmd = pythonPath;
        args = ["-m", "jaxtyc.cli.main", "trace", target];
      } else if (discovered) {
        const isPython = discovered.args[0] === "-m";
        cmd = discovered.command;
        args = isPython
          ? ["-m", "jaxtyc.cli.main", "trace", target]
          : ["trace", target];
      } else {
        vscode.window.showErrorMessage("jaxtyc not found.");
        return;
      }

      outputChannel.appendLine(`Tracing ${target}...`);

      execFile(cmd, args, { cwd: folderPath }, (error, stdout, stderr) => {
        if (stderr) outputChannel.appendLine(stderr);
        if (error && !stdout) {
          outputChannel.appendLine(`Trace failed: ${error.message}`);
          vscode.window.showErrorMessage(`Trace failed: ${error.message}`);
          return;
        }
        const data = parseTraceOutput(stdout);
        const panel = vscode.window.createWebviewPanel(
          "jaxtycTrace",
          `jaxtyc Trace: ${funcName}`,
          vscode.ViewColumn.Beside,
          { enableScripts: false }
        );
        const cssUri = panel.webview.asWebviewUri(
          vscode.Uri.joinPath(context.extensionUri, "media", "trace.css")
        );
        panel.webview.html = renderTraceHtml(data, cssUri.toString());
      });
    })
  );

  let configRestartTimer: ReturnType<typeof setTimeout> | undefined;

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (
        e.affectsConfiguration("jaxtyc.mode") ||
        e.affectsConfiguration("jaxtyc.pythonPath")
      ) {
        if (configRestartTimer) clearTimeout(configRestartTimer);
        configRestartTimer = setTimeout(async () => {
          await stopAllClients();
          for (const folder of vscode.workspace.workspaceFolders ?? []) {
            await startClientForFolder(folder);
          }
        }, 1000);
      }
    })
  );

  context.subscriptions.push(outputChannel, statusBarItem);
}

export async function deactivate(): Promise<void> {
  disposeDecorations();
  await stopAllClients();
}
