/**
 * Multi-color shape decorations for inlay hints.
 *
 * Replaces single-color LSP inlay hints with VS Code text decorations
 * that color each segment (dtype, brackets, dims) using the active
 * theme's token colors.
 */

import { readFileSync } from "fs";
import * as path from "path";
import * as vscode from "vscode";
import type { InlayHint as LspInlayHint } from "vscode-languageclient";

// ---------------------------------------------------------------------------
// Theme color detection
// ---------------------------------------------------------------------------

const TYPE_SCOPES = [
  "support.type.python",
  "entity.name.type.class.python",
  "support.type",
  "entity.name.type",
  "storage.type",
];

const STRING_SCOPES = [
  "string.quoted.double.python",
  "string.quoted.single.python",
  "string.quoted",
  "string",
];

const BRACKET_SCOPES = [
  "punctuation.definition.list.begin.python",
  "punctuation.definition.list.end.python",
  "punctuation.section.brackets.begin.python",
  "punctuation.section.brackets.end.python",
  "punctuation.bracket",
  "punctuation",
];

const PUNCTUATION_SCOPES = [
  "punctuation.separator.colon.python",
  "keyword.operator.annotation.python",
  "punctuation.separator",
  "keyword.operator",
];

const KEYWORD_SCOPES = [
  "constant.language.python",
  "constant.language",
  "keyword.constant",
  "keyword",
];

interface TokenColor {
  scope?: string | string[];
  settings?: { foreground?: string };
}

interface ThemeData {
  tokenColors?: TokenColor[];
  include?: string;
}

export interface ThemeColors {
  type: string | vscode.ThemeColor;
  string: string | vscode.ThemeColor;
  bracket: string | vscode.ThemeColor;
  punctuation: string | vscode.ThemeColor;
  keyword: string | vscode.ThemeColor;
  error: string | vscode.ThemeColor;
}

function readThemeJson(filePath: string): ThemeData | undefined {
  try {
    return JSON.parse(readFileSync(filePath, "utf-8"));
  } catch {
    return undefined;
  }
}

function findScopeColor(
  data: ThemeData,
  targetScopes: string[]
): string | undefined {
  if (!data.tokenColors) return undefined;
  for (const target of targetScopes) {
    for (const tc of data.tokenColors) {
      const scopes = Array.isArray(tc.scope)
        ? tc.scope
        : tc.scope
          ? [tc.scope]
          : [];
      if (scopes.includes(target) && tc.settings?.foreground) {
        return tc.settings.foreground;
      }
    }
  }
  return undefined;
}

function findAllColorsInTheme(
  themePath: string,
  partial: Partial<ThemeColors>
): Partial<ThemeColors> {
  const data = readThemeJson(themePath);
  if (!data) return partial;

  if (!partial.type) partial.type = findScopeColor(data, TYPE_SCOPES);
  if (!partial.string) partial.string = findScopeColor(data, STRING_SCOPES);
  if (!partial.bracket) partial.bracket = findScopeColor(data, BRACKET_SCOPES);
  if (!partial.punctuation)
    partial.punctuation = findScopeColor(data, PUNCTUATION_SCOPES);
  if (!partial.keyword)
    partial.keyword = findScopeColor(data, KEYWORD_SCOPES);

  // Follow include chain for any missing colors
  if (
    data.include &&
    (!partial.type ||
      !partial.string ||
      !partial.bracket ||
      !partial.punctuation ||
      !partial.keyword)
  ) {
    const includePath = path.join(path.dirname(themePath), data.include);
    return findAllColorsInTheme(includePath, partial);
  }

  return partial;
}

export function detectThemeColors(): ThemeColors {
  // Fallbacks use VS Code ThemeColor references — resolved at render time
  // by VS Code based on the active theme. No hardcoded hex values.
  const fallbacks: ThemeColors = {
    type: new vscode.ThemeColor("symbolIcon.typeParameterForeground"),
    string: new vscode.ThemeColor("debugTokenExpression.string"),
    bracket: new vscode.ThemeColor("editorBracketHighlight.foreground1"),
    punctuation: new vscode.ThemeColor("foreground"),
    keyword: new vscode.ThemeColor("debugTokenExpression.name"),
    error: new vscode.ThemeColor("editorError.foreground"),
  };

  const themeId = vscode.workspace
    .getConfiguration("workbench")
    .get<string>("colorTheme");
  if (!themeId) return fallbacks;

  // Try to read exact colors from the theme's tokenColors for best match
  for (const ext of vscode.extensions.all) {
    const themes = ext.packageJSON?.contributes?.themes;
    if (!Array.isArray(themes)) continue;
    for (const theme of themes) {
      if (theme.id === themeId || theme.label === themeId) {
        const themePath = path.join(ext.extensionPath, theme.path);
        const detected = findAllColorsInTheme(themePath, {});
        return {
          type: detected.type ?? fallbacks.type,
          string: detected.string ?? fallbacks.string,
          bracket: detected.bracket ?? fallbacks.bracket,
          punctuation: detected.punctuation ?? fallbacks.punctuation,
          keyword: detected.keyword ?? fallbacks.keyword,
          error: fallbacks.error,
        };
      }
    }
  }

  return fallbacks;
}

// ---------------------------------------------------------------------------
// Decoration types and rendering
// ---------------------------------------------------------------------------

// Regex to parse hint labels: "prefix dtype[dims] suffix"
// suffix captures sharding (| P(...)) and/or error text (| message)
// Also matches scalar hints (dtype with no brackets): "prefix dtype suffix"
const SHAPE_HINT_RE = /^(: | -> |-> )?([\w]+)(?:\[([^\]]*)\])?(.*)$/;

// Split suffix into sharding and error parts
// Sharding: " | P(...)"  Error: " | message" or " \u26a0 message"
const SUFFIX_PARTS_RE = /^( \| P\([^)]*\))?(.*)?$/;

// Extract sharding internals: " | P(spec1, spec2, ...)"
const SHARD_RE = /^ \| (P)\(([^)]*)\)$/;

// Decoration types created in render order so `after` pseudo-elements stack
// left-to-right.
let dPrefix: vscode.TextEditorDecorationType;
let dDtype: vscode.TextEditorDecorationType;
let dOpenBracket: vscode.TextEditorDecorationType;
let dDims: vscode.TextEditorDecorationType;
let dCloseBracket: vscode.TextEditorDecorationType;
// Sharding sub-segments (in render order):
// pipe " | ", func name "P", and body "('data', None)" kept as 3 types
// to avoid ordering issues with many small types.
let dShardPipe: vscode.TextEditorDecorationType;
let dShardType: vscode.TextEditorDecorationType;
let dShardBody: vscode.TextEditorDecorationType;
let dError: vscode.TextEditorDecorationType;

let currentColors: ThemeColors | undefined;

interface DecorationSet {
  prefix: vscode.DecorationOptions[];
  dtype: vscode.DecorationOptions[];
  openBracket: vscode.DecorationOptions[];
  dims: vscode.DecorationOptions[];
  closeBracket: vscode.DecorationOptions[];
  shardPipe: vscode.DecorationOptions[];
  shardType: vscode.DecorationOptions[];
  shardBody: vscode.DecorationOptions[];
  error: vscode.DecorationOptions[];
}

function disposeDecorationTypes(): void {
  dPrefix?.dispose();
  dDtype?.dispose();
  dOpenBracket?.dispose();
  dDims?.dispose();
  dCloseBracket?.dispose();
  dShardPipe?.dispose();
  dShardType?.dispose();
  dShardBody?.dispose();
  dError?.dispose();
}

export function initDecorationTypes(): void {
  disposeDecorationTypes();
  currentColors = detectThemeColors();

  // Empty types -- content and color set per-instance via renderOptions.after
  // Created in render order for left-to-right stacking.
  dPrefix = vscode.window.createTextEditorDecorationType({});
  dDtype = vscode.window.createTextEditorDecorationType({});
  dOpenBracket = vscode.window.createTextEditorDecorationType({});
  dDims = vscode.window.createTextEditorDecorationType({});
  dCloseBracket = vscode.window.createTextEditorDecorationType({});
  dShardPipe = vscode.window.createTextEditorDecorationType({});
  dShardType = vscode.window.createTextEditorDecorationType({});
  dShardBody = vscode.window.createTextEditorDecorationType({});
  dError = vscode.window.createTextEditorDecorationType({});
}

function emptySet(): DecorationSet {
  return {
    prefix: [],
    dtype: [],
    openBracket: [],
    dims: [],
    closeBracket: [],
    shardPipe: [],
    shardType: [],
    shardBody: [],
    error: [],
  };
}

function buildDecorationSet(
  hints: LspInlayHint[],
  document: vscode.TextDocument,
  colors: ThemeColors
): DecorationSet {
  const set = emptySet();

  for (const hint of hints) {
    // Extract label text
    const label =
      typeof hint.label === "string"
        ? hint.label
        : (hint.label as Array<{ value: string }>)
            .map((p) => p.value)
            .join("");

    const m = SHAPE_HINT_RE.exec(label);
    if (!m) continue;

    const [, prefix, dtype, dims, rawSuffix] = m;

    // Resolve position -- clamp to actual line end for EOL hints
    const line = hint.position.line;
    let char = hint.position.character;
    if (line >= document.lineCount) continue;
    const lineEnd = document.lineAt(line).range.end.character;
    if (char > lineEnd) char = lineEnd;
    const pos = new vscode.Position(line, char);
    const range = new vscode.Range(pos, pos);

    if (prefix) {
      set.prefix.push({
        range,
        renderOptions: {
          after: { contentText: prefix, color: colors.punctuation },
        },
      });
    }

    set.dtype.push({
      range,
      renderOptions: {
        after: { contentText: dtype, color: colors.type },
      },
    });

    set.openBracket.push({
      range,
      renderOptions: {
        after: { contentText: "[", color: colors.bracket },
      },
    });

    set.dims.push({
      range,
      renderOptions: {
        after: { contentText: dims, color: colors.string },
      },
    });

    set.closeBracket.push({
      range,
      renderOptions: {
        after: { contentText: "]", color: colors.bracket },
      },
    });

    // Split suffix into sharding and error parts
    if (rawSuffix && rawSuffix.trim()) {
      const sm = SUFFIX_PARTS_RE.exec(rawSuffix);
      if (sm) {
        const [, shardingPart, errorPart] = sm;

        // Break sharding into colored sub-segments
        if (shardingPart && shardingPart.trim()) {
          const shardMatch = SHARD_RE.exec(shardingPart);
          if (shardMatch) {
            const [, pName, specsInner] = shardMatch;
            // " | " — punctuation
            set.shardPipe.push({
              range,
              renderOptions: {
                after: { contentText: " | ", color: colors.punctuation },
              },
            });
            // "P" — type color (PartitionSpec)
            set.shardType.push({
              range,
              renderOptions: {
                after: { contentText: pName, color: colors.type },
              },
            });
            // Body: "(specs)" as a single decoration to avoid ordering issues
            const specParts = specsInner.split(/,\s*/);
            const specText = specParts.map((s) => s.trim()).join(", ");
            set.shardBody.push({
              range,
              renderOptions: {
                after: {
                  contentText: `(${specText})`,
                  color: colors.string,
                },
              },
            });
          }
        }

        if (errorPart && errorPart.trim()) {
          set.error.push({
            range,
            renderOptions: {
              after: {
                contentText: errorPart,
                color: colors.error,
              },
            },
          });
        }
      }
    }
  }

  return set;
}

function applyToEditor(
  editor: vscode.TextEditor,
  set: DecorationSet
): void {
  editor.setDecorations(dPrefix, set.prefix);
  editor.setDecorations(dDtype, set.dtype);
  editor.setDecorations(dOpenBracket, set.openBracket);
  editor.setDecorations(dDims, set.dims);
  editor.setDecorations(dCloseBracket, set.closeBracket);
  editor.setDecorations(dShardPipe, set.shardPipe);
  editor.setDecorations(dShardType, set.shardType);
  editor.setDecorations(dShardBody, set.shardBody);
  editor.setDecorations(dError, set.error);
}

function clearEditor(editor: vscode.TextEditor): void {
  editor.setDecorations(dPrefix, []);
  editor.setDecorations(dDtype, []);
  editor.setDecorations(dOpenBracket, []);
  editor.setDecorations(dDims, []);
  editor.setDecorations(dCloseBracket, []);
  editor.setDecorations(dShardPipe, []);
  editor.setDecorations(dShardType, []);
  editor.setDecorations(dShardBody, []);
  editor.setDecorations(dError, []);
}

// Cache of decoration sets per URI so we can re-apply on editor switch
const decorationCache = new Map<string, DecorationSet>();

/**
 * Convert LSP inlay hints to colored decorations.
 *
 * Called from the middleware's provideInlayHints. Returns an empty array
 * so VS Code does not render the single-color default hints.
 */
export function handleInlayHints(
  document: vscode.TextDocument,
  hints: LspInlayHint[]
): vscode.InlayHint[] {
  if (!currentColors || !dPrefix) return [];

  const uri = document.uri.toString();
  const set = buildDecorationSet(hints, document, currentColors);
  decorationCache.set(uri, set);

  // Apply to all visible editors showing this document
  for (const editor of vscode.window.visibleTextEditors) {
    if (editor.document.uri.toString() === uri) {
      applyToEditor(editor, set);
    }
  }

  // Return empty -- we handle rendering via decorations
  return [];
}

/**
 * Re-apply cached decorations when the active editor changes.
 */
export function onEditorChange(
  editor: vscode.TextEditor | undefined
): void {
  if (!editor || !currentColors || !dPrefix) return;
  const uri = editor.document.uri.toString();
  const set = decorationCache.get(uri);
  if (set) {
    applyToEditor(editor, set);
  } else {
    clearEditor(editor);
  }
}

/**
 * Clear cached decorations for a closed document.
 */
export function onDocumentClose(uri: string): void {
  decorationCache.delete(uri);
}

/**
 * Dispose decoration types (call on deactivate or theme change).
 */
export function dispose(): void {
  disposeDecorationTypes();
  decorationCache.clear();
}
