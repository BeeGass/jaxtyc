export interface Intermediate {
  line: number;
  op: string;
  shape: string;
  dtype: string;
}

export interface TraceOutput {
  shape: string;
  status: string;
}

export interface TraceData {
  signature: string;
  intermediates: Intermediate[];
  output: TraceOutput | undefined;
  error: string | undefined;
}

const INTERMEDIATE_RE = /^\s+Line (\d+): (\S+) -> (\([^)]*\))\s+\[(\w+)]$/;
const OUTPUT_RE = /^\s+Output: (\([^)]*\))\s*\[(\w+)]$/;
const ERROR_RE = /^\s+Trace error: (.+)$/;

export function parseTraceOutput(raw: string): TraceData {
  const lines = raw.split("\n");
  const signature = lines[0]?.trim() ?? "";
  const intermediates: Intermediate[] = [];
  let output: TraceOutput | undefined;
  let error: string | undefined;

  for (const line of lines.slice(1)) {
    let m = INTERMEDIATE_RE.exec(line);
    if (m) {
      intermediates.push({
        line: parseInt(m[1], 10),
        op: m[2],
        shape: m[3],
        dtype: m[4],
      });
      continue;
    }
    m = OUTPUT_RE.exec(line);
    if (m) {
      output = { shape: m[1], status: m[2] };
      continue;
    }
    m = ERROR_RE.exec(line);
    if (m) {
      error = m[1];
    }
  }

  return { signature, intermediates, output, error };
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderTraceHtml(data: TraceData, cssUri: string): string {
  const rows = data.intermediates
    .map(
      (i) =>
        `<tr><td>${i.line}</td><td>${escapeHtml(i.op)}</td>` +
        `<td><code>${escapeHtml(i.shape)}</code></td>` +
        `<td>${escapeHtml(i.dtype)}</td></tr>`
    )
    .join("\n");

  let outputHtml = "";
  if (data.output) {
    const cls = data.output.status === "MISMATCH" ? "mismatch" : "ok";
    outputHtml = `<div class="${cls}">
      <strong>Output:</strong> <code>${escapeHtml(data.output.shape)}</code>
      <span class="status">[${escapeHtml(data.output.status)}]</span>
    </div>`;
  }

  let errorHtml = "";
  if (data.error) {
    errorHtml = `<div class="error">Trace error: ${escapeHtml(data.error)}</div>`;
  }

  return `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="${cssUri}">
</head>
<body>
  <h2>${escapeHtml(data.signature)}</h2>
  ${
    data.intermediates.length > 0
      ? `<table>
    <thead><tr><th>Line</th><th>Operation</th><th>Shape</th><th>dtype</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`
      : ""
  }
  ${outputHtml}
  ${errorHtml}
</body>
</html>`;
}
