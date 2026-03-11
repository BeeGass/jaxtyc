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
