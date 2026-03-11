import { describe, it, expect } from "vitest";
import {
  parseTraceOutput,
  renderTraceHtml,
  TraceData,
} from "../src/trace-panel";

const SAMPLE_OUTPUT = `linear(x: Float[batch, seq, d_in], w: Float[d_in, d_out]) -> Float[batch, seq, d_out]

  Line 8: transpose -> (d_out, d_in)  [float32]
  Line 8: dot_general -> (batch, seq, d_in)  [float32]

  Output: (batch, seq, d_in) [MISMATCH]`;

describe("parseTraceOutput", () => {
  it("parses function signature", () => {
    const data = parseTraceOutput(SAMPLE_OUTPUT);
    expect(data.signature).toBe(
      "linear(x: Float[batch, seq, d_in], w: Float[d_in, d_out]) -> Float[batch, seq, d_out]"
    );
  });

  it("parses intermediate operations", () => {
    const data = parseTraceOutput(SAMPLE_OUTPUT);
    expect(data.intermediates).toHaveLength(2);
    expect(data.intermediates[0]).toEqual({
      line: 8,
      op: "transpose",
      shape: "(d_out, d_in)",
      dtype: "float32",
    });
    expect(data.intermediates[1]).toEqual({
      line: 8,
      op: "dot_general",
      shape: "(batch, seq, d_in)",
      dtype: "float32",
    });
  });

  it("parses output with mismatch", () => {
    const data = parseTraceOutput(SAMPLE_OUTPUT);
    expect(data.output).toEqual({
      shape: "(batch, seq, d_in)",
      status: "MISMATCH",
    });
  });

  it("parses output with matches", () => {
    const out = SAMPLE_OUTPUT.replace("[MISMATCH]", "[matches]");
    const data = parseTraceOutput(out);
    expect(data.output!.status).toBe("matches");
  });

  it("handles trace error output", () => {
    const data = parseTraceOutput("func()\n\n  Trace error: could not trace");
    expect(data.error).toBe("could not trace");
    expect(data.intermediates).toHaveLength(0);
  });

  it("handles empty output", () => {
    const data = parseTraceOutput("");
    expect(data.signature).toBe("");
    expect(data.intermediates).toHaveLength(0);
  });
});

// -- renderTraceHtml -----------------------------------------------------------

describe("renderTraceHtml", () => {
  const data: TraceData = {
    signature: "linear(x: Float[batch, d_in]) -> Float[batch, d_out]",
    intermediates: [
      { line: 5, op: "dot_general", shape: "(batch, d_out)", dtype: "float32" },
    ],
    output: { shape: "(batch, d_out)", status: "matches" },
    error: undefined,
  };

  it("includes function signature", () => {
    const html = renderTraceHtml(data, "style.css");
    expect(html).toContain("linear(x: Float[batch, d_in])");
  });

  it("renders intermediate operations as table rows", () => {
    const html = renderTraceHtml(data, "style.css");
    expect(html).toContain("dot_general");
    expect(html).toContain("(batch, d_out)");
  });

  it("marks mismatch output with error class", () => {
    const mismatch: TraceData = {
      ...data,
      output: { shape: "(batch, d_in)", status: "MISMATCH" },
    };
    const html = renderTraceHtml(mismatch, "style.css");
    expect(html).toContain('class="mismatch"');
  });

  it("marks matching output with ok class", () => {
    const html = renderTraceHtml(data, "style.css");
    expect(html).toContain('class="ok"');
  });

  it("renders trace error message", () => {
    const errData: TraceData = {
      signature: "func()",
      intermediates: [],
      output: undefined,
      error: "could not trace",
    };
    const html = renderTraceHtml(errData, "style.css");
    expect(html).toContain("could not trace");
  });

  it("links stylesheet", () => {
    const html = renderTraceHtml(data, "http://example.com/style.css");
    expect(html).toContain('href="http://example.com/style.css"');
  });
});
