import { describe, it, expect } from "vitest";
import { parseTraceOutput, TraceData } from "../src/trace-panel";

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
