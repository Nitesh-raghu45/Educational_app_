import { describe, expect, it } from "vitest";
import { niceTicks, parseChartSpec } from "./chartSpec";
import { compileExpression } from "./expr";
import { hasVisual, mapOutsideCode, prepareAnswer, stripCitations } from "./markdownPrep";

describe("markdown preparation", () => {
  it("turns citations into chip links and moves punctuation before them", () => {
    expect(prepareAnswer("Plants make glucose [1] .")).toBe("Plants make glucose. [1](#cite-1)");
    expect(prepareAnswer("बड़ा पैकेट जोड़ा जाता है 【2】।")).toBe("बड़ा पैकेट जोड़ा जाता है। [2](#cite-2)");
    expect(prepareAnswer("Fact [1][1] here")).toBe("Fact [1](#cite-1) here");
  });

  it("never rewrites code: chart JSON arrays and diagram labels stay intact", () => {
    const answer = 'Growth [1].\n\n```chart\n{"type": "bar", "labels": ["A"], "series": [{"name": "n", "values": [12]}]}\n```\n\nSee `x[1]` too.';
    const out = prepareAnswer(answer);
    expect(out).toContain('"values": [12]');
    expect(out).toContain("`x[1]`");
    expect(out).toContain("Growth. [1](#cite-1)");
  });

  it("protects an unfinished code block while streaming", () => {
    const streaming = 'Intro [1]\n\n```mermaid\nflowchart LR\n  A["Step [1]"]';
    expect(prepareAnswer(streaming)).toContain('A["Step [1]"]');
  });

  it("converts LaTeX delimiters outside code only", () => {
    expect(prepareAnswer("Area \\(a^2\\)")).toBe("Area $a^2$");
    expect(mapOutsideCode("```\n\\(x\\)\n```", (t) => t.toUpperCase())).toBe("```\n\\(x\\)\n```");
  });

  it("turns figure markers into figure blocks on their own lines, outside code only", () => {
    const out = prepareAnswer("The layers stack up [1]. [[figure:3]] Next part.");
    expect(out).toContain("\n\n```figure\n3\n```\n\n");
    expect(out).toContain("[1](#cite-1)");
    expect(prepareAnswer("```\n[[figure:3]]\n```")).toBe("```\n[[figure:3]]\n```");
    expect(hasVisual("See this:\n\n[[figure:2]]")).toBe(true);
    expect(stripCitations("Intro [1].\n\n[[figure:2]]\n\nEnd.")).toBe("Intro .\n\nEnd.");
  });

  it("strips citations for copying and detects visuals", () => {
    expect(stripCitations("Fact [1]. More 【2†src】")).toBe("Fact . More");
    expect(hasVisual("text\n```mermaid\nflowchart LR\n```")).toBe(true);
    expect(hasVisual("```python\nprint(1)\n```")).toBe(false);
  });
});

describe("formula evaluator", () => {
  const f = (expr: string, x: number) => compileExpression(expr)(x);

  it("handles precedence, powers and unary minus like maths", () => {
    expect(f("x^2 - 2*x + 1", 3)).toBe(4);
    expect(f("-x^2", 3)).toBe(-9);
    expect(f("2^3^2", 0)).toBe(512);
    expect(f("x**2", 4)).toBe(16);
  });

  it("supports implicit multiplication, functions and constants", () => {
    expect(f("2x", 5)).toBe(10);
    expect(f("3(x+1)", 1)).toBe(6);
    expect(f("2sin(x)", Math.PI / 2)).toBeCloseTo(2);
    expect(f("ln(e)", 0)).toBeCloseTo(1);
    expect(f("log(100)", 0)).toBeCloseTo(2);
    expect(f("sqrt(x) + abs(-2)", 9)).toBe(5);
    expect(f("2πx".replace("π", "pi"), 1)).toBeCloseTo(2 * Math.PI);
  });

  it("rejects anything that isn't a formula (no code execution)", () => {
    for (const bad of ["alert(1)", "x; y", "constructor", "x[0]", "2 +", "(x", "window.x"]) {
      expect(() => compileExpression(bad)).toThrow();
    }
  });
});

describe("chart specs", () => {
  it("accepts valid bar, pie and function charts", () => {
    expect(parseChartSpec('{"type":"bar","labels":["A","B"],"series":[{"name":"s","values":[1,2]}]}').type).toBe("bar");
    expect(parseChartSpec('{"type":"pie","labels":["A","B"],"values":[3,1]}').type).toBe("pie");
    const fn = parseChartSpec('{"type":"function","x":[0,4],"functions":[{"expr":"x^2"}]}');
    expect(fn.type === "function" && fn.functions[0].fn(3)).toBe(9);
  });

  it("rejects malformed or misleading data", () => {
    const bad = [
      "not json",
      '{"type":"bar","labels":["A","B"],"series":[{"values":[1]}]}', // value count mismatch
      '{"type":"line","labels":["A"],"series":[{"values":["1"]}]}', // string number
      '{"type":"pie","labels":["A","B"],"values":[-1,2]}', // negative slice
      '{"type":"function","x":[5,1],"functions":[{"expr":"x"}]}', // reversed range
      '{"type":"function","functions":[{"expr":"alert(1)"}]}', // not a formula
      '{"type":"radar","labels":["A"]}',
    ];
    for (const source of bad) expect(() => parseChartSpec(source)).toThrow();
  });

  it("makes round axis ticks", () => {
    expect(niceTicks(0, 97)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(niceTicks(-3, 3).includes(0)).toBe(true);
  });
});
