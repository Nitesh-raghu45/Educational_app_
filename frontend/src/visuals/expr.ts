/**
 * Safe evaluator for one-variable formulas in function graphs, e.g. "x^2 - 2*x", "2sin(x)".
 * A small recursive-descent parser: numbers, x, pi, e, + - * / ^ (or **), parentheses,
 * implicit multiplication ("2x", "3(x+1)") and a fixed list of functions. No eval, so text
 * from a document or the model can never run code.
 */

type Fn = (x: number) => number;

// Maps (not plain objects), so names like "constructor" can't resolve to built-in properties.
const FUNCTIONS = new Map<string, (v: number) => number>(Object.entries({
  sqrt: Math.sqrt,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  log: Math.log10,
  ln: Math.log,
  exp: Math.exp,
  abs: Math.abs,
  floor: Math.floor,
  ceil: Math.ceil,
}));
const CONSTANTS = new Map<string, number>([["pi", Math.PI], ["e", Math.E]]);
// Longest first, so "2pix" reads as 2 · pi · x and "sinh" isn't split into "sin" + "h".
const NAMES = ["x", ...CONSTANTS.keys(), ...FUNCTIONS.keys()].sort((a, b) => b.length - a.length);

function splitNames(word: string): string[] {
  const names: string[] = [];
  let rest = word;
  while (rest) {
    const name = NAMES.find((n) => rest.startsWith(n));
    if (!name) throw new Error(`Unknown name "${word}"`);
    names.push(name);
    rest = rest.slice(name.length);
  }
  return names;
}

type Token = { kind: "num"; value: number } | { kind: "id"; value: string } | { kind: "op"; value: string };

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  const text = source.replace(/\*\*/g, "^").replace(/[×·]/g, "*").replace(/[−–]/g, "-").replace(/÷/g, "/");
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (/\s/.test(ch)) {
      i++;
    } else if (/[0-9.]/.test(ch)) {
      const m = /^\d*\.?\d+(?:e[+-]?\d+)?/i.exec(text.slice(i));
      if (!m) throw new Error(`Bad number at ${i}`);
      tokens.push({ kind: "num", value: parseFloat(m[0]) });
      i += m[0].length;
    } else if (/[a-zA-Z]/.test(ch)) {
      const m = /^[a-zA-Z]+/.exec(text.slice(i))!;
      for (const name of splitNames(m[0].toLowerCase())) tokens.push({ kind: "id", value: name });
      i += m[0].length;
    } else if ("+-*/^()".includes(ch)) {
      tokens.push({ kind: "op", value: ch });
      i++;
    } else {
      throw new Error(`Unexpected "${ch}"`);
    }
  }
  return tokens;
}

export function compileExpression(source: string): Fn {
  if (source.length > 200) throw new Error("Formula too long");
  const tokens = tokenize(source);
  let pos = 0;
  const peek = () => tokens[pos];
  const isOp = (value: string) => peek()?.kind === "op" && peek()!.value === value;
  const startsPrimary = () => {
    const t = peek();
    return !!t && (t.kind === "num" || t.kind === "id" || (t.kind === "op" && t.value === "("));
  };

  function expression(): Fn {
    let left = term();
    while (isOp("+") || isOp("-")) {
      const op = tokens[pos++].value;
      const a = left;
      const b = term();
      left = op === "+" ? (x) => a(x) + b(x) : (x) => a(x) - b(x);
    }
    return left;
  }

  function term(): Fn {
    let left = unary();
    for (;;) {
      if (isOp("*") || isOp("/")) {
        const op = tokens[pos++].value;
        const a = left;
        const b = unary();
        left = op === "*" ? (x) => a(x) * b(x) : (x) => a(x) / b(x);
      } else if (startsPrimary()) {
        // Implicit multiplication: 2x, 3(x+1), x sin(x)
        const a = left;
        const b = power();
        left = (x) => a(x) * b(x);
      } else {
        return left;
      }
    }
  }

  function unary(): Fn {
    if (isOp("-")) {
      pos++;
      const inner = unary();
      return (x) => -inner(x);
    }
    if (isOp("+")) {
      pos++;
      return unary();
    }
    return power();
  }

  function power(): Fn {
    const base = primary();
    if (isOp("^")) {
      pos++;
      const exponent = unary(); // right-associative: 2^3^2 = 2^(3^2); -x^2 = -(x^2)
      return (x) => Math.pow(base(x), exponent(x));
    }
    return base;
  }

  function primary(): Fn {
    const t = tokens[pos++];
    if (!t) throw new Error("Unexpected end of formula");
    if (t.kind === "num") {
      const v = t.value;
      return () => v;
    }
    if (t.kind === "op" && t.value === "(") {
      const inner = expression();
      if (!isOp(")")) throw new Error("Missing )");
      pos++;
      return inner;
    }
    if (t.kind === "id") {
      if (t.value === "x") return (x) => x;
      const constant = CONSTANTS.get(t.value);
      if (constant !== undefined) return () => constant;
      const fn = FUNCTIONS.get(t.value);
      if (fn) {
        const arg = isOp("(") ? primary() : power(); // sin(x) or sin x
        return (x) => fn(arg(x));
      }
      throw new Error(`Unknown name "${t.value}"`);
    }
    throw new Error(`Unexpected "${t.value}"`);
  }

  const fn = expression();
  if (pos !== tokens.length) throw new Error("Unexpected text after formula");
  return fn;
}
