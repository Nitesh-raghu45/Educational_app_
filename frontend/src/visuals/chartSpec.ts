/** Validation for ```chart blocks written by the model. Anything malformed is rejected, and the
 * renderer then shows the raw block instead of a misleading chart. */

import { compileExpression } from "./expr";

export interface Series {
  name: string;
  values: number[];
}

export type ChartSpec =
  | { type: "bar" | "line"; title?: string; xLabel?: string; yLabel?: string; labels: string[]; series: Series[] }
  | { type: "pie"; title?: string; labels: string[]; values: number[] }
  | {
      type: "function";
      title?: string;
      x: [number, number];
      functions: { label: string; fn: (x: number) => number }[];
    };

const MAX_POINTS = 40;
const MAX_SERIES = 6;

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const text = (v: unknown, max = 80): string | undefined =>
  typeof v === "string" && v.trim() ? v.trim().slice(0, max) : undefined;

export function parseChartSpec(source: string): ChartSpec {
  let raw: Record<string, unknown>;
  try {
    raw = JSON.parse(source);
  } catch {
    throw new Error("The chart data isn't valid JSON");
  }
  if (!raw || typeof raw !== "object") throw new Error("The chart data must be an object");
  const title = text(raw.title, 120);

  if (raw.type === "bar" || raw.type === "line") {
    const labels = Array.isArray(raw.labels) ? raw.labels.map((l) => String(l).slice(0, 40)) : [];
    if (labels.length < 1 || labels.length > MAX_POINTS) throw new Error("A chart needs 1–40 labels");
    const series = (Array.isArray(raw.series) ? raw.series : []).slice(0, MAX_SERIES).map((s: any, i: number) => {
      const values = Array.isArray(s?.values) ? s.values : [];
      if (values.length !== labels.length || !values.every(isNum)) {
        throw new Error("Every series needs one number per label");
      }
      return { name: text(s?.name, 40) ?? `Series ${i + 1}`, values };
    });
    if (series.length === 0) throw new Error("A chart needs at least one series");
    return { type: raw.type, title, xLabel: text(raw.x_label, 40), yLabel: text(raw.y_label, 40), labels, series };
  }

  if (raw.type === "pie") {
    const labels = Array.isArray(raw.labels) ? raw.labels.map((l) => String(l).slice(0, 40)) : [];
    const values = Array.isArray(raw.values) ? raw.values : [];
    if (labels.length < 2 || labels.length > 12 || values.length !== labels.length) {
      throw new Error("A pie chart needs 2–12 labels with one value each");
    }
    if (!values.every((v) => isNum(v) && v >= 0) || values.every((v) => v === 0)) {
      throw new Error("Pie values must be non-negative numbers");
    }
    return { type: "pie", title, labels, values };
  }

  if (raw.type === "function") {
    const range = Array.isArray(raw.x) && raw.x.length === 2 && raw.x.every(isNum) ? (raw.x as [number, number]) : [-10, 10];
    const [lo, hi] = range;
    if (!(hi > lo)) throw new Error("The x range must go from low to high");
    const fns = (Array.isArray(raw.functions) ? raw.functions : []).slice(0, 4).map((f: any) => {
      const expr = text(f?.expr, 200);
      if (!expr) throw new Error("Each function needs a formula");
      return { label: text(f?.label, 60) ?? `y = ${expr}`, fn: compileExpression(expr) };
    });
    if (fns.length === 0) throw new Error("A function graph needs at least one formula");
    return { type: "function", title, x: [lo, hi], functions: fns };
  }

  throw new Error(`Unknown chart type "${String(raw.type)}"`);
}

/** Round axis ticks: 0, 5, 10… or 0, 0.2, 0.4… covering [min, max]. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (min === max) {
    const pad = Math.abs(min) || 1;
    min -= pad;
    max += pad;
  }
  const rough = (max - min) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ?? 10 * magnitude;
  const ticks: number[] = [];
  for (let v = Math.floor(min / step) * step; v <= max + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toFixed(10)));
  }
  if (ticks[ticks.length - 1] < max) ticks.push(Number((ticks[ticks.length - 1] + step).toFixed(10)));
  return ticks;
}

export function formatNumber(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1e6) return `${(v / 1e6).toFixed(abs >= 1e7 ? 0 : 1)}M`;
  if (abs >= 1e4) return `${(v / 1e3).toFixed(abs >= 1e5 ? 0 : 1)}k`;
  return Number(v.toFixed(2)).toString();
}
