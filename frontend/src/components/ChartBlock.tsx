import { useId, useMemo } from "react";
import { type ChartSpec, formatNumber, niceTicks, parseChartSpec } from "../visuals/chartSpec";

const W = 640;
const H = 340;
const M = { top: 20, right: 20, bottom: 56, left: 56 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;
const color = (i: number) => `var(--chart-${(i % 6) + 1})`;
const short = (s: string, n = 14) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function YAxis({ ticks, y, label }: { ticks: number[]; y: (v: number) => number; label?: string }) {
  return (
    <g>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className="chart-grid" />
          <text x={M.left - 8} y={y(t)} className="chart-tick" textAnchor="end" dominantBaseline="middle">
            {formatNumber(t)}
          </text>
        </g>
      ))}
      {label && (
        <text className="chart-axis-label" transform={`translate(14 ${M.top + PH / 2}) rotate(-90)`} textAnchor="middle">
          {label}
        </text>
      )}
    </g>
  );
}

function XLabels({ labels, x, label }: { labels: string[]; x: (i: number) => number; label?: string }) {
  const tilt = labels.length > 6;
  return (
    <g>
      {labels.map((l, i) => (
        <text
          key={i}
          className="chart-tick"
          textAnchor={tilt ? "end" : "middle"}
          transform={`translate(${x(i)} ${M.top + PH + 16})${tilt ? " rotate(-30)" : ""}`}
        >
          <title>{l}</title>
          {short(l, tilt ? 12 : 16)}
        </text>
      ))}
      {label && (
        <text className="chart-axis-label" x={M.left + PW / 2} y={H - 6} textAnchor="middle">
          {label}
        </text>
      )}
    </g>
  );
}

function Legend({ names }: { names: string[] }) {
  if (names.length < 2) return null;
  return (
    <div className="chart-legend">
      {names.map((n, i) => (
        <span key={n + i}>
          <i style={{ background: color(i) }} />
          {n}
        </span>
      ))}
    </div>
  );
}

function BarLine({ spec }: { spec: Extract<ChartSpec, { type: "bar" | "line" }> }) {
  const all = spec.series.flatMap((s) => s.values);
  const ticks = niceTicks(Math.min(0, ...all), Math.max(0, ...all));
  const lo = ticks[0];
  const hi = ticks[ticks.length - 1];
  const y = (v: number) => M.top + PH - ((v - lo) / (hi - lo)) * PH;
  const band = PW / spec.labels.length;
  const x = (i: number) => M.left + band * (i + 0.5);
  const barW = Math.min(48, (band * 0.75) / spec.series.length);
  return (
    <>
      <Legend names={spec.series.map((s) => s.name)} />
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
        <YAxis ticks={ticks} y={y} label={spec.yLabel} />
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} className="chart-axis" />
        {spec.type === "bar"
          ? spec.series.map((s, si) =>
              s.values.map((v, i) => {
                const left = x(i) - (barW * spec.series.length) / 2 + si * barW;
                return (
                  <rect key={`${si}-${i}`} x={left + 1} width={barW - 2} y={Math.min(y(v), y(0))}
                    height={Math.max(1, Math.abs(y(v) - y(0)))} rx={3} fill={color(si)}>
                    <title>{`${s.name} · ${spec.labels[i]}: ${v}`}</title>
                  </rect>
                );
              }),
            )
          : spec.series.map((s, si) => (
              <g key={si}>
                <polyline points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")} fill="none"
                  stroke={color(si)} strokeWidth={2.5} strokeLinejoin="round" />
                {s.values.map((v, i) => (
                  <circle key={i} cx={x(i)} cy={y(v)} r={3.5} fill={color(si)}>
                    <title>{`${s.name} · ${spec.labels[i]}: ${v}`}</title>
                  </circle>
                ))}
              </g>
            ))}
        <XLabels labels={spec.labels} x={x} label={spec.xLabel} />
      </svg>
    </>
  );
}

function Pie({ spec }: { spec: Extract<ChartSpec, { type: "pie" }> }) {
  const total = spec.values.reduce((a, b) => a + b, 0);
  const cx = 170;
  const cy = H / 2;
  const r = 120;
  let angle = -Math.PI / 2;
  const slices = spec.values.map((v, i) => {
    const sweep = (v / total) * Math.PI * 2;
    const start = angle;
    angle += sweep;
    const [x1, y1] = [cx + r * Math.cos(start), cy + r * Math.sin(start)];
    const [x2, y2] = [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
    const d = sweep >= Math.PI * 2 - 1e-9
      ? `M ${cx - r} ${cy} a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0`
      : `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${sweep > Math.PI ? 1 : 0} 1 ${x2} ${y2} Z`;
    return { d, i, pct: (v / total) * 100 };
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
      {slices.map((s) => (
        <path key={s.i} d={s.d} fill={color(s.i)} className="chart-slice">
          <title>{`${spec.labels[s.i]}: ${spec.values[s.i]} (${s.pct.toFixed(1)}%)`}</title>
        </path>
      ))}
      {spec.labels.map((l, i) => (
        <g key={i} transform={`translate(330 ${cy - (spec.labels.length * 24) / 2 + i * 24})`}>
          <rect width={12} height={12} rx={3} y={-9} fill={color(i)} />
          <text x={20} className="chart-legend-text">
            {short(l, 26)} · {slices[i].pct.toFixed(1)}%
          </text>
        </g>
      ))}
    </svg>
  );
}

function FunctionGraph({ spec }: { spec: Extract<ChartSpec, { type: "function" }> }) {
  const clipId = `plot-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const [x0, x1] = spec.x;
  const samples = 240;
  const curves = spec.functions.map((f) =>
    Array.from({ length: samples + 1 }, (_, k) => {
      const xv = x0 + ((x1 - x0) * k) / samples;
      const yv = f.fn(xv);
      return [xv, Number.isFinite(yv) ? yv : NaN] as const;
    }),
  );
  // Robust y range: ignore the extreme 2% (e.g. near asymptotes of tan or 1/x).
  const ys = curves.flat().map(([, yv]) => yv).filter(Number.isFinite).sort((a, b) => a - b);
  if (ys.length === 0) return <div className="visual-error-note">The formula has no real values in this range.</div>;
  const q = (p: number) => ys[Math.min(ys.length - 1, Math.max(0, Math.round(p * (ys.length - 1))))];
  const yTicks = niceTicks(Math.min(q(0.02), 0), Math.max(q(0.98), 0));
  const xTicks = niceTicks(x0, x1, 8).filter((t) => t >= x0 && t <= x1);
  const lo = yTicks[0];
  const hi = yTicks[yTicks.length - 1];
  const sx = (v: number) => M.left + ((v - x0) / (x1 - x0)) * PW;
  const sy = (v: number) => M.top + PH - ((v - lo) / (hi - lo)) * PH;
  const paths = curves.map((pts) => {
    let d = "";
    let pen = false;
    for (const [xv, yv] of pts) {
      if (!Number.isFinite(yv) || yv < lo - (hi - lo) || yv > hi + (hi - lo)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"} ${sx(xv).toFixed(1)} ${sy(yv).toFixed(1)} `;
      pen = true;
    }
    return d;
  });
  return (
    <>
      <Legend names={spec.functions.map((f) => f.label)} />
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
        <defs>
          <clipPath id={clipId}>
            <rect x={M.left} y={M.top} width={PW} height={PH} />
          </clipPath>
        </defs>
        <YAxis ticks={yTicks} y={sy} />
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={M.top} y2={M.top + PH} className="chart-grid" />
            <text x={sx(t)} y={M.top + PH + 16} className="chart-tick" textAnchor="middle">
              {formatNumber(t)}
            </text>
          </g>
        ))}
        {lo <= 0 && hi >= 0 && <line x1={M.left} x2={W - M.right} y1={sy(0)} y2={sy(0)} className="chart-axis" />}
        {x0 <= 0 && x1 >= 0 && <line x1={sx(0)} x2={sx(0)} y1={M.top} y2={M.top + PH} className="chart-axis" />}
        <g clipPath={`url(#${clipId})`}>
          {paths.map((d, i) => (
            <path key={i} d={d} fill="none" stroke={color(i)} strokeWidth={2.5} strokeLinejoin="round" />
          ))}
        </g>
        <text className="chart-axis-label" x={M.left + PW / 2} y={H - 10} textAnchor="middle">x</text>
      </svg>
      {spec.functions.length === 1 && <div className="chart-caption">{spec.functions[0].label}</div>}
    </>
  );
}

function DataTable({ spec }: { spec: ChartSpec }) {
  if (spec.type === "function") return null;
  const rows = spec.type === "pie"
    ? spec.labels.map((l, i) => [l, spec.values[i]])
    : spec.labels.map((l, i) => [l, ...spec.series.map((s) => s.values[i])]);
  const head = spec.type === "pie" ? ["", "Value"] : ["", ...spec.series.map((s) => s.name)];
  return (
    <details className="chart-data">
      <summary>Show data</summary>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>{head.map((h, i) => <th key={i}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export default function ChartBlock({ source }: { source: string }) {
  const result = useMemo(() => {
    try {
      return { spec: parseChartSpec(source) };
    } catch (e) {
      return { error: (e as Error).message };
    }
  }, [source]);

  let body: JSX.Element;
  try {
    if (!result.spec) throw new Error(result.error);
    const spec = result.spec;
    body = (
      <figure className="visual chart" role="img" aria-label={spec.title ?? `${spec.type} chart`}>
        {spec.title && <figcaption className="visual-title">{spec.title}</figcaption>}
        {spec.type === "pie" ? <Pie spec={spec} /> : spec.type === "function" ? <FunctionGraph spec={spec} /> : <BarLine spec={spec} />}
        <DataTable spec={spec} />
      </figure>
    );
  } catch (e) {
    body = (
      <div className="visual visual-error">
        <div className="visual-error-note">Couldn't draw this chart ({(e as Error).message}).</div>
        <pre>
          <code>{source}</code>
        </pre>
      </div>
    );
  }
  return body;
}
