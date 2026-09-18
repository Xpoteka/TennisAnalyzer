// A small line chart of one value over sessions, oldest to newest.

export default function Sparkline({
  values,
  width = 220,
  height = 56,
  format = (v: number) => v.toFixed(0),
}: {
  values: (number | null)[];
  width?: number;
  height?: number;
  format?: (v: number) => string;
}) {
  const points = values.map((v, i) => [i, v] as const).filter((p): p is readonly [number, number] => p[1] != null);
  if (points.length === 0) return <div className="small muted">not measured yet</div>;
  const ys = points.map((p) => p[1]);
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const pad = 6;
  const x = (i: number) => (values.length <= 1 ? width / 2 : pad + (i / (values.length - 1)) * (width - 2 * pad));
  const y = (v: number) => (hi === lo ? height / 2 : height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad));
  const last = points[points.length - 1];
  return (
    <div className="row" style={{ gap: 10, alignItems: "center" }}>
      <svg width={width} height={height} role="img" aria-label="trend over sessions">
        <polyline
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeLinejoin="round"
          points={points.map(([i, v]) => `${x(i)},${y(v)}`).join(" ")}
        />
        {points.map(([i, v]) => (
          <circle key={i} cx={x(i)} cy={y(v)} r={3} fill="var(--accent)">
            <title>{format(v)}</title>
          </circle>
        ))}
      </svg>
      <strong>{format(last[1])}</strong>
    </div>
  );
}
