// A top-down doubles court in metres, drawn with the near baseline at the bottom.
// Court coordinates (as the API gives them): origin at the centre of the net, x across
// (positive to the right seen from the near baseline), y along (positive towards the far end).

export const COURT = {
  length: 23.77,
  doublesWidth: 10.97,
  singlesWidth: 8.23,
  serviceLine: 6.4, // from the net
};

export type CourtDot = {
  x: number;
  y: number;
  color: string;
  title?: string;
  hollow?: boolean;
  onClick?: () => void;
};

/** A straight line on the court, such as a shot from the hitter to the bounce. */
export type CourtLine = {
  from: [number, number];
  to: [number, number];
  color: string;
  faint?: boolean;
  dashed?: boolean;
};

/** A player standing on the court, with a letter inside the marker. */
export type CourtMark = { x: number; y: number; color: string; label?: string };

/** A free line through several points, such as where a player ran. */
export type CourtTrail = { points: [number, number][]; color: string };

const MARGIN = 3; // metres of run-off drawn around the lines

export default function Court({
  dots = [],
  lines = [],
  marks = [],
  trails = [],
  height = 420,
}: {
  dots?: CourtDot[];
  lines?: CourtLine[];
  marks?: CourtMark[];
  trails?: CourtTrail[];
  height?: number;
}) {
  const halfL = COURT.length / 2;
  const halfW = COURT.doublesWidth / 2;
  const halfS = COURT.singlesWidth / 2;
  const vb = {
    x: -halfW - MARGIN,
    y: -halfL - MARGIN,
    w: COURT.doublesWidth + 2 * MARGIN,
    h: COURT.length + 2 * MARGIN,
  };
  // SVG y grows downwards; court y grows towards the far end (top), so flip it.
  const Y = (y: number) => -y;
  return (
    <svg
      viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
      style={{ height, maxWidth: "100%", display: "block", margin: "0 auto" }}
      role="img"
      aria-label="Court diagram"
    >
      <rect x={vb.x} y={vb.y} width={vb.w} height={vb.h} rx={0.6} fill="var(--court)" opacity={0.85} />
      <g stroke="var(--court-line)" strokeWidth={0.08} fill="none" opacity={0.9}>
        <rect x={-halfW} y={-halfL} width={COURT.doublesWidth} height={COURT.length} />
        <line x1={-halfS} y1={-halfL} x2={-halfS} y2={halfL} />
        <line x1={halfS} y1={-halfL} x2={halfS} y2={halfL} />
        <line x1={-halfS} y1={Y(COURT.serviceLine)} x2={halfS} y2={Y(COURT.serviceLine)} />
        <line x1={-halfS} y1={Y(-COURT.serviceLine)} x2={halfS} y2={Y(-COURT.serviceLine)} />
        <line x1={0} y1={Y(COURT.serviceLine)} x2={0} y2={Y(-COURT.serviceLine)} />
        <line x1={0} y1={-halfL} x2={0} y2={-halfL + 0.3} />
        <line x1={0} y1={halfL} x2={0} y2={halfL - 0.3} />
      </g>
      <line
        x1={-halfW - 0.9}
        y1={0}
        x2={halfW + 0.9}
        y2={0}
        stroke="var(--court-line)"
        strokeWidth={0.14}
        strokeDasharray="0.25 0.12"
      />
      {trails.map((tr, i) => (
        <polyline
          key={`t${i}`}
          points={tr.points.map(([x, y]) => `${x},${Y(y)}`).join(" ")}
          fill="none"
          stroke={tr.color}
          strokeWidth={0.12}
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={0.55}
        />
      ))}
      {lines.map((l, i) => (
        <line
          key={`l${i}`}
          x1={l.from[0]}
          y1={Y(l.from[1])}
          x2={l.to[0]}
          y2={Y(l.to[1])}
          stroke={l.color}
          strokeWidth={l.faint ? 0.07 : 0.14}
          strokeDasharray={l.dashed ? "0.3 0.2" : undefined}
          strokeLinecap="round"
          opacity={l.faint ? 0.45 : 0.95}
        />
      ))}
      {dots.map((d, i) => (
        <circle
          key={i}
          cx={d.x}
          cy={Y(d.y)}
          r={0.28}
          fill={d.hollow ? "none" : d.color}
          stroke={d.color}
          strokeWidth={d.hollow ? 0.1 : 0.04}
          opacity={0.9}
          style={{ cursor: d.onClick ? "pointer" : undefined }}
          onClick={d.onClick}
        >
          {d.title && <title>{d.title}</title>}
        </circle>
      ))}
      {marks.map((m, i) => (
        <g key={`m${i}`}>
          <circle cx={m.x} cy={Y(m.y)} r={0.55} fill={m.color} stroke="#10130c" strokeWidth={0.08} />
          {m.label && (
            <text x={m.x} y={Y(m.y) + 0.28} fontSize={0.8} fontWeight={700} textAnchor="middle" fill="#10130c">
              {m.label}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
