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

const MARGIN = 3; // metres of run-off drawn around the lines

export default function Court({ dots = [], height = 420 }: { dots?: CourtDot[]; height?: number }) {
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
    </svg>
  );
}
