// Two tabs of the session page that turn the shot list into answers: which errors a player
// makes (Errors), and what tends to happen in which situation (Patterns). Every number links
// to the moments behind it.

import { useMemo, useState } from "react";
import type { Rally, SessionDetail, Shot } from "../api";
import Court, { COURT, type CourtDot } from "../components/Court";
import { STROKE_LABELS, formatClock, pct } from "../format";
import { ERROR_LABELS, missSide, type ErrorKind, type ShotIndex } from "../review";

type PlayAt = (t: number, videoId?: number) => void;
type Props = {
  s: SessionDetail;
  shots: Shot[];
  rallies: Rally[];
  index: ShotIndex;
  colorOf: (playerId: number | null) => string;
  playAt: PlayAt;
};

const KINDS: ErrorKind[] = ["net", "out", "fault", "double_fault"];

export function Errors({ s, shots, index, colorOf, playAt }: Props) {
  const [who, setWho] = useState("all");
  const [kind, setKind] = useState("all");
  const [stroke, setStroke] = useState("all");
  const [map, setMap] = useState<"landed" | "hit">("landed");
  const [selected, setSelected] = useState<number>();
  const names = Object.fromEntries(s.players.map((p) => [p.player_id, p.name]));
  const errors = useMemo(() => shots.filter((x) => index.errorOf.has(x.id)), [shots, index]);
  const rows = errors.filter(
    (x) =>
      (who === "all" || String(x.player_id) === who) &&
      (kind === "all" || index.errorOf.get(x.id) === kind) &&
      (stroke === "all" || x.stroke === stroke),
  );
  if (!shots.length) return <div className="card empty">No shots found yet.</div>;
  if (!errors.length) {
    return (
      <div className="card empty">
        No errors were found. A ball counts as an error when its measured flight ends in the net or
        outside the lines, which needs the court to be visible.
      </div>
    );
  }
  const dots: CourtDot[] = rows.flatMap((x) => {
    const [px, py] = map === "landed" ? [x.bounce_x, x.bounce_y] : [x.hit_x, x.hit_y];
    if (px == null || py == null) return [];
    return [
      {
        x: px,
        y: py,
        color: colorOf(x.player_id),
        hollow: map === "landed",
        title: `${STROKE_LABELS[x.stroke] ?? x.stroke}, ${ERROR_LABELS[index.errorOf.get(x.id)!]} at ${formatClock(x.t)}`,
        onClick: () => playAt(x.t, x.video_id),
      },
    ];
  });
  const pick = (playerId: number, st: string, k: string) => {
    setWho(String(playerId));
    setStroke(st);
    setKind(k);
  };
  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" }}>
        {s.players.map((p) => {
          const own = shots.filter((x) => x.player_id === p.player_id);
          const bad = errors.filter((x) => x.player_id === p.player_id);
          const strokes = [...new Set(bad.map((x) => x.stroke))];
          const count = (st: string, k: ErrorKind) =>
            bad.filter((x) => x.stroke === st && index.errorOf.get(x.id) === k).length;
          const kinds = KINDS.filter((k) => bad.some((x) => index.errorOf.get(x.id) === k));
          return (
            <div key={p.player_id} className="card stack" style={{ gap: 10 }}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3>
                  <span className="dot" style={{ background: colorOf(p.player_id) }} /> {p.name}
                </h3>
                <span className="small muted">
                  {bad.length} errors in {own.length} shots{own.length ? ` (${pct(bad.length / own.length)})` : ""}
                </span>
              </div>
              {bad.length > 0 && (
                <table className="compact">
                  <thead>
                    <tr>
                      <th />
                      {kinds.map((k) => (
                        <th key={k} className="num">
                          {ERROR_LABELS[k]}
                        </th>
                      ))}
                      <th className="num">of shots</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strokes.map((st) => {
                      const all = own.filter((x) => x.stroke === st).length;
                      const n = bad.filter((x) => x.stroke === st).length;
                      return (
                        <tr key={st}>
                          <td>
                            <button className="link" onClick={() => pick(p.player_id, st, "all")}>
                              {STROKE_LABELS[st] ?? st}
                            </button>
                          </td>
                          {kinds.map((k) => (
                            <td key={k} className="num">
                              {count(st, k) ? (
                                <button className="link" onClick={() => pick(p.player_id, st, k)}>
                                  {count(st, k)}
                                </button>
                              ) : (
                                <span className="muted">·</span>
                              )}
                            </td>
                          ))}
                          <td className="num muted">{all ? pct(n / all) : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          );
        })}
      </div>

      <div className="row" style={{ flexWrap: "wrap" }}>
        <select className="select" value={who} onChange={(e) => setWho(e.target.value)}>
          <option value="all">All players</option>
          {s.players.map((p) => (
            <option key={p.player_id} value={p.player_id}>
              {p.name}
            </option>
          ))}
        </select>
        <select className="select" value={stroke} onChange={(e) => setStroke(e.target.value)}>
          <option value="all">All strokes</option>
          {[...new Set(errors.map((x) => x.stroke))].map((k) => (
            <option key={k} value={k}>
              {STROKE_LABELS[k] ?? k}
            </option>
          ))}
        </select>
        <select className="select" value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="all">All errors</option>
          {KINDS.map((k) => (
            <option key={k} value={k}>
              {ERROR_LABELS[k]}
            </option>
          ))}
        </select>
        <span className="muted small">{rows.length} errors · click one to watch it</span>
      </div>

      <div className="split">
        <div className="card table-wrap" style={{ padding: 0, maxHeight: 520, overflowY: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Player</th>
                <th>Stroke</th>
                <th>Error</th>
                <th className="num">km/h</th>
                <th>Hit from</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((x) => (
                <tr
                  key={x.id}
                  className={`clickable${selected === x.id ? " selected" : ""}`}
                  onClick={() => {
                    setSelected(x.id);
                    playAt(x.t, x.video_id);
                  }}
                >
                  <td className="mono">{formatClock(x.t)}</td>
                  <td>{x.player_id != null ? (names[x.player_id] ?? "?") : "?"}</td>
                  <td>{STROKE_LABELS[x.stroke] ?? x.stroke}</td>
                  <td>
                    {ERROR_LABELS[index.errorOf.get(x.id)!]}
                    {missSide(x) ? <span className="muted small"> · {missSide(x)}</span> : null}
                  </td>
                  <td className="num">{x.speed_kmh == null ? "—" : Math.round(x.speed_kmh)}</td>
                  <td>{x.hit_y == null ? "—" : ZONES[zoneOf(x.hit_y)]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>{map === "landed" ? "Where they landed" : "Where they were hit from"}</h3>
            <select className="select" value={map} onChange={(e) => setMap(e.target.value as "landed" | "hit")}>
              <option value="landed">Landing spots</option>
              <option value="hit">Hitting positions</option>
            </select>
          </div>
          <Court dots={dots} height={380} />
          <div className="small muted" style={{ textAlign: "center" }}>
            {map === "landed"
              ? "Balls into the net never land, so they only show under hitting positions."
              : "Both ends of the court: the players change sides."}
          </div>
        </div>
      </div>
    </div>
  );
}

// --- patterns ----------------------------------------------------------------------------------

const ZONES = ["at the net", "mid-court", "baseline", "behind the baseline"];
/** How far from the net the hitter stood, as one of four bands. */
function zoneOf(hitY: number): number {
  const d = Math.abs(hitY);
  if (d < COURT.serviceLine) return 0;
  if (d < COURT.length / 2 - 1.4) return 1;
  return d < COURT.length / 2 + 1 ? 2 : 3;
}

type Row = { label: string; n: number; hits: number; clips: Shot[] | Rally[] };

export function Patterns({ s, shots, rallies, index, playAt }: Props) {
  const [who, setWho] = useState(String(s.players[0]?.player_id ?? ""));
  const me = Number(who);
  const player = s.players.find((p) => p.player_id === me);
  const isMatch = s.kind === "match";

  const data = useMemo(() => {
    const own = shots.filter((x) => x.player_id === me);
    const rallyShots = own.filter((x) => x.stroke !== "serve");
    const isError = (x: Shot) => index.errorOf.has(x.id);
    const rate = (label: string, group: Shot[]): Row => ({
      label,
      n: group.length,
      hits: group.filter(isError).length,
      clips: group.filter(isError),
    });

    const byZone = ZONES.map((label, z) =>
      rate(label, rallyShots.filter((x) => x.hit_y != null && zoneOf(x.hit_y) === z)),
    );

    // The ball a shot answers: the one before it in the rally, hit by someone else.
    const incoming = new Map<number, Shot>();
    for (const list of index.shotsOf.values()) {
      for (let i = 1; i < list.length; i++) {
        if (list[i - 1].player_id !== list[i].player_id) incoming.set(list[i].id, list[i - 1]);
      }
    }
    const byDepth = ["short", "mid", "deep"].map((d) =>
      rate(`a ${d} ball`, rallyShots.filter((x) => incoming.get(x.id)?.depth === d)),
    );
    const bySpeed = (
      [
        ["a slow ball (under 60 km/h)", 0, 60],
        ["a medium ball", 60, 90],
        ["a fast ball (over 90 km/h)", 90, 1e9],
      ] as [string, number, number][]
    ).map(([label, lo, hi]) =>
      rate(
        label,
        rallyShots.filter((x) => {
          const v = incoming.get(x.id)?.speed_kmh;
          return v != null && v >= lo && v < hi;
        }),
      ),
    );

    const points = rallies.filter((r) => r.winner_id != null && r.score_before != null);
    const byLength: Row[] = (
      [
        ["1–4 shots", 1, 4],
        ["5–8 shots", 5, 8],
        ["9 or more", 9, 1e9],
      ] as [string, number, number][]
    ).map(([label, lo, hi]) => {
      const group = points.filter((r) => r.shot_count >= lo && r.shot_count <= hi);
      const won = group.filter((r) => r.winner_id === me);
      return { label, n: group.length, hits: won.length, clips: group.filter((r) => r.winner_id !== me) };
    });

    // Serves by where they were aimed: thirds of the service box, from the centre line out.
    const third = COURT.singlesWidth / 2 / 3;
    const serves = own.filter((x) => x.stroke === "serve");
    const byServe = (
      [
        ["down the T", 0, third],
        ["at the body", third, 2 * third],
        ["out wide", 2 * third, 1e9],
      ] as [string, number, number][]
    ).map(([label, lo, hi]) => {
      const group = serves.filter(
        (x) => x.bounce_x != null && Math.abs(x.bounce_x) >= lo && Math.abs(x.bounce_x) < hi,
      );
      const good = group.filter((x) => !isError(x));
      const played = good.filter((x) => index.rallyOf.get(x.rally_id ?? -1)?.winner_id != null);
      const won = played.filter((x) => index.rallyOf.get(x.rally_id ?? -1)?.winner_id === me);
      return { label, n: group.length, in: good.length, played: played.length, won: won.length, clips: group };
    });

    const metric = (name: string) => own.map((x) => x.metrics[name]).filter((v): v is number => v != null);
    const split = metric("split_step");
    const recovery = metric("recovery_m").sort((a, b) => a - b);
    return {
      byZone,
      byDepth,
      bySpeed,
      byLength,
      byServe,
      splitRate: split.length ? split.filter((v) => v >= 0.5).length / split.length : null,
      splitN: split.length,
      recoveryMedian: recovery.length ? recovery[Math.floor(recovery.length / 2)] : null,
      recovered: recovery.length ? recovery.filter((v) => v <= 1.5).length / recovery.length : null,
      // The shots after which the player stayed furthest from the centre.
      stranded: own
        .filter((x) => (x.metrics.recovery_m ?? 0) > 2.5)
        .sort((a, b) => b.metrics.recovery_m - a.metrics.recovery_m),
    };
  }, [shots, rallies, index, me]);

  if (!shots.length || !player) return <div className="card empty">No shots found yet.</div>;
  const distance = (player.stats as { distance_m?: number }).distance_m;
  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="row">
        <select className="select" value={who} onChange={(e) => setWho(e.target.value)}>
          {s.players.map((p) => (
            <option key={p.player_id} value={p.player_id}>
              {p.name}
            </option>
          ))}
        </select>
        <span className="small muted">Times are clips: click one to watch it.</span>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))" }}>
        <RateCard
          title="Errors by where you hit from"
          note="Rally shots only, serves left out."
          rows={data.byZone}
          unit="shots"
          what="errors"
          bad
          playAt={playAt}
        />
        <RateCard
          title="Errors when answering…"
          note="How deep and how fast the ball before yours was."
          rows={[...data.byDepth, ...data.bySpeed]}
          unit="replies"
          what="errors"
          bad
          playAt={playAt}
        />
        {isMatch && (
          <RateCard
            title="Points won by rally length"
            note="Point winners are estimated and often wrong, so read this as a hint. Clips: points lost."
            rows={data.byLength}
            unit="points"
            what="won"
            playAt={playAt}
          />
        )}
        <div className="card stack" style={{ gap: 10 }}>
          <h3>Serve placement</h3>
          <table className="compact">
            <thead>
              <tr>
                <th />
                <th className="num">Serves</th>
                <th className="num">In</th>
                {isMatch && <th className="num">Point won</th>}
              </tr>
            </thead>
            <tbody>
              {data.byServe.map((r) => (
                <tr key={r.label}>
                  <td>{r.label}</td>
                  <td className="num">{r.n}</td>
                  <td className="num">{r.n ? pct(r.in / r.n) : "—"}</td>
                  {isMatch && <td className="num">{r.played ? pct(r.won / r.played) : "—"}</td>}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="small muted">
            Only serves whose bounce was measured. A serve into the net has no bounce and is not
            listed here; see the Errors tab.
          </div>
        </div>
        <div className="card stack" style={{ gap: 10 }}>
          <h3>Movement</h3>
          <div className="kpis" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
            <Figure
              label="Split step"
              value={data.splitRate == null ? "—" : pct(data.splitRate)}
              sub={`of ${data.splitN} opponent hits`}
            />
            <Figure
              label="Back near the centre"
              value={data.recovered == null ? "—" : pct(data.recovered)}
              sub="within 1.5 m, 1.2 s after hitting"
            />
            <Figure
              label="Typical recovery"
              value={data.recoveryMedian == null ? "—" : `${data.recoveryMedian.toFixed(1)} m`}
              sub="from the centre line"
            />
            <Figure label="Ran" value={distance == null ? "—" : `${(distance / 1000).toFixed(2)} km`} />
          </div>
          {data.stranded.length > 0 && (
            <div className="small">
              <span className="muted">Left furthest out of position after: </span>
              <Clips clips={data.stranded} playAt={playAt} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Figure({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={{ fontSize: 20 }}>
        {value}
      </div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function Clips({ clips, playAt }: { clips: Shot[] | Rally[]; playAt: PlayAt }) {
  const [all, setAll] = useState(false);
  const shown = all ? clips : clips.slice(0, 6);
  return (
    <span className="clips">
      {shown.map((c) => {
        const t = "t" in c ? c.t : c.start_s;
        return (
          <button key={c.id} className="link mono" onClick={() => playAt(t, "video_id" in c ? c.video_id : undefined)}>
            {formatClock(t)}
          </button>
        );
      })}
      {clips.length > shown.length && (
        <button className="link muted" onClick={() => setAll(true)}>
          +{clips.length - shown.length} more
        </button>
      )}
    </span>
  );
}

function RateCard({
  title,
  note,
  rows,
  unit,
  what,
  bad = false,
  playAt,
}: {
  title: string;
  note: string;
  rows: Row[];
  unit: string;
  what: string;
  bad?: boolean;
  playAt: PlayAt;
}) {
  return (
    <div className="card stack" style={{ gap: 10 }}>
      <h3>{title}</h3>
      <table className="compact">
        <thead>
          <tr>
            <th />
            <th className="num">{unit}</th>
            <th className="num">{what}</th>
            <th style={{ width: "30%" }} />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} style={r.n ? undefined : { opacity: 0.45 }}>
              <td>{r.label}</td>
              <td className="num">{r.n}</td>
              <td className="num">{r.n ? `${r.hits} (${pct(r.hits / r.n)})` : "—"}</td>
              <td>
                {r.n > 0 && (
                  <div className="bar">
                    <div className={bad ? "bad" : ""} style={{ width: pct(r.hits / r.n) }} />
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.some((r) => r.clips.length > 0) && (
        <div className="stack small" style={{ gap: 4 }}>
          {rows
            .filter((r) => r.clips.length > 0)
            .map((r) => (
              <div key={r.label}>
                <span className="muted">{r.label}: </span>
                <Clips clips={r.clips} playAt={playAt} />
              </div>
            ))}
        </div>
      )}
      <div className="small muted">{note} Small counts say little.</div>
    </div>
  );
}
