import { useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Rally, type SessionDetail, type Shot, type Video } from "../api";
import { Avatar, KindBadge, StatusBadge } from "../components/Badges";
import Court, { type CourtDot } from "../components/Court";
import {
  STROKE_LABELS,
  formatBytes,
  formatClock,
  formatDate,
  formatDuration,
  pct,
  sessionTitle,
} from "../format";
import { useData } from "../hooks";

type Tab = "overview" | "shots" | "points" | "videos";

const running = (s?: SessionDetail) => s?.status === "queued" || s?.status === "processing";

// One colour per player (A, B, C, D), readable on the court blue in both themes.
export const PLAYER_COLORS = ["#c8e639", "#ff9f43", "#5ee0ff", "#ff6bcb"];

export default function SessionPage() {
  const id = Number(useParams().id);
  const navigate = useNavigate();
  const session = useData(() => api.session(id), [id], 2000, running);
  const ready = session.data?.status === "ready";
  const shots = useData(() => (ready ? api.shots(id) : Promise.resolve([])), [id, ready]);
  const rallies = useData(() => (ready ? api.rallies(id) : Promise.resolve([])), [id, ready]);
  const [tab, setTab] = useState<Tab>("overview");
  const player = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState<{ video: Video; t: number }>();

  const s = session.data;
  if (session.error && !s) return <div className="notice notice-error">{session.error}</div>;
  if (!s) return null;

  const colorOf = (playerId: number | null) => {
    const i = s.players.findIndex((p) => p.player_id === playerId);
    return i >= 0 ? PLAYER_COLORS[i % PLAYER_COLORS.length] : "#bbbbbb";
  };

  /** Play the moment at session time `t` in the video that covers it. */
  const playAt = (t: number, videoId?: number) => {
    const video =
      s.videos.find((v) => v.id === videoId) ??
      s.videos.find((v) => t >= v.offset_s && t <= v.offset_s + (v.duration_s ?? 0)) ??
      s.videos[0];
    if (!video?.proxy_url) return;
    const local = t - video.offset_s - (video.proxy_start_pts ?? 0);
    setPlaying({ video, t: Math.max(0, local - 2) });
    requestAnimationFrame(() => player.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  };

  const isMatch = s.kind === "match";
  const tabs: [Tab, string][] = [
    ["overview", "Overview"],
    ["shots", `Shots${shots.data?.length ? ` (${shots.data.length})` : ""}`],
    ...(isMatch ? ([["points", "Points"]] as [Tab, string][]) : []),
    ["videos", `Videos (${s.videos.length})`],
  ];

  return (
    <div className="stack" style={{ gap: 20 }}>
      <Header s={s} onChanged={session.reload} onDeleted={() => navigate("/sessions")} />
      {running(s) && s.job && (
        <div className="card stack" style={{ gap: 8 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <strong>{s.job.status === "queued" ? "Waiting for the analyser…" : s.job.stage}</strong>
            <span className="row">
              <span className="muted small">{pct(s.job.progress)}</span>
              <button className="btn btn-ghost small" onClick={() => api.cancelJob(s.job!.id).then(session.reload)}>
                Cancel
              </button>
            </span>
          </div>
          <div className="progress">
            <div style={{ width: pct(s.job.progress) }} />
          </div>
          <div className="small muted">
            Long videos take a while: roughly the video's own length on a Mac, more on a server
            without a GPU. You can leave this page.
          </div>
        </div>
      )}
      {s.status === "failed" && (
        <div className="notice notice-error">
          Analysis failed: {s.error}{" "}
          <button className="btn small" onClick={() => api.reanalyze(id).then(session.reload)}>
            Try again
          </button>
        </div>
      )}
      <Warnings s={s} />

      {playing?.video.proxy_url && (
        <video
          ref={player}
          key={playing.video.id}
          className="video-player"
          src={playing.video.proxy_url}
          controls
          autoPlay
          onLoadedMetadata={(e) => (e.currentTarget.currentTime = playing.t)}
        />
      )}

      <div>
        <div className="tabs" role="tablist">
          {tabs.map(([key, label]) => (
            <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>
              {label}
            </button>
          ))}
        </div>
        {tab === "overview" && <Overview s={s} shots={shots.data ?? []} colorOf={colorOf} playAt={playAt} />}
        {tab === "shots" && <ShotTable s={s} shots={shots.data ?? []} playAt={playAt} />}
        {tab === "points" && <Points s={s} rallies={rallies.data ?? []} playAt={playAt} />}
        {tab === "videos" && <Videos s={s} onChanged={session.reload} playAt={playAt} />}
      </div>
    </div>
  );
}

function Header({
  s,
  onChanged,
  onDeleted,
}: {
  s: SessionDetail;
  onChanged: () => void;
  onDeleted: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(s.name ?? "");
  const save = async () => {
    await api.patchSession(s.id, { name });
    setEditing(false);
    onChanged();
  };
  return (
    <div className="page-head">
      <div>
        <div className="row small" style={{ marginBottom: 6 }}>
          <Link to="/sessions" className="muted">
            Sessions
          </Link>
          <span className="muted">/</span>
        </div>
        {editing ? (
          <form className="row" onSubmit={(e) => (e.preventDefault(), save())}>
            <input className="input" autoFocus value={name} onChange={(e) => setName(e.target.value)} />
            <button className="btn btn-primary">Save</button>
            <button type="button" className="btn btn-ghost" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </form>
        ) : (
          <div className="row" style={{ gap: 12 }}>
            <h1>{sessionTitle(s)}</h1>
            <KindBadge s={s} />
            <StatusBadge s={s} />
            <button className="btn btn-ghost small" onClick={() => setEditing(true)}>
              Rename
            </button>
          </div>
        )}
        <p>
          {formatDate(s.recorded_at ?? s.created_at)} · {formatDuration(s.duration_s)}
          {s.players.length > 0 && ` · ${s.players.map((p) => p.name).join(" vs ")}`}
        </p>
      </div>
      <div className="row">
        {s.status !== "queued" && s.status !== "processing" && (
          <>
            <label className="row small muted" title="Correct the automatic training/match decision">
              Type
              <select
                className="select"
                value={s.kind_source === "manual" ? s.kind : "auto"}
                onChange={(e) => api.patchSession(s.id, { kind: e.target.value }).then(onChanged)}
              >
                <option value="auto">Automatic{s.kind_source === "auto" && s.kind !== "unknown" ? ` (${s.kind})` : ""}</option>
                <option value="training">Training</option>
                <option value="match">Match</option>
              </select>
            </label>
            <button className="btn" onClick={() => api.reanalyze(s.id).then(onChanged)}>
              Re-analyse
            </button>
          </>
        )}
        <button
          className="btn btn-danger"
          onClick={async () => {
            if (!confirm("Delete this session and its results? The video files stay on the server.")) return;
            await api.deleteSession(s.id);
            onDeleted();
          }}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

function Warnings({ s }: { s: SessionDetail }) {
  const warnings = [
    ...((s.summary.warnings as string[] | undefined) ?? []),
    ...s.videos.flatMap((v) => v.warnings.map((w) => (s.videos.length > 1 ? `${v.filename}: ${w}` : w))),
  ];
  if (!warnings.length) return null;
  return (
    <details className="notice">
      <summary style={{ cursor: "pointer" }}>
        {warnings.length} note{warnings.length > 1 ? "s" : ""} about this analysis
      </summary>
      <ul style={{ margin: "8px 0 0", paddingLeft: 20 }}>
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </details>
  );
}

type PlayerStats = {
  shots?: number;
  by_stroke?: Record<string, number>;
  speed?: Record<string, { avg: number; max: number; n: number }>;
  net_clearance_avg_m?: number;
  in_pct?: number;
  winners?: number;
  errors?: number;
  first_serve_in_pct?: number;
  points_won?: number;
  distance_m?: number;
};

function Overview({
  s,
  shots,
  colorOf,
  playAt,
}: {
  s: SessionDetail;
  shots: Shot[];
  colorOf: (id: number | null) => string;
  playAt: (t: number, videoId?: number) => void;
}) {
  const [stroke, setStroke] = useState("all");
  const score = s.summary.score as { sets?: number[][]; text?: string } | undefined;
  const bounces = shots.filter(
    (sh) => sh.bounce_x != null && sh.bounce_y != null && (stroke === "all" || sh.stroke === stroke),
  );
  const dots: CourtDot[] = bounces.map((sh) => ({
    x: sh.bounce_x!,
    y: sh.bounce_y!,
    color: colorOf(sh.player_id),
    hollow: sh.in_court === false,
    title: `${STROKE_LABELS[sh.stroke] ?? sh.stroke} at ${formatClock(sh.t)}${sh.speed_kmh ? `, ${Math.round(sh.speed_kmh)} km/h` : ""}`,
    onClick: () => playAt(sh.t, sh.video_id),
  }));
  const strokes = useMemo(() => [...new Set(shots.map((x) => x.stroke))], [shots]);

  if (s.status !== "ready" && !shots.length) {
    return <div className="card empty">Results appear here once the analysis is done.</div>;
  }
  if (!s.players.length && !shots.length) {
    return (
      <div className="card empty">
        No players or shots were found in this session yet. Shot and player detection arrive in
        the next update of the analyser.
      </div>
    );
  }
  return (
    <div className="stack" style={{ gap: 20 }}>
      {score?.text && <Scoreboard s={s} />}
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
        {s.players.map((p, i) => {
          const st = p.stats as PlayerStats;
          return (
            <div key={p.player_id} className="card stack" style={{ gap: 12 }}>
              <Link to={`/players/${p.player_id}`} className="row" style={{ textDecoration: "none", gap: 12 }}>
                <Avatar name={p.name} url={p.thumbnail_url} />
                <div>
                  <h3>
                    <span
                      style={{
                        display: "inline-block",
                        width: 10,
                        height: 10,
                        borderRadius: 5,
                        background: PLAYER_COLORS[i % PLAYER_COLORS.length],
                        marginRight: 8,
                      }}
                    />
                    {p.name}
                  </h3>
                  <div className="small muted">
                    {p.new_profile ? "New player" : "Recognised"}
                    {p.handedness ? ` · ${p.handedness}-handed` : ""}
                  </div>
                </div>
              </Link>
              <div className="kpis" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                <Kpi label="Shots" value={st.shots} />
                <Kpi label="Forehands" value={st.by_stroke?.forehand} />
                <Kpi label="Backhands" value={st.by_stroke?.backhand} />
                <Kpi label="FH speed" value={st.speed?.forehand?.avg} unit="km/h" sub={st.speed?.forehand ? `max ${Math.round(st.speed.forehand.max)}` : undefined} />
                <Kpi label="BH speed" value={st.speed?.backhand?.avg} unit="km/h" sub={st.speed?.backhand ? `max ${Math.round(st.speed.backhand.max)}` : undefined} />
                <Kpi label="Serve" value={st.speed?.serve?.avg} unit="km/h" sub={st.speed?.serve ? `max ${Math.round(st.speed.serve.max)}` : undefined} />
                <Kpi label="In" value={st.in_pct != null ? Math.round(st.in_pct * 100) : undefined} unit="%" />
                <Kpi label="Over net" value={st.net_clearance_avg_m} unit="m" digits={2} />
                {s.kind === "match" ? (
                  <Kpi label="Points won" value={st.points_won} />
                ) : (
                  <Kpi label="Ran" value={st.distance_m != null ? st.distance_m / 1000 : undefined} unit="km" digits={2} />
                )}
              </div>
            </div>
          );
        })}
      </div>
      {bounces.length > 0 || shots.some((x) => x.bounce_x != null) ? (
        <div className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>Where the balls landed</h2>
            <select className="select" value={stroke} onChange={(e) => setStroke(e.target.value)}>
              <option value="all">All strokes</option>
              {strokes.map((k) => (
                <option key={k} value={k}>
                  {STROKE_LABELS[k] ?? k}
                </option>
              ))}
            </select>
          </div>
          <Court dots={dots} />
          <div className="small muted" style={{ textAlign: "center" }}>
            Filled: in · hollow: out · click a dot to watch the shot
          </div>
        </div>
      ) : shots.length > 0 ? (
        <div className="card muted small">
          The court was not visible in this video, so bounce positions, ball speed and height are
          not available. Shot counts and technique still are.
        </div>
      ) : null}
    </div>
  );
}

function Scoreboard({ s }: { s: SessionDetail }) {
  const score = s.summary.score as
    | { sets?: number[][]; current?: number[]; players?: number[]; points?: number }
    | undefined;
  if (!score?.players) return null;
  const columns = [...(score.sets ?? [])];
  const current = score.current ?? [0, 0];
  const unfinished = current[0] + current[1] > 0 || columns.length === 0;
  if (unfinished) columns.push(current);
  return (
    <div className="card stack" style={{ gap: 10 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Score</h2>
        <span className="small muted">
          counted from {score.points ?? 0} points · point winners are estimated and can be wrong
        </span>
      </div>
      <table style={{ width: "auto" }}>
        <tbody>
          {score.players.map((pid, i) => {
            const p = s.players.find((x) => x.player_id === pid);
            return (
              <tr key={pid}>
                <td style={{ paddingLeft: 0 }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 5,
                      marginRight: 8,
                      background: PLAYER_COLORS[s.players.findIndex((x) => x.player_id === pid) % PLAYER_COLORS.length],
                    }}
                  />
                  <strong>{p?.name ?? `Player ${i + 1}`}</strong>
                </td>
                {columns.map((set, k) => (
                  <td
                    key={k}
                    className="num"
                    style={{
                      fontSize: 20,
                      fontWeight: set[i] > set[1 - i] ? 700 : 400,
                      color: unfinished && k === columns.length - 1 ? "var(--muted)" : undefined,
                    }}
                  >
                    {set[i]}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      {unfinished && <div className="small muted">The last set was still being played when the video ended.</div>}
    </div>
  );
}

function Kpi({
  label,
  value,
  unit,
  sub,
  digits = 0,
}: {
  label: string;
  value: number | undefined | null;
  unit?: string;
  sub?: string;
  digits?: number;
}) {
  return (
    <div>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={{ fontSize: 20 }}>
        {value == null ? "—" : value.toFixed(digits)}
        {value != null && unit && <span className="small muted"> {unit}</span>}
      </div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function ShotTable({
  s,
  shots,
  playAt,
}: {
  s: SessionDetail;
  shots: Shot[];
  playAt: (t: number, videoId?: number) => void;
}) {
  const [who, setWho] = useState("all");
  const [stroke, setStroke] = useState("all");
  const [selected, setSelected] = useState<number>();
  const names = Object.fromEntries(s.players.map((p) => [p.player_id, p.name]));
  const rows = shots.filter(
    (x) => (who === "all" || String(x.player_id) === who) && (stroke === "all" || x.stroke === stroke),
  );
  if (!shots.length) return <div className="card empty">No shots found yet.</div>;
  const f = (v: number | null, d = 0) => (v == null ? "—" : v.toFixed(d));
  return (
    <div className="stack">
      <div className="row">
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
          {[...new Set(shots.map((x) => x.stroke))].map((k) => (
            <option key={k} value={k}>
              {STROKE_LABELS[k] ?? k}
            </option>
          ))}
        </select>
        <span className="muted small">{rows.length} shots · click one to watch it</span>
      </div>
      <div className="card table-wrap" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Player</th>
              <th>Stroke</th>
              <th className="num">Speed km/h</th>
              <th className="num">Over net m</th>
              <th>Depth</th>
              <th>Direction</th>
              <th>Result</th>
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
                <td>{x.player_id != null ? names[x.player_id] ?? "?" : "?"}</td>
                <td>
                  {STROKE_LABELS[x.stroke] ?? x.stroke}
                  {x.spin && x.spin !== "flat" ? <span className="muted small"> · {x.spin}</span> : null}
                </td>
                <td className="num">{f(x.speed_kmh)}</td>
                <td className="num">{f(x.net_clearance_m, 2)}</td>
                <td>{x.depth ?? "—"}</td>
                <td>{x.direction ?? "—"}</td>
                <td>{x.outcome ?? (x.in_court == null ? "—" : x.in_court ? "in" : "out")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const ENDINGS: Record<string, string> = {
  winner: "winner",
  ace: "ace",
  out: "out",
  net: "into the net",
  double_fault: "double fault",
  fault: "first serve fault",
  not_a_point: "between points",
  unknown: "—",
};

function Points({
  s,
  rallies,
  playAt,
}: {
  s: SessionDetail;
  rallies: Rally[];
  playAt: (t: number) => void;
}) {
  const [all, setAll] = useState(false);
  const names = Object.fromEntries(s.players.map((p) => [p.player_id, p.name]));
  const shown = rallies.filter((r) => all || (r.end_reason !== "not_a_point" && r.end_reason !== "fault"));
  if (!rallies.length) return <div className="card empty">No points found yet.</div>;
  let point = 0;
  return (
    <div className="stack">
      <label className="row small muted" style={{ gap: 6 }}>
        <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
        Also show faults and balls hit between points
      </label>
      <div className="card table-wrap" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>Point</th>
              <th>Time</th>
              <th>Score before</th>
              <th>Server</th>
              <th className="num">Shots</th>
              <th>Won by</th>
              <th>Ended</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const isPoint = r.score_before != null;
              if (isPoint) point += 1;
              return (
                <tr
                  key={r.id}
                  className="clickable"
                  onClick={() => playAt(r.start_s)}
                  style={isPoint ? undefined : { opacity: 0.55 }}
                >
                  <td>{isPoint ? point : ""}</td>
                  <td className="mono">{formatClock(r.start_s)}</td>
                  <td className="mono">{(r.score_before?.text as string) ?? "—"}</td>
                  <td>{r.server_id != null ? names[r.server_id] : "—"}</td>
                  <td className="num">{r.shot_count}</td>
                  <td>{r.winner_id != null ? names[r.winner_id] : "—"}</td>
                  <td>{ENDINGS[r.end_reason ?? "unknown"] ?? r.end_reason}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Videos({
  s,
  onChanged,
  playAt,
}: {
  s: SessionDetail;
  onChanged: () => void;
  playAt: (t: number, videoId?: number) => void;
}) {
  return (
    <div className="stack">
      <div className="card table-wrap" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>File</th>
              <th>Starts at</th>
              <th>Length</th>
              <th>Format</th>
              <th>Court</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {s.videos.map((v) => (
              <tr key={v.id}>
                <td>
                  {v.filename}
                  <div className="small muted">{formatBytes(v.size_bytes)}</div>
                </td>
                <td className="mono" title={v.sync_method ? `lined up by ${v.sync_method.replace("_", " ")}` : ""}>
                  {formatClock(v.offset_s)}
                </td>
                <td>{formatDuration(v.duration_s)}</td>
                <td className="small">
                  {v.width && v.height ? `${v.width}×${v.height}` : "—"}
                  {v.fps ? ` · ${Math.round(v.fps)} fps` : ""}
                </td>
                <td className="small">
                  {v.court_quality == null ? "—" : v.has_court ? `found (${pct(v.court_quality)})` : "not visible"}
                  {v.court_image_url && (
                    <>
                      {" · "}
                      <a href={v.court_image_url} target="_blank" rel="noreferrer">
                        view
                      </a>
                    </>
                  )}
                </td>
                <td className="small">{v.error ? <span style={{ color: "var(--danger)" }}>{v.error}</span> : v.status}</td>
                <td className="num">
                  <span className="row" style={{ justifyContent: "flex-end" }}>
                    {v.proxy_url && (
                      <button className="btn small" onClick={() => playAt(v.offset_s + (v.proxy_start_pts ?? 0) + 2, v.id)}>
                        Play
                      </button>
                    )}
                    {s.videos.length > 1 && (
                      <button
                        className="btn btn-ghost small btn-danger"
                        onClick={async () => {
                          if (!confirm(`Remove ${v.filename} from this session?`)) return;
                          await api.removeVideo(s.id, v.id);
                          onChanged();
                        }}
                      >
                        Remove
                      </button>
                    )}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div>
        <Link to="/upload" className="btn">
          Add a video to this session
        </Link>
      </div>
    </div>
  );
}
