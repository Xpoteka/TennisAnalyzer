// The video with everything the analysis knows drawn over it: skeletons, where the players
// ran, the ball, bounces and the court. Beside it, the point being played on a small court;
// under it, a timeline of every point and shot, and buttons to step through points and errors.

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type Overlay, type Rally, type SessionDetail, type Shot, type Video } from "../api";
import { STROKE_LABELS, formatClock } from "../format";
import { useData } from "../hooks";
import { ERROR_LABELS, TrackCache, missSide, rallyAt, type ShotIndex } from "../review";
import Court, { type CourtDot, type CourtLine, type CourtMark, type CourtTrail } from "./Court";

// COCO-17 limb segments, as tennis/pose_backends/base.py draws them.
const SKELETON: [number, number][] = [
  [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 5], [0, 6],
]; // prettier-ignore

const LAYERS = {
  skeleton: "Skeletons",
  trail: "Running paths",
  ball: "Ball",
  bounce: "Bounces",
  labels: "Shot labels",
  court: "Court lines",
} as const;
type Layer = keyof typeof LAYERS;
type Layers = Record<Layer, boolean>;
const DEFAULT_LAYERS: Layers = {
  skeleton: true,
  trail: false,
  ball: true,
  bounce: true,
  labels: true,
  court: false,
};

const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
const TRAIL_S = 4; // how far back a running path goes
const BALL_S = 0.7;
const IN = "#7be07b";
const OUT = "#ff6b5e";

const TECHNIQUE: [string, string, (v: number) => string][] = [
  ["knee_bend_deg", "Knee angle", (v) => `${Math.round(v)}°`],
  ["elbow_contact_deg", "Elbow at contact", (v) => `${Math.round(v)}°`],
  ["shoulder_turn", "Shoulder turn", (v) => v.toFixed(2)],
  ["racket_arm_speed", "Arm speed", (v) => `${v.toFixed(1)} heights/s`],
  ["split_step", "Split step", (v) => (v >= 0.5 ? "yes" : "no")],
  ["recovery_m", "Recovered to", (v) => `${v.toFixed(1)} m from the centre`],
];

function stored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // private window: the choice just does not survive a reload
  }
}

export default function ReviewPlayer({
  s,
  video,
  target,
  shots,
  rallies,
  index,
  colorOf,
  playAt,
  onClose,
}: {
  s: SessionDetail;
  video: Video;
  /** Where to start playing, in the proxy's own seconds. `n` changes with every request. */
  target: { t: number; n: number };
  shots: Shot[];
  rallies: Rally[];
  index: ShotIndex;
  colorOf: (playerId: number | null) => string;
  playAt: (t: number, videoId?: number, lead?: number) => void;
  onClose: () => void;
}) {
  const videoEl = useRef<HTMLVideoElement>(null);
  const canvasEl = useRef<HTMLCanvasElement>(null);
  const stageEl = useRef<HTMLDivElement>(null);
  const [layers, setLayers] = useState<Layers>(() => stored("review.layers", DEFAULT_LAYERS));
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(false);
  const [who, setWho] = useState("all");
  const [now, setNow] = useState(video.offset_s + (video.proxy_start_pts ?? 0) + target.t);
  const [length, setLength] = useState(video.duration_s ?? 0);
  const overlay = useData(() => api.overlay(video.id), [video.id]);

  // Session time = proxy time + `base`; PTS = proxy time + `startPts`.
  const startPts = video.proxy_start_pts ?? 0;
  const base = video.offset_s + startPts;

  const cache = useMemo(() => new TrackCache(video.id, () => undefined), [video.id]);
  const mine = useMemo(() => shots.filter((x) => x.video_id === video.id), [shots, video.id]);
  const points = useMemo(
    () => rallies.filter((r) => r.end_reason !== "not_a_point" && r.end_reason !== "fault"),
    [rallies],
  );
  const errors = useMemo(
    () => shots.filter((x) => index.errorOf.has(x.id) && (who === "all" || String(x.player_id) === who)),
    [shots, index, who],
  );

  // Seek whenever the page asks for a moment, also within the video that is already loaded.
  useEffect(() => {
    const v = videoEl.current;
    if (!v) return;
    const go = () => {
      v.currentTime = target.t;
      v.play().catch(() => undefined); // autoplay may be refused; the controls still work
    };
    if (v.readyState >= 1) go();
    else v.addEventListener("loadedmetadata", go, { once: true });
    return () => v.removeEventListener("loadedmetadata", go);
  }, [target, video.id]);

  useEffect(() => {
    if (videoEl.current) videoEl.current.playbackRate = speed;
  }, [speed, video.id]);

  // Everything the drawing loop reads, kept in a ref so the loop itself never restarts.
  const live = useRef({ layers, loop, overlay: overlay.data, mine, rallies, colorOf });
  live.current = { layers, loop, overlay: overlay.data, mine, rallies, colorOf };
  const looped = useRef<Rally | undefined>(undefined);

  useEffect(() => {
    let frame = 0;
    let lastNow = -1;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const v = videoEl.current;
      const c = canvasEl.current;
      if (!v || !c) return;
      const t = v.currentTime + base;
      if (Math.abs(t - lastNow) >= 0.12) {
        lastNow = t;
        setNow(t);
      }
      const st = live.current;
      const rally = rallyAt(st.rallies, t);
      if (rally) looped.current = rally;
      const last = looped.current;
      // Past the end of the point being looped (but not after a jump elsewhere): once more.
      if (st.loop && last && !v.paused && t > last.end_s + 1.5 && t < last.end_s + 2.5) {
        v.currentTime = Math.max(0, last.start_s - 1.5 - base);
      }
      cache.want(v.currentTime + startPts);
      draw(c, v, v.currentTime + startPts, t, cache, st);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [base, startPts, cache]);

  const toggle = (k: Layer) =>
    setLayers((old) => {
      const next = { ...old, [k]: !old[k] };
      store("review.layers", next);
      return next;
    });

  const step = useCallback(
    (frames: number) => {
      const v = videoEl.current;
      if (!v) return;
      v.pause();
      v.currentTime = Math.max(0, v.currentTime + frames / (video.fps || 30));
    },
    [video.fps],
  );

  const jump = useCallback(
    (kind: "point" | "error", dir: 1 | -1) => {
      const t = (videoEl.current?.currentTime ?? 0) + base;
      if (kind === "point") {
        const r =
          dir > 0
            ? points.find((p) => p.start_s > t + 2.5)
            : [...points].reverse().find((p) => p.start_s < t - 3);
        if (r) playAt(r.start_s);
      } else {
        const e =
          dir > 0 ? errors.find((x) => x.t > t + 2.5) : [...errors].reverse().find((x) => x.t < t - 1);
        if (e) playAt(e.t, e.video_id);
      }
    },
    [base, points, errors, playAt],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && /^(INPUT|SELECT|TEXTAREA|BUTTON|VIDEO)$/.test(el.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const v = videoEl.current;
      if (!v) return;
      const actions: Record<string, () => void> = {
        " ": () => (v.paused ? void v.play() : v.pause()),
        ArrowLeft: () => (v.currentTime = Math.max(0, v.currentTime - 2)),
        ArrowRight: () => (v.currentTime += 2),
        ",": () => step(-1),
        ".": () => step(1),
        n: () => jump("point", 1),
        p: () => jump("point", -1),
        e: () => jump("error", 1),
        E: () => jump("error", -1),
        l: () => setLoop((x) => !x),
        s: () => setSpeed((x) => (x === 1 ? 0.25 : 1)),
      };
      const act = actions[e.key];
      if (!act) return;
      e.preventDefault();
      act();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, jump]);

  const rally = rallyAt(rallies, now);
  const rallyShots = rally ? (index.shotsOf.get(rally.id) ?? []) : [];
  const current = [...(rally ? rallyShots : mine)].reverse().find((x) => x.t <= now + 0.1 && x.t > now - 4);
  const names = Object.fromEntries(s.players.map((p) => [p.player_id, p.name]));

  return (
    <div className="review">
      <div className="review-main">
        <div className="review-stage" ref={stageEl}>
          <video
            ref={videoEl}
            key={video.id}
            src={video.proxy_url ?? undefined}
            controls
            controlsList="nofullscreen nodownload"
            disablePictureInPicture
            playsInline
            onDurationChange={(e) => setLength(e.currentTarget.duration || length)}
          />
          <canvas ref={canvasEl} />
        </div>
        <Timeline
          from={base}
          length={length}
          now={now}
          rallies={rallies}
          shots={mine}
          index={index}
          colorOf={colorOf}
          onSeek={(t) => playAt(t, video.id, 0)}
        />
        <div className="review-bar">
          <span className="row" style={{ gap: 4 }}>
            <button className="btn small" title="Previous point (P)" onClick={() => jump("point", -1)}>
              ‹ Point
            </button>
            <button className="btn small" title="Next point (N)" onClick={() => jump("point", 1)}>
              Point ›
            </button>
          </span>
          <span className="row" style={{ gap: 4 }}>
            <button
              className="btn small btn-error"
              title="Previous error (Shift+E)"
              disabled={!errors.length}
              onClick={() => jump("error", -1)}
            >
              ‹ Error
            </button>
            <button
              className="btn small btn-error"
              title="Next error (E)"
              disabled={!errors.length}
              onClick={() => jump("error", 1)}
            >
              Error ›
            </button>
            <select className="select small" value={who} onChange={(e) => setWho(e.target.value)} title="Whose errors to step through">
              <option value="all">everyone's</option>
              {s.players.map((p) => (
                <option key={p.player_id} value={p.player_id}>
                  {p.name}'s
                </option>
              ))}
            </select>
          </span>
          <span className="row" style={{ gap: 4 }}>
            <button className="btn small" title="One frame back (,)" onClick={() => step(-1)}>
              −1
            </button>
            <button className="btn small" title="One frame on (.)" onClick={() => step(1)}>
              +1
            </button>
            <select
              className="select small"
              value={speed}
              title="Playback speed (S switches between slow motion and normal)"
              onChange={(e) => setSpeed(Number(e.target.value))}
            >
              {SPEEDS.map((x) => (
                <option key={x} value={x}>
                  {x}×
                </option>
              ))}
            </select>
          </span>
          <label className="row small" style={{ gap: 4 }} title="Play the current point again and again (L)">
            <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} />
            Loop point
          </label>
          <span style={{ flex: 1 }} />
          <button className="btn small btn-ghost" onClick={() => stageEl.current?.requestFullscreen?.()}>
            Full screen
          </button>
          <button className="btn small btn-ghost" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="review-layers">
          {(Object.keys(LAYERS) as Layer[]).map((k) => (
            <label key={k} className={`chip${layers[k] ? " on" : ""}`}>
              <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} />
              {LAYERS[k]}
            </label>
          ))}
          {overlay.data && !overlay.data.court.length && (
            <span className="small muted">No court in this video: bounces and court lines cannot be drawn.</span>
          )}
        </div>
      </div>
      <div className="review-side">
        <MiniCourt
          cache={cache}
          pts={now - video.offset_s}
          overlay={overlay.data}
          s={s}
          shots={rallyShots}
          current={current}
          colorOf={colorOf}
          onShot={(x) => playAt(x.t, x.video_id, 1)}
        />
        <div className="small muted" style={{ textAlign: "center" }}>
          {rally
            ? `${rally.score_before?.text ? `${rally.score_before.text as string} · ` : ""}${rallyShots.length} shot${rallyShots.length === 1 ? "" : "s"} in this ${points.includes(rally) ? "point" : "rally"}`
            : "Between points"}
        </div>
        {rallyShots.length > 0 && (
          <ol className="review-shots">
            {rallyShots.map((x) => {
              const err = index.errorOf.get(x.id);
              return (
                <li
                  key={x.id}
                  className={x.id === current?.id ? "current" : ""}
                  onClick={() => playAt(x.t, x.video_id, 1)}
                >
                  <span className="dot" style={{ background: colorOf(x.player_id) }} />
                  <span>{STROKE_LABELS[x.stroke] ?? x.stroke}</span>
                  <span className="muted">{x.speed_kmh ? `${Math.round(x.speed_kmh)} km/h` : ""}</span>
                  {err && <span className="tag-error">{missSide(x) ?? ERROR_LABELS[err]}</span>}
                </li>
              );
            })}
          </ol>
        )}
        {current && (
          <div className="review-shot">
            <strong>
              {(current.player_id != null && names[current.player_id]) || "?"} ·{" "}
              {STROKE_LABELS[current.stroke] ?? current.stroke}
              {current.spin && current.spin !== "flat" ? `, ${current.spin}` : ""}
            </strong>
            <dl>
              {current.net_clearance_m != null && (
                <>
                  <dt>Over the net</dt>
                  <dd>{current.net_clearance_m.toFixed(2)} m</dd>
                </>
              )}
              {(current.depth || current.direction) && (
                <>
                  <dt>Placement</dt>
                  <dd>{[current.depth, current.direction].filter(Boolean).join(", ")}</dd>
                </>
              )}
              {TECHNIQUE.map(([key, label, show]) =>
                current.metrics[key] == null ? null : (
                  <span key={key} style={{ display: "contents" }}>
                    <dt>{label}</dt>
                    <dd>{show(current.metrics[key])}</dd>
                  </span>
                ),
              )}
            </dl>
          </div>
        )}
      </div>
    </div>
  );
}

// --- the small court beside the video ------------------------------------------------------

function MiniCourt({
  cache,
  pts,
  overlay,
  s,
  shots,
  current,
  colorOf,
  onShot,
}: {
  cache: TrackCache;
  pts: number;
  overlay: Overlay | undefined;
  s: SessionDetail;
  shots: Shot[];
  current: Shot | undefined;
  colorOf: (playerId: number | null) => string;
  onShot: (shot: Shot) => void;
}) {
  const byTrack = new Map<number, [number, number][]>();
  for (const [, track, , , cx, cy] of cache.rows("feet", pts - TRAIL_S, pts + 0.05)) {
    if (cx == null || cy == null) continue;
    if (!byTrack.has(track)) byTrack.set(track, []);
    byTrack.get(track)!.push([cx, cy]);
  }
  const trails: CourtTrail[] = [];
  const marks: CourtMark[] = [];
  for (const [track, points] of byTrack) {
    const playerId = overlay?.players[String(track)] ?? null;
    if (playerId == null) continue;
    const [x, y] = points[points.length - 1];
    trails.push({ points, color: colorOf(playerId) });
    marks.push({
      x,
      y,
      color: colorOf(playerId),
      label: s.players.find((p) => p.player_id === playerId)?.label,
    });
  }
  const drawn = shots.filter((x) => x.hit_x != null && x.hit_y != null && x.bounce_x != null && x.bounce_y != null);
  const lines: CourtLine[] = drawn.map((x) => ({
    from: [x.hit_x!, x.hit_y!],
    to: [x.bounce_x!, x.bounce_y!],
    color: colorOf(x.player_id),
    faint: x.id !== current?.id,
  }));
  const dots: CourtDot[] = drawn.map((x) => ({
    x: x.bounce_x!,
    y: x.bounce_y!,
    color: x.in_court === false ? OUT : colorOf(x.player_id),
    hollow: x.in_court === false,
    onClick: () => onShot(x),
  }));
  return <Court height={300} dots={dots} lines={lines} marks={marks} trails={trails} />;
}

// --- the timeline under the video ------------------------------------------------------------

function Timeline({
  from,
  length,
  now,
  rallies,
  shots,
  index,
  colorOf,
  onSeek,
}: {
  from: number;
  length: number;
  now: number;
  rallies: Rally[];
  shots: Shot[];
  index: ShotIndex;
  colorOf: (playerId: number | null) => string;
  onSeek: (t: number) => void;
}) {
  if (!(length > 0)) return <div className="timeline" />;
  const left = (t: number) => `${Math.min(100, Math.max(0, ((t - from) / length) * 100))}%`;
  return (
    <div
      className="timeline"
      title="Points, shots (coloured by player) and errors (red). Click to go there."
      onClick={(e) => {
        const box = e.currentTarget.getBoundingClientRect();
        onSeek(from + ((e.clientX - box.left) / box.width) * length);
      }}
    >
      <TimelineMarks from={from} length={length} rallies={rallies} shots={shots} index={index} colorOf={colorOf} />
      <div className="timeline-head" style={{ left: left(now) }} />
      <span className="timeline-clock mono">{formatClock(now)}</span>
    </div>
  );
}

// Hundreds of marks that only change with the data, so the moving playhead does not redraw them.
const TimelineMarks = memo(function TimelineMarks({
  from,
  length,
  rallies,
  shots,
  index,
  colorOf,
}: {
  from: number;
  length: number;
  rallies: Rally[];
  shots: Shot[];
  index: ShotIndex;
  colorOf: (playerId: number | null) => string;
}) {
  const pos = (t: number) => ((t - from) / length) * 100;
  return (
    <>
      {rallies
        .filter((r) => r.end_s >= from && r.start_s <= from + length)
        .map((r) => (
          <div
            key={`r${r.id}`}
            className={`timeline-rally${r.end_reason === "not_a_point" ? " idle" : ""}`}
            style={{ left: `${pos(r.start_s)}%`, width: `${Math.max(0.15, pos(r.end_s) - pos(r.start_s))}%` }}
          />
        ))}
      {shots.map((x) => {
        const err = index.errorOf.has(x.id);
        return (
          <div
            key={x.id}
            className={`timeline-shot${err ? " error" : ""}`}
            style={{ left: `${pos(x.t)}%`, background: err ? undefined : colorOf(x.player_id) }}
          />
        );
      })}
    </>
  );
});

// --- drawing over the video --------------------------------------------------------------------

type Live = {
  layers: Layers;
  overlay: Overlay | undefined;
  mine: Shot[];
  colorOf: (playerId: number | null) => string;
};

function draw(c: HTMLCanvasElement, v: HTMLVideoElement, pts: number, now: number, cache: TrackCache, st: Live) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(c.clientWidth * dpr);
  const H = Math.round(c.clientHeight * dpr);
  if (c.width !== W || c.height !== H) {
    c.width = W;
    c.height = H;
  }
  const g = c.getContext("2d");
  if (!g) return;
  g.clearRect(0, 0, W, H);
  const srcW = st.overlay?.width || v.videoWidth;
  const srcH = st.overlay?.height || v.videoHeight;
  if (!srcW || !srcH || !W) return;
  // The picture is fitted inside the element (bars on two sides when the shapes differ).
  const k = Math.min(W / srcW, H / srcH);
  const ox = (W - srcW * k) / 2;
  const oy = (H - srcH * k) / 2;
  const X = (x: number) => ox + x * k;
  const Y = (y: number) => oy + y * k;
  const u = Math.max(1, (W / 960) * 1.4); // line widths and text grow with the picture
  const color = (track: number) => st.colorOf(st.overlay?.players[String(track)] ?? null);
  g.lineCap = "round";
  g.lineJoin = "round";

  if (st.layers.court && st.overlay) {
    g.strokeStyle = "rgba(255,255,255,0.55)";
    g.lineWidth = u;
    for (const line of st.overlay.court) {
      g.beginPath();
      line.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))));
      g.stroke();
    }
  }

  if (st.layers.trail) {
    const paths = new Map<number, [number, number, number][]>();
    for (const [t, track, px, py] of cache.rows("feet", pts - TRAIL_S, pts + 0.05)) {
      if (!paths.has(track)) paths.set(track, []);
      paths.get(track)!.push([t, px, py]);
    }
    for (const [track, path] of paths) {
      g.strokeStyle = color(track);
      g.lineWidth = 2.5 * u;
      for (let i = 1; i < path.length; i++) {
        if (path[i][0] - path[i - 1][0] > 0.5) continue; // the tracker lost them in between
        g.globalAlpha = 0.15 + 0.7 * (1 - (pts - path[i][0]) / TRAIL_S);
        g.beginPath();
        g.moveTo(X(path[i - 1][1]), Y(path[i - 1][2]));
        g.lineTo(X(path[i][1]), Y(path[i][2]));
        g.stroke();
      }
      g.globalAlpha = 1;
    }
  }

  // The pose nearest to this frame, per track. Also tells the labels where each player is.
  const poses = new Map<number, { x: (number | null)[]; y: (number | null)[]; d: number }>();
  for (const [t, track, x, y] of cache.rows("poses", pts - 0.08, pts + 0.08)) {
    const d = Math.abs(t - pts);
    if (!poses.has(track) || d < poses.get(track)!.d) poses.set(track, { x, y, d });
  }
  if (st.layers.skeleton) {
    for (const [track, p] of poses) {
      g.strokeStyle = color(track);
      g.fillStyle = color(track);
      g.lineWidth = 2 * u;
      for (const [a, b] of SKELETON) {
        const [xa, ya, xb, yb] = [p.x[a], p.y[a], p.x[b], p.y[b]];
        if (xa == null || ya == null || xb == null || yb == null) continue;
        g.beginPath();
        g.moveTo(X(xa), Y(ya));
        g.lineTo(X(xb), Y(yb));
        g.stroke();
      }
      for (const i of [9, 10]) {
        const [x, y] = [p.x[i], p.y[i]]; // the wrists: where the racket is
        if (x == null || y == null) continue;
        g.beginPath();
        g.arc(X(x), Y(y), 3 * u, 0, 2 * Math.PI);
        g.fill();
      }
    }
  }

  if (st.layers.ball) {
    const ball = cache.rows("ball", pts - BALL_S, pts + 0.02);
    g.strokeStyle = "#e8ff4a";
    g.lineWidth = 2.5 * u;
    for (let i = 1; i < ball.length; i++) {
      const [t0, x0, y0, tr0] = ball[i - 1];
      const [t1, x1, y1, tr1] = ball[i];
      if (tr0 !== tr1 || t1 - t0 > 0.15) continue;
      g.globalAlpha = 0.1 + 0.85 * (1 - (pts - t1) / BALL_S);
      g.beginPath();
      g.moveTo(X(x0), Y(y0));
      g.lineTo(X(x1), Y(y1));
      g.stroke();
    }
    g.globalAlpha = 1;
    const last = ball[ball.length - 1];
    if (last && pts - last[0] < 0.08) {
      g.beginPath();
      g.arc(X(last[1]), Y(last[2]), 7 * u, 0, 2 * Math.PI);
      g.stroke();
    }
  }

  for (const shot of st.mine) {
    const age = now - shot.t;
    if (age < -0.1 || age > 3) continue;
    const spot = st.overlay?.bounces[String(shot.id)];
    if (st.layers.bounce && spot && age > 0.4) {
      const out = shot.in_court === false;
      g.strokeStyle = out ? OUT : IN;
      g.lineWidth = 2.5 * u;
      g.globalAlpha = Math.min(1, (3 - age) / 0.6);
      g.beginPath();
      g.ellipse(X(spot[0]), Y(spot[1]), 11 * u, 5 * u, 0, 0, 2 * Math.PI);
      g.stroke();
      if (out) label(g, "OUT", X(spot[0]), Y(spot[1]) - 12 * u, u, OUT);
      g.globalAlpha = 1;
    }
    if (st.layers.labels && age <= 1.6) {
      const text = [
        STROKE_LABELS[shot.stroke] ?? shot.stroke,
        shot.speed_kmh ? `${Math.round(shot.speed_kmh)} km/h` : null,
        shot.outcome === "net" ? "into the net" : null,
      ]
        .filter(Boolean)
        .join(" · ");
      const head = [...poses].find(([track]) => st.overlay?.players[String(track)] === shot.player_id)?.[1];
      const ys = head?.y.filter((y): y is number => y != null) ?? [];
      const xs = head?.x.filter((x): x is number => x != null) ?? [];
      if (xs.length && ys.length) {
        const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
        label(g, text, X(cx), Y(Math.min(...ys)) - 14 * u, u, st.colorOf(shot.player_id));
      } else {
        label(g, text, ox + 90 * u, oy + 24 * u, u, st.colorOf(shot.player_id));
      }
    }
  }
}

function label(g: CanvasRenderingContext2D, text: string, x: number, y: number, u: number, color: string) {
  g.font = `600 ${Math.round(12 * u)}px ui-sans-serif, system-ui, sans-serif`;
  g.textAlign = "center";
  g.textBaseline = "middle";
  const w = g.measureText(text).width + 12 * u;
  const h = 18 * u;
  g.fillStyle = "rgba(12,14,9,0.78)";
  g.beginPath();
  g.roundRect(x - w / 2, y - h / 2, w, h, 4 * u);
  g.fill();
  g.fillStyle = color;
  g.fillText(text, x, y);
}
