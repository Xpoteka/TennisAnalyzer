import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import Sparkline from "../components/Sparkline";
import { Avatar, KindBadge } from "../components/Badges";
import { STROKE_LABELS, formatDate, sessionTitle } from "../format";
import { useData } from "../hooks";

export default function PlayerPage() {
  const id = Number(useParams().id);
  const navigate = useNavigate();
  const { data: p, error, reload } = useData(() => api.player(id), [id]);
  const others = useData(() => api.players(), []);
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState("");
  const [mergeInto, setMergeInto] = useState("");

  if (error && !p) return <div className="notice notice-error">{error}</div>;
  if (!p) return null;
  if (p.id !== id) navigate(`/players/${p.id}`, { replace: true }); // merged away

  const profile = p.profile ?? { by_stroke: {}, technique: {}, trend: [], insights: [] };
  const strokes = Object.entries(profile.by_stroke);
  const trend = profile.trend;

  return (
    <div className="stack" style={{ gap: 24 }}>
      <div className="page-head">
        <div className="row" style={{ gap: 18 }}>
          <Avatar name={p.name} url={p.thumbnail_url} large />
          <div>
            <div className="row small" style={{ marginBottom: 6 }}>
              <Link to="/players" className="muted">
                Players
              </Link>
              <span className="muted">/</span>
            </div>
            {renaming ? (
              <form
                className="row"
                onSubmit={async (e) => {
                  e.preventDefault();
                  await api.renamePlayer(p.id, name);
                  setRenaming(false);
                  reload();
                }}
              >
                <input className="input" autoFocus value={name} onChange={(e) => setName(e.target.value)} />
                <button className="btn btn-primary" disabled={!name.trim()}>
                  Save
                </button>
                <button type="button" className="btn btn-ghost" onClick={() => setRenaming(false)}>
                  Cancel
                </button>
              </form>
            ) : (
              <div className="row" style={{ gap: 12 }}>
                <h1>{p.name}</h1>
                <button
                  className="btn btn-ghost small"
                  onClick={() => {
                    setName(p.name);
                    setRenaming(true);
                  }}
                >
                  Rename
                </button>
              </div>
            )}
            <p>
              {[
                p.handedness && `${p.handedness}-handed`,
                p.height_m && `about ${p.height_m.toFixed(2)} m tall`,
                `first seen ${formatDate(p.created_at)}`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
          </div>
        </div>
        <form
          className="row"
          onSubmit={async (e) => {
            e.preventDefault();
            const target = others.data?.find((o) => String(o.id) === mergeInto);
            if (!target || !confirm(`Merge ${p.name} into ${target.name}? Their sessions and shots move to ${target.name}.`)) return;
            await api.mergePlayer(p.id, target.id);
            navigate(`/players/${target.id}`);
          }}
        >
          <select className="select" value={mergeInto} onChange={(e) => setMergeInto(e.target.value)}>
            <option value="">Same person as…</option>
            {others.data
              ?.filter((o) => o.id !== p.id)
              .map((o) => (
                <option key={o.id} value={o.id}>
                  {o.name}
                </option>
              ))}
          </select>
          <button className="btn" disabled={!mergeInto}>
            Merge
          </button>
        </form>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="kpi-label">Sessions</div>
          <div className="kpi-value">{p.session_count}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Shots</div>
          <div className="kpi-value">{p.shot_count}</div>
        </div>
        {Object.entries(p.strokes)
          .filter(([k]) => k !== "unknown")
          .sort((a, b) => b[1] - a[1])
          .map(([k, n]) => (
            <div className="kpi" key={k}>
              <div className="kpi-label">{STROKE_LABELS[k] ?? k}</div>
              <div className="kpi-value">{n}</div>
              <div className="kpi-sub">{p.shot_count ? Math.round((n / p.shot_count) * 100) : 0}% of shots</div>
            </div>
          ))}
      </div>

      {strokes.length > 0 && (
        <section className="stack">
          <h2>Strokes</h2>
          <div className="card table-wrap" style={{ padding: 0 }}>
            <table>
              <thead>
                <tr>
                  <th>Stroke</th>
                  <th className="num">Shots</th>
                  <th className="num">Speed km/h</th>
                  <th className="num">Top</th>
                  <th className="num">In</th>
                  <th className="num">Over net m</th>
                  <th className="num">Contact height m</th>
                  <th className="num">Knee angle</th>
                </tr>
              </thead>
              <tbody>
                {strokes.map(([k, v]) => (
                  <tr key={k}>
                    <td>{STROKE_LABELS[k] ?? k}</td>
                    <td className="num">{v.count}</td>
                    <td className="num">{v.speed_avg ?? "—"}</td>
                    <td className="num">{v.speed_max ?? "—"}</td>
                    <td className="num">{v.in_pct != null ? `${Math.round(v.in_pct * 100)}%` : "—"}</td>
                    <td className="num">{v.net_clearance_avg ?? "—"}</td>
                    <td className="num">{v.contact_height_m ?? "—"}</td>
                    <td className="num">
                      {v.technique.knee_bend_deg != null ? `${Math.round(v.technique.knee_bend_deg)}°` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small muted">
            Speed, height over the net and in/out only count shots whose ball flight could be measured.
          </div>
        </section>
      )}

      <section className="stack">
        <h2>Technique</h2>
        {profile.insights.length ? (
          <div className="card stack" style={{ gap: 8 }}>
            {profile.insights.map((i, n) => (
              <div key={n} className="row" style={{ alignItems: "flex-start", gap: 10, flexWrap: "nowrap" }}>
                <span className={`badge ${i.level === "good" ? "badge-training" : i.level === "work" ? "badge-busy" : ""}`}>
                  {i.level === "good" ? "Strength" : i.level === "work" ? "Work on" : "Note"}
                </span>
                <span>{i.text}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="card muted">
            The technique analysis needs analysed shots with the player clearly visible. It appears
            here after a session with this player has been analysed.
          </div>
        )}
        <div className="kpis">
          {Object.entries(profile.technique)
            .filter(([, t]) => t.median != null)
            .map(([name, t]) => (
              <div className="kpi" key={name}>
                <div className="kpi-label">{t.label}</div>
                <div className="kpi-value" style={{ fontSize: 20 }}>
                  {name === "split_step"
                    ? `${Math.round((t.median ?? 0) * 100)}%`
                    : name === "shoulder_turn"
                      ? (t.median ?? 0).toFixed(2)
                      : `${(t.median ?? 0).toFixed(name.endsWith("_deg") ? 0 : 1)}${t.unit === "°" ? "°" : ""}`}
                  {t.unit && t.unit !== "°" && <span className="small muted"> {t.unit}</span>}
                </div>
                <div className="kpi-sub">from {t.n} shots</div>
              </div>
            ))}
        </div>
      </section>

      {trend.length > 0 && (
        <section className="stack">
          <h2>Over time</h2>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
            {(
              [
                ["Forehand speed (km/h)", trend.map((t) => t.forehand_kmh), (v: number) => v.toFixed(0)],
                ["Backhand speed (km/h)", trend.map((t) => t.backhand_kmh), (v: number) => v.toFixed(0)],
                ["Serve speed (km/h)", trend.map((t) => t.serve_kmh), (v: number) => v.toFixed(0)],
                ["Balls in", trend.map((t) => t.in_pct), (v: number) => `${Math.round(v * 100)}%`],
                ["Knee angle at the load", trend.map((t) => t.knee_bend_deg), (v: number) => `${v.toFixed(0)}°`],
                ["Split step", trend.map((t) => t.split_step_pct), (v: number) => `${Math.round(v * 100)}%`],
              ] as [string, (number | null)[], (v: number) => string][]
            ).map(([label, values, format]) => (
              <div className="card stack" key={label} style={{ gap: 6 }}>
                <div className="kpi-label">{label}</div>
                <Sparkline values={values} format={format} />
              </div>
            ))}
          </div>
          {trend.length === 1 && (
            <div className="small muted">One session so far: the trend lines grow with every session.</div>
          )}
        </section>
      )}

      <section className="stack">
        <h2>Sessions</h2>
        <div className="card table-wrap" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Session</th>
                <th>Date</th>
                <th>Type</th>
                <th className="num">Shots</th>
              </tr>
            </thead>
            <tbody>
              {p.sessions
                .slice()
                .reverse()
                .map((s) => (
                  <tr key={s.session_id} className="clickable" onClick={() => navigate(`/sessions/${s.session_id}`)}>
                    <td>{sessionTitle({ name: s.name, recorded_at: s.recorded_at, created_at: s.recorded_at ?? "" })}</td>
                    <td>{formatDate(s.recorded_at)}</td>
                    <td>
                      <KindBadge s={{ kind: s.kind, kind_confidence: null, kind_source: "auto" }} />
                    </td>
                    <td className="num">{(s.stats.shots as number | undefined) ?? "—"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
