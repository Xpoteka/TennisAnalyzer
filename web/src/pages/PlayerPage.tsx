import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type PlayerDetail } from "../api";
import { Avatar, KindBadge } from "../components/Badges";
import { STROKE_LABELS, formatDate, sessionTitle } from "../format";
import { useData } from "../hooks";

type Technique = {
  insights?: { text: string; level: "good" | "info" | "work" }[];
  metrics?: Record<
    string,
    Record<string, { label: string; unit: string; median: number; spread: number; n: number; trend?: number }>
  >;
};

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

  const technique = (p as PlayerDetail & { technique?: Technique }).technique;

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

      <section className="stack">
        <h2>Technique</h2>
        {technique?.insights?.length ? (
          <div className="card stack" style={{ gap: 8 }}>
            {technique.insights.map((i, n) => (
              <div key={n} className="row" style={{ alignItems: "flex-start", gap: 10 }}>
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
      </section>

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
