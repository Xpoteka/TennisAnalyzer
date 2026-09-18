import { Link } from "react-router-dom";
import { api } from "../api";
import { Avatar } from "../components/Badges";
import { STROKE_LABELS, formatDate } from "../format";
import { useData } from "../hooks";

export default function PlayersPage() {
  const { data, error } = useData(() => api.players(), []);
  const players = [...(data ?? [])].sort(
    (a, b) => (b.last_seen ?? "").localeCompare(a.last_seen ?? "") || b.shot_count - a.shot_count,
  );
  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Players</h1>
          <p>
            Everyone recognised in your sessions. Players are matched across sessions
            automatically by build, handedness and appearance. Rename a profile, or merge two
            that are the same person.
          </p>
        </div>
      </div>
      {error && <div className="notice notice-error">{error}</div>}
      {data && data.length === 0 && (
        <div className="card empty">
          <h2>No players yet</h2>
          <p>Players appear here after a session has been analysed.</p>
        </div>
      )}
      <div className="grid">
        {players.map((p) => {
          const top = Object.entries(p.strokes)
            .filter(([k]) => k !== "unknown")
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
          return (
            <Link key={p.id} to={`/players/${p.id}`} className="card card-link stack" style={{ gap: 12 }}>
              <div className="row" style={{ gap: 14 }}>
                <Avatar name={p.name} url={p.thumbnail_url} />
                <div>
                  <h3>{p.name}</h3>
                  <div className="small muted">
                    {[
                      p.handedness && `${p.handedness}-handed`,
                      p.height_m && `${p.height_m.toFixed(2)} m`,
                    ]
                      .filter(Boolean)
                      .join(" · ") || "—"}
                  </div>
                </div>
              </div>
              <div className="row small" style={{ gap: 16 }}>
                <span>
                  <strong>{p.session_count}</strong> session{p.session_count === 1 ? "" : "s"}
                </span>
                <span>
                  <strong>{p.shot_count}</strong> shots
                </span>
                <span className="muted">last {formatDate(p.last_seen)}</span>
              </div>
              {top.length > 0 && (
                <div className="row small muted">
                  {top.map(([k, n]) => (
                    <span key={k} className="badge">
                      {STROKE_LABELS[k] ?? k} {n}
                    </span>
                  ))}
                </div>
              )}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
