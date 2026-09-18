import { Link } from "react-router-dom";
import { api, type SessionSummary } from "../api";
import { Avatar, KindBadge, StatusBadge } from "../components/Badges";
import { formatDate, formatDuration, sessionTitle } from "../format";
import { useData } from "../hooks";

const busy = (list?: SessionSummary[]) =>
  !!list?.some((s) => s.status === "queued" || s.status === "processing");

export default function SessionsPage() {
  const { data, error } = useData(() => api.sessions(), [], 3000, busy);

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Sessions</h1>
          <p>Every analysed training and match, newest first.</p>
        </div>
        <Link to="/upload" className="btn btn-primary">
          Upload a session
        </Link>
      </div>
      {error && <div className="notice notice-error">{error}</div>}
      {data && data.length === 0 && (
        <div className="card empty">
          <h2>No sessions yet</h2>
          <p>Upload a video of a training or a match to get started.</p>
          <Link to="/upload" className="btn btn-primary">
            Upload a session
          </Link>
        </div>
      )}
      <div className="grid">
        {data?.map((s) => (
          <Link key={s.id} to={`/sessions/${s.id}`} className="card card-link stack" style={{ gap: 10 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3>{sessionTitle(s)}</h3>
              <div className="row">
                <KindBadge s={s} />
                <StatusBadge s={s} />
              </div>
            </div>
            <div className="small muted">
              {formatDate(s.recorded_at ?? s.created_at)} · {formatDuration(s.duration_s)}
              {s.video_count > 1 ? ` · ${s.video_count} videos` : ""}
            </div>
            <Headline s={s} />
            {s.players.length > 0 && (
              <div className="row" style={{ gap: 12 }}>
                {s.players.map((p) => (
                  <div key={p.player_id} className="row small" style={{ gap: 6 }}>
                    <Avatar name={p.name} url={p.thumbnail_url} />
                    <span>{p.name}</span>
                  </div>
                ))}
              </div>
            )}
          </Link>
        ))}
      </div>
    </div>
  );
}

function Headline({ s }: { s: SessionSummary }) {
  const h = s.headline as { score?: string; shots?: number; rallies?: number };
  if (s.status === "failed") return <div className="small notice notice-error">{s.error}</div>;
  if (!h.score && h.shots == null) return null;
  return (
    <div className="row small" style={{ gap: 16 }}>
      {h.score && <strong style={{ fontSize: 16 }}>{h.score}</strong>}
      {h.shots != null && <span>{h.shots} shots</span>}
      {h.rallies != null && <span>{h.rallies} rallies</span>}
    </div>
  );
}
