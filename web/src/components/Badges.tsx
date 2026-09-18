import type { SessionSummary } from "../api";
import { pct } from "../format";

export function KindBadge({ s }: { s: Pick<SessionSummary, "kind" | "kind_confidence" | "kind_source"> }) {
  if (s.kind === "unknown") return null;
  const title =
    s.kind_source === "manual"
      ? "set by hand"
      : s.kind_confidence != null
        ? `detected automatically (${pct(s.kind_confidence)} sure)`
        : "detected automatically";
  return (
    <span className={`badge badge-${s.kind}`} title={title}>
      {s.kind === "match" ? "Match" : "Training"}
    </span>
  );
}

export function StatusBadge({ s }: { s: Pick<SessionSummary, "status" | "job"> }) {
  if (s.status === "ready") return null;
  if (s.status === "failed") return <span className="badge badge-failed">Failed</span>;
  if (s.status === "processing" || s.job?.status === "running")
    return <span className="badge badge-busy">Analysing {pct(s.job?.progress ?? 0)}</span>;
  if (s.status === "queued") return <span className="badge badge-busy">Queued</span>;
  return <span className="badge">Not analysed</span>;
}

export function Avatar({ name, url, large }: { name: string; url?: string | null; large?: boolean }) {
  const cls = `avatar${large ? " avatar-lg" : ""}`;
  if (url) return <img className={cls} src={url} alt="" />;
  return (
    <div className={cls} aria-hidden>
      {name.replace(/^Player\s*/, "").slice(0, 2).toUpperCase() || "?"}
    </div>
  );
}
