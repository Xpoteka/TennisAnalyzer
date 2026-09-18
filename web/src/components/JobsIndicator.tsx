import { Link } from "react-router-dom";
import { api } from "../api";
import { useData } from "../hooks";
import { pct } from "../format";

/** A small badge in the top bar while analyses are queued or running. */
export default function JobsIndicator() {
  const { data } = useData(() => api.jobs(true), [], 3000);
  if (!data || data.length === 0) return null;
  const running = data.find((j) => j.status === "running");
  const queued = data.filter((j) => j.status === "queued").length;
  return (
    <Link
      to={`/sessions/${(running ?? data[data.length - 1]).session_id}`}
      className="badge badge-busy"
      style={{ textDecoration: "none" }}
      title={running?.stage ?? "waiting"}
    >
      {running ? `Analysing ${pct(running.progress)}` : "Queued"}
      {queued > 0 && running ? ` · ${queued} waiting` : ""}
    </Link>
  );
}
