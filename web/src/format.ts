import type { SessionSummary } from "./api";

// The interface is in English, so dates are too (a German weekday next to "afternoon" reads
// oddly). British order: day before month.
const LOCALE = "en-GB";

export function sessionTitle(s: Pick<SessionSummary, "name" | "recorded_at" | "created_at">) {
  if (s.name) return s.name;
  const d = new Date(s.recorded_at ?? s.created_at);
  const day = d.toLocaleDateString(LOCALE, { weekday: "short", day: "numeric", month: "short" });
  const h = d.getHours();
  const part = h < 12 ? "morning" : h < 17 ? "afternoon" : h < 21 ? "evening" : "night";
  return `${day}, ${part}`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(LOCALE, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function formatDuration(s: number | null | undefined): string {
  if (s == null) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h} h ${String(m).padStart(2, "0")} min`;
  if (m) return `${m} min`;
  return `${Math.round(s)} s`;
}

export function formatClock(s: number): string {
  const sign = s < 0 ? "-" : "";
  s = Math.abs(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = String(m).padStart(h ? 2 : 1, "0");
  return `${sign}${h ? `${h}:` : ""}${mm}:${String(sec).padStart(2, "0")}`;
}

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

export function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export const STROKE_LABELS: Record<string, string> = {
  serve: "Serve",
  forehand: "Forehand",
  backhand: "Backhand",
  volley_forehand: "FH volley",
  volley_backhand: "BH volley",
  overhead: "Overhead",
  unknown: "Other",
};
