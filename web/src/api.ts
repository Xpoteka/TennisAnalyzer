// Typed access to the JSON API. Every call goes to the same origin (/api/...).

export type Job = {
  id: number;
  session_id: number;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  stage: string | null;
  progress: number;
  message: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type Video = {
  id: number;
  filename: string;
  size_bytes: number;
  recorded_at: string | null;
  duration_s: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  offset_s: number;
  sync_method: string | null;
  court_quality: number | null;
  has_court: boolean;
  status: string;
  error: string | null;
  warnings: string[];
  proxy_url: string | null;
  proxy_start_pts: number | null;
  court_image_url: string | null;
};

export type SessionPlayer = {
  player_id: number;
  name: string;
  label: string;
  handedness: string | null;
  new_profile: boolean;
  stats: Record<string, unknown>;
  thumbnail_url: string | null;
};

export type SessionKind = "unknown" | "training" | "match";

export type SessionSummary = {
  id: number;
  name: string | null;
  created_at: string;
  recorded_at: string | null;
  kind: SessionKind;
  kind_confidence: number | null;
  kind_source: "auto" | "manual";
  status: "pending" | "queued" | "processing" | "ready" | "failed";
  error: string | null;
  duration_s: number | null;
  video_count: number;
  players: SessionPlayer[];
  job: Job | null;
  headline: Record<string, unknown>;
};

export type SessionDetail = SessionSummary & {
  videos: Video[];
  summary: Record<string, unknown>;
};

export type Shot = {
  id: number;
  video_id: number;
  rally_id: number | null;
  player_id: number | null;
  t: number;
  index_in_rally: number | null;
  stroke: string;
  spin: string | null;
  stroke_confidence: number | null;
  speed_kmh: number | null;
  avg_speed_kmh: number | null;
  net_clearance_m: number | null;
  apex_m: number | null;
  contact_height_m: number | null;
  hit_x: number | null;
  hit_y: number | null;
  bounce_x: number | null;
  bounce_y: number | null;
  in_court: boolean | null;
  depth: string | null;
  direction: string | null;
  outcome: string | null;
  sources: string[];
  quality: Record<string, unknown>;
  metrics: Record<string, number>;
};

export type Rally = {
  id: number;
  index: number;
  start_s: number;
  end_s: number;
  shot_count: number;
  server_id: number | null;
  winner_id: number | null;
  end_reason: string | null;
  score_before: Record<string, unknown> | null;
};

export type PlayerSummary = {
  id: number;
  name: string;
  handedness: string | null;
  height_m: number | null;
  created_at: string;
  thumbnail_url: string | null;
  session_count: number;
  shot_count: number;
  strokes: Record<string, number>;
  last_seen: string | null;
};

export type PlayerDetail = PlayerSummary & {
  sessions: {
    session_id: number;
    name: string | null;
    recorded_at: string | null;
    kind: SessionKind;
    label: string;
    stats: Record<string, unknown>;
  }[];
};

export type Library = {
  dir: string;
  videos: { name: string; path: string; size: number; modified: number; sessions: number[] }[];
  partial: { name: string; size: number; received: number; modified: number }[];
  disk: { free: number; total: number };
};

export class ApiError extends Error {
  status: number;
  body: Record<string, unknown>;
  constructor(status: number, message: string, body: Record<string, unknown>) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

// Listeners told when a request comes back 401, so the app can show the login screen.
const unauthorizedListeners = new Set<() => void>();
export function onUnauthorized(fn: () => void): () => void {
  unauthorizedListeners.add(fn);
  return () => unauthorizedListeners.delete(fn);
}

export async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  init?: RequestInit,
): Promise<T> {
  const headers: Record<string, string> = {};
  let payload: BodyInit | undefined;
  if (body instanceof Blob || body instanceof ArrayBuffer) {
    payload = body;
    headers["Content-Type"] = "application/octet-stream";
  } else if (body !== undefined) {
    payload = JSON.stringify(body);
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, { method, headers, body: payload, ...init });
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) {
    if (res.status === 401 && path !== "/api/login") unauthorizedListeners.forEach((f) => f());
    const detail = typeof data.detail === "string" ? data.detail : res.statusText;
    throw new ApiError(res.status, detail, data);
  }
  return data as T;
}

export const api = {
  me: () =>
    request<{ password_required: boolean; logged_in: boolean; version: string }>("GET", "/api/me"),
  login: (password: string) => request("POST", "/api/login", { password }),
  logout: () => request("POST", "/api/logout"),

  sessions: () => request<SessionSummary[]>("GET", "/api/sessions"),
  session: (id: number) => request<SessionDetail>("GET", `/api/sessions/${id}`),
  createSession: (paths: string[], name?: string) =>
    request<SessionDetail>("POST", "/api/sessions", { paths, name }),
  addVideos: (id: number, paths: string[]) =>
    request<SessionDetail>("POST", `/api/sessions/${id}/videos`, { paths }),
  removeVideo: (id: number, videoId: number) =>
    request<SessionDetail>("DELETE", `/api/sessions/${id}/videos/${videoId}`),
  patchSession: (id: number, patch: { name?: string; kind?: string }) =>
    request<SessionDetail>("PATCH", `/api/sessions/${id}`, patch),
  reanalyze: (id: number, force = false) =>
    request<Job>("POST", `/api/sessions/${id}/analyze?force=${force}`),
  deleteSession: (id: number) => request("DELETE", `/api/sessions/${id}`),
  shots: (id: number) => request<Shot[]>("GET", `/api/sessions/${id}/shots`),
  rallies: (id: number) => request<Rally[]>("GET", `/api/sessions/${id}/rallies`),

  jobs: (active = false) => request<Job[]>("GET", `/api/jobs?active=${active}`),
  cancelJob: (id: number) => request("POST", `/api/jobs/${id}/cancel`),

  players: () => request<PlayerSummary[]>("GET", "/api/players"),
  player: (id: number) => request<PlayerDetail>("GET", `/api/players/${id}`),
  renamePlayer: (id: number, name: string) =>
    request<PlayerDetail>("PATCH", `/api/players/${id}`, { name }),
  mergePlayer: (id: number, into: number) =>
    request<PlayerDetail>("POST", `/api/players/${id}/merge`, { into }),

  library: () => request<Library>("GET", "/api/library"),
  deleteLibraryVideo: (name: string) =>
    request("DELETE", `/api/library/${encodeURIComponent(name)}`),
  config: () =>
    request<{ path: string; text: string; defaults: Record<string, unknown> }>(
      "GET",
      "/api/config",
    ),
  saveConfig: (text: string) => request("PUT", "/api/config", { text }),
};
