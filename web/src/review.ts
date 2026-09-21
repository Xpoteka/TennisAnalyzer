// What the review player, the Errors tab and the Patterns tab agree on: what counts as an
// error, which point a moment belongs to, and the tracks drawn over the video.

import { api, type Rally, type Shot, type TrackSlice } from "./api";
import { COURT } from "./components/Court";

export type ErrorKind = "net" | "out" | "fault" | "double_fault";

export const ERROR_LABELS: Record<ErrorKind, string> = {
  net: "into the net",
  out: "out",
  fault: "serve fault",
  double_fault: "double fault",
};

/** The error this shot was, if any. Serves count as faults, not as plain errors. */
export function shotError(shot: Shot, rally?: Rally): ErrorKind | null {
  const missed = shot.outcome === "net" || shot.outcome === "out";
  if (shot.stroke === "serve") {
    if (rally?.end_reason === "double_fault") return "double_fault";
    if (rally?.end_reason === "fault" || missed) return "fault";
    return null;
  }
  return missed ? (shot.outcome as ErrorKind) : null;
}

/** How a ball that went out missed: past the baseline (or service line) or past the side. */
export function missSide(shot: Shot): "long" | "wide" | null {
  if (shot.bounce_x == null || shot.bounce_y == null || shot.outcome !== "out") return null;
  const limit = shot.stroke === "serve" ? COURT.serviceLine : COURT.length / 2;
  return Math.abs(shot.bounce_y) > limit ? "long" : "wide";
}

export type ShotIndex = {
  rallyOf: Map<number, Rally>;
  shotsOf: Map<number, Shot[]>;
  errorOf: Map<number, ErrorKind>;
};

export function indexShots(shots: Shot[], rallies: Rally[]): ShotIndex {
  const rallyOf = new Map(rallies.map((r) => [r.id, r]));
  const shotsOf = new Map<number, Shot[]>();
  const errorOf = new Map<number, ErrorKind>();
  for (const s of shots) {
    if (s.rally_id != null) {
      if (!shotsOf.has(s.rally_id)) shotsOf.set(s.rally_id, []);
      shotsOf.get(s.rally_id)!.push(s);
    }
    const kind = shotError(s, s.rally_id != null ? rallyOf.get(s.rally_id) : undefined);
    if (kind) errorOf.set(s.id, kind);
  }
  return { rallyOf, shotsOf, errorOf };
}

/** The rally being played at session time `t`, with a little room either side. */
export function rallyAt(rallies: Rally[], t: number): Rally | undefined {
  return rallies.find((r) => t >= r.start_s - 1.5 && t <= r.end_s + 2);
}

const SLICE_S = 10;
const KEEP = 12;

/** A video's tracks, fetched in ten-second slices as the playhead needs them. */
export class TrackCache {
  private slices = new Map<number, TrackSlice | "loading">();
  constructor(
    private videoId: number,
    private onLoaded: () => void,
  ) {}

  /** Make sure the slices around PTS time `t` are loaded or on their way. */
  want(t: number) {
    const k = Math.floor(t / SLICE_S);
    for (const i of [k, k + 1, k - 1]) if (i >= 0 && !this.slices.has(i)) this.fetch(i);
  }

  private async fetch(i: number) {
    this.slices.set(i, "loading");
    try {
      this.slices.set(i, await api.tracks(this.videoId, i * SLICE_S, (i + 1) * SLICE_S));
      for (const key of this.slices.keys()) {
        if (this.slices.size <= KEEP) break;
        if (Math.abs(key - i) > 2) this.slices.delete(key);
      }
      this.onLoaded();
    } catch {
      this.slices.delete(i); // asked for again the next time it is wanted
    }
  }

  /** Rows of one kind with a time in [from, to], oldest first. */
  rows<K extends "ball" | "poses" | "feet">(kind: K, from: number, to: number): TrackSlice[K] {
    const out: unknown[] = [];
    for (let i = Math.max(0, Math.floor(from / SLICE_S)); i <= Math.floor(to / SLICE_S); i++) {
      const slice = this.slices.get(i);
      if (!slice || slice === "loading") continue;
      for (const row of slice[kind]) if (row[0] >= from && row[0] <= to) out.push(row);
    }
    return out as TrackSlice[K];
  }
}
