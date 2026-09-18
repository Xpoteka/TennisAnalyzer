// The upload batch lives outside React, so it keeps going while you look at other pages, and
// its plan (which files, which session) is kept in IndexedDB, so a reload shows how far it got.
// A browser cannot keep sending a file across a reload. Chromium browsers can keep a handle to
// a dropped file, though, so resuming takes a click (and at most a permission prompt); other
// browsers need the file chosen again.

import { useSyncExternalStore } from "react";
import { api } from "./api";
import { uploadFile, uploadStatus } from "./upload";

/** The File System Access parts that TypeScript's DOM types do not include yet. */
export type Handle = FileSystemFileHandle & {
  queryPermission?(d: { mode: "read" }): Promise<PermissionState>;
  requestPermission?(d: { mode: "read" }): Promise<PermissionState>;
};

export type Picked = { file: File; handle?: Handle };

export type Item = {
  key: string;
  name: string;
  size: number;
  file?: File;
  handle?: Handle;
  path?: string; // set once the whole file is on the server
  fromLibrary?: boolean;
  sent: number; // bytes already on the server
  rate: number;
};

export type UploadState = {
  items: Item[];
  target: string; // "new" or a session id
  name: string;
  phase: "draft" | "uploading" | "paused";
  error?: string;
  // The session the last batch went into; `seq` tells two batches for one session apart.
  finished?: { session: number; seq: number };
};

let state: UploadState = { items: [], target: "new", name: "", phase: "draft" };
const listeners = new Set<() => void>();
let controller: AbortController | undefined;
let finishedSeq = 0;

function set(patch: Partial<UploadState>) {
  state = { ...state, ...patch };
  listeners.forEach((f) => f());
}

function patchItem(key: string, patch: Partial<Item>) {
  set({ items: state.items.map((it) => (it.key === key ? { ...it, ...patch } : it)) });
}

export function useUploads(): UploadState {
  return useSyncExternalStore(
    (f) => {
      listeners.add(f);
      return () => listeners.delete(f);
    },
    () => state,
  );
}

export const fileKey = (name: string, size: number) => `${name}:${size}`;

/** Items whose bytes still have to reach the server. */
export const pending = (s: UploadState) => s.items.filter((it) => !it.path);

/** Fraction of the batch's uploaded bytes already on the server. */
export function overallProgress(s: UploadState): number {
  const own = s.items.filter((it) => !it.fromLibrary);
  const size = own.reduce((a, it) => a + it.size, 0);
  return size ? own.reduce((a, it) => a + it.sent, 0) / size : 1;
}

// --- editing the batch ----------------------------------------------------------------------

export function addFiles(picked: Picked[]) {
  const items = [...state.items];
  const added: string[] = [];
  for (const { file, handle } of picked) {
    const key = fileKey(file.name, file.size);
    const existing = items.findIndex((it) => it.key === key);
    if (existing >= 0) {
      // The same file again: this is how a paused upload gets its file back.
      if (!items[existing].path) items[existing] = { ...items[existing], file, handle };
      continue;
    }
    items.push({ key, name: file.name, size: file.size, file, handle, sent: 0, rate: 0 });
    added.push(key);
  }
  set({ items, error: undefined });
  if (state.phase !== "draft") save();
  for (const key of added) refreshSent(key);
}

export function addLibraryVideo(v: { name: string; path: string; size: number }) {
  if (state.items.some((it) => it.path === v.path)) return;
  const item = { key: v.path, name: v.name, size: v.size, path: v.path, fromLibrary: true };
  set({ items: [...state.items, { ...item, sent: v.size, rate: 0 }] });
}

export function removeItem(key: string) {
  set({ items: state.items.filter((it) => it.key !== key) });
  if (!state.items.length) {
    set({ phase: "draft", error: undefined });
    clearSaved();
  } else if (state.phase !== "draft") {
    save();
  }
}

export const setTarget = (target: string) => set({ target });
export const setName = (name: string) => set({ name });

/** Ask the server how much of an item already arrived (an earlier, interrupted upload). */
async function refreshSent(key: string) {
  const it = state.items.find((x) => x.key === key);
  if (!it || it.path) return;
  try {
    const status = await uploadStatus(it.name, it.size);
    patchItem(key, status.done ? { path: status.path, sent: it.size } : { sent: status.received });
  } catch {
    // Shown as not started; the upload itself reports real errors.
  }
}

// --- running ----------------------------------------------------------------------------------

/** Upload and analyse: send every file, then create the session or add to it. */
export async function start() {
  if (state.phase === "uploading") return;
  controller = new AbortController();
  const signal = controller.signal;
  set({ phase: "uploading", error: undefined });
  await save();
  try {
    for (const key of state.items.map((it) => it.key)) {
      const it = state.items.find((x) => x.key === key);
      if (!it || it.path) continue;
      if (!it.file) throw new Error(`${it.name} has to be chosen again`);
      const path = await uploadFile(
        it.file,
        (p) => patchItem(key, { sent: p.sent, rate: p.rate }),
        signal,
      );
      patchItem(key, { path, sent: it.size, rate: 0 });
      await save();
    }
    const paths = state.items.map((it) => it.path!);
    const session =
      state.target === "new"
        ? await api.createSession(paths, state.name || undefined)
        : await api.addVideos(Number(state.target), paths);
    await clearSaved();
    set({
      items: [],
      phase: "draft",
      name: "",
      target: "new",
      finished: { session: session.id, seq: ++finishedSeq },
    });
  } catch (err) {
    const items = state.items.map((it) => ({ ...it, rate: 0 }));
    if (signal.aborted) {
      set({ items, phase: "paused" });
    } else {
      const message = err instanceof Error ? err.message : String(err);
      set({ items, phase: "paused", error: `Upload stopped: ${message}.` });
    }
  }
}

export function pause() {
  controller?.abort();
}

/**
 * Get the files of a paused batch back and go on. Must run from a click: asking for
 * permission to read a kept handle needs one. Returns the names still missing a file.
 */
export async function resume(): Promise<string[]> {
  for (const it of pending(state)) {
    if (it.file || !it.handle) continue;
    const file = await fileFromHandle(it.handle, true);
    if (file && file.size === it.size) patchItem(it.key, { file });
  }
  const missing = pending(state).filter((it) => !it.file);
  if (missing.length) {
    set({ error: `Drop or choose ${missing.map((it) => it.name).join(", ")} again to resume.` });
    return missing.map((it) => it.name);
  }
  start();
  return [];
}

async function fileFromHandle(handle: Handle, ask: boolean): Promise<File | undefined> {
  try {
    let perm = (await handle.queryPermission?.({ mode: "read" })) ?? "granted";
    if (perm !== "granted" && ask && handle.requestPermission) {
      perm = await handle.requestPermission({ mode: "read" });
    }
    return perm === "granted" ? await handle.getFile() : undefined;
  } catch {
    return undefined; // moved, deleted, or permission refused
  }
}

// --- kept across reloads ----------------------------------------------------------------------

type Saved = {
  target: string;
  name: string;
  items: Pick<Item, "key" | "name" | "size" | "handle" | "path" | "fromLibrary">[];
};

const DB = "tennis-uploads";
const STORE = "batch";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  fn: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const req = fn(db.transaction(STORE, mode).objectStore(STORE));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

async function save() {
  const saved = (withHandles: boolean): Saved => ({
    target: state.target,
    name: state.name,
    items: state.items.map(({ key, name, size, handle, path, fromLibrary }) => ({
      key,
      name,
      size,
      path,
      fromLibrary,
      handle: withHandles ? handle : undefined,
    })),
  });
  try {
    await withStore("readwrite", (s) => s.put(saved(true), "current"));
  } catch {
    // A browser that cannot store file handles still keeps the plan.
    await withStore("readwrite", (s) => s.put(saved(false), "current")).catch(() => {});
  }
}

async function clearSaved() {
  await withStore("readwrite", (s) => s.delete("current")).catch(() => {});
}

let restored = false;

/** Once logged in: bring back a batch that a reload interrupted, and go on if allowed to. */
export async function restoreUploads() {
  if (restored) return;
  restored = true;
  let saved: Saved | undefined;
  try {
    saved = await withStore<Saved | undefined>("readonly", (s) => s.get("current"));
  } catch {
    return; // private window, or storage blocked
  }
  if (!saved?.items.length || state.items.length) return;
  set({
    target: saved.target,
    name: saved.name,
    phase: "paused",
    items: saved.items.map((it) => ({ ...it, sent: it.path ? it.size : 0, rate: 0 })),
  });
  await Promise.all(pending(state).map((it) => refreshSent(it.key)));
  // Permission can outlive the page ("allow on every visit"); then no click is needed.
  for (const it of pending(state)) {
    const file = it.handle && (await fileFromHandle(it.handle, false));
    if (file && file.size === it.size) patchItem(it.key, { file });
  }
  if (!pending(state).some((it) => !it.file)) start();
}

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", (e) => {
    if (state.phase === "uploading") e.preventDefault(); // "Leave site?"; the upload would pause
  });
}
