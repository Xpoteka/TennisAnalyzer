import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { formatBytes, formatDate, sessionTitle } from "../format";
import { useData } from "../hooks";
import { uploadFile, type UploadProgress } from "../upload";

type Item = {
  key: string;
  file?: File;
  path?: string; // set once on the server
  name: string;
  size: number;
  progress?: UploadProgress;
  error?: string;
};

const VIDEO = /\.(mp4|mov|m4v|mkv|avi|mts|webm)$/i;

export default function UploadPage() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [over, setOver] = useState(false);
  const [target, setTarget] = useState<string>("new");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const sessions = useData(() => api.sessions(), []);
  const library = useData(() => api.library(), []);

  const addFiles = (files: FileList | File[]) => {
    const add = Array.from(files).filter((f) => VIDEO.test(f.name));
    const rejected = Array.from(files).length - add.length;
    setError(rejected ? `${rejected} file(s) skipped: not a video` : undefined);
    setItems((cur) => [
      ...cur,
      ...add
        .filter((f) => !cur.some((c) => c.key === `${f.name}:${f.size}`))
        .map((f) => ({ key: `${f.name}:${f.size}`, file: f, name: f.name, size: f.size })),
    ]);
  };

  const addServerVideo = (v: { name: string; path: string; size: number }) =>
    setItems((cur) =>
      cur.some((c) => c.path === v.path)
        ? cur
        : [...cur, { key: v.path, path: v.path, name: v.name, size: v.size }],
    );

  const update = (key: string, patch: Partial<Item>) =>
    setItems((cur) => cur.map((it) => (it.key === key ? { ...it, ...patch } : it)));

  const start = async () => {
    setBusy(true);
    setError(undefined);
    const paths: string[] = [];
    try {
      for (const it of items) {
        if (it.path) {
          paths.push(it.path);
          continue;
        }
        try {
          const path = await uploadFile(it.file!, (p) => update(it.key, { progress: p }));
          update(it.key, { path });
          paths.push(path);
        } catch (err) {
          update(it.key, { error: err instanceof Error ? err.message : String(err) });
          throw err;
        }
      }
      const session =
        target === "new"
          ? await api.createSession(paths, name || undefined)
          : await api.addVideos(Number(target), paths);
      navigate(`/sessions/${session.id}`);
    } catch (err) {
      setError(
        `Upload stopped: ${err instanceof Error ? err.message : err}. ` +
          "Start again to resume where it stopped.",
      );
    } finally {
      setBusy(false);
    }
  };

  const unused = library.data?.videos.filter((v) => !items.some((i) => i.path === v.path)) ?? [];

  return (
    <div className="stack" style={{ gap: 24 }}>
      <div className="page-head">
        <div>
          <h1>Upload a session</h1>
          <p>
            Drop the videos of one session, training or match. Several files are fine: two
            cameras, or one recording split into parts. They are lined up automatically.
          </p>
        </div>
      </div>

      <div
        className={`dropzone${over ? " over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          addFiles(e.dataTransfer.files);
        }}
      >
        <h2>Drop videos here</h2>
        <div className="muted">or click to choose files · MP4, MOV, MKV …</div>
        <input
          ref={input}
          type="file"
          accept="video/*,.mkv,.mts"
          multiple
          hidden
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />
      </div>

      {items.length > 0 && (
        <div className="card stack">
          <div>
            {items.map((it) => (
              <div className="file-row" key={it.key}>
                <div>
                  <div style={{ fontWeight: 500 }}>{it.name}</div>
                  <div className="small muted">
                    {formatBytes(it.size)}
                    {it.file ? "" : " · already on the server"}
                    {it.progress && it.progress.sent < it.size && it.progress.rate > 0
                      ? ` · ${formatBytes(it.progress.rate)}/s`
                      : ""}
                    {it.error ? ` · ${it.error}` : ""}
                  </div>
                </div>
                {!busy ? (
                  <button
                    className="btn btn-ghost small"
                    onClick={() => setItems((c) => c.filter((x) => x.key !== it.key))}
                  >
                    Remove
                  </button>
                ) : (
                  <span className="small muted">
                    {it.path ? "Ready" : `${Math.floor(((it.progress?.sent ?? 0) / it.size) * 100)}%`}
                  </span>
                )}
                {it.file && (
                  <div className="progress" style={{ gridColumn: "1 / -1" }}>
                    <div style={{ width: `${((it.progress?.sent ?? 0) / it.size) * 100}%` }} />
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="row" style={{ gap: 12 }}>
            <select
              className="select"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={busy}
            >
              <option value="new">New session</option>
              {sessions.data?.map((s) => (
                <option key={s.id} value={s.id}>
                  Add to: {sessionTitle(s)} ({formatDate(s.recorded_at ?? s.created_at)})
                </option>
              ))}
            </select>
            {target === "new" && (
              <input
                className="input"
                placeholder="Name (optional)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={busy}
                style={{ flex: "1 1 200px" }}
              />
            )}
            <button className="btn btn-primary" onClick={start} disabled={busy || !items.length}>
              {busy ? "Uploading…" : "Upload and analyse"}
            </button>
          </div>
        </div>
      )}

      {error && <div className="notice notice-error">{error}</div>}

      {unused.length > 0 && (
        <div className="stack">
          <h2>Videos already on the server</h2>
          <div className="card table-wrap" style={{ padding: 0 }}>
            <table>
              <tbody>
                {unused.map((v) => (
                  <tr key={v.path}>
                    <td>{v.name}</td>
                    <td className="muted small">{formatBytes(v.size)}</td>
                    <td className="muted small">
                      {v.sessions.length ? `in ${v.sessions.length} session(s)` : "not analysed"}
                    </td>
                    <td className="num">
                      <button className="btn small" onClick={() => addServerVideo(v)} disabled={busy}>
                        Add
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
