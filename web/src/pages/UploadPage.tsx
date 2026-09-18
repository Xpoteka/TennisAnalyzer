import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { formatBytes, formatDate, pct, sessionTitle } from "../format";
import { useData } from "../hooks";
import { discardUpload } from "../upload";
import {
  addFiles,
  addLibraryVideo,
  fileKey,
  pause,
  pending,
  removeItem,
  resume,
  setName,
  setTarget,
  start,
  useUploads,
  type Handle,
  type Picked,
} from "../uploadStore";

const VIDEO = /\.(mp4|mov|m4v|mkv|avi|mts|webm)$/i;
const EXTENSIONS = [".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".webm"];

/** The server's name for an uploaded file (see safe_name in tennis/api/uploads.py). */
const safeName = (name: string) => name.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^\.+/, "");

type FilePicker = (options: {
  multiple: boolean;
  types: { description: string; accept: Record<string, string[]> }[];
}) => Promise<Handle[]>;

/** Files dropped on the page, with a handle to each where the browser offers one. */
async function droppedFiles(dt: DataTransfer): Promise<Picked[]> {
  const files = Array.from(dt.files);
  // Handles must be asked for before the drop event returns.
  const requests = Array.from(dt.items)
    .filter((i) => i.kind === "file")
    .map((i) => {
      const get = (i as { getAsFileSystemHandle?: () => Promise<FileSystemHandle | null> })
        .getAsFileSystemHandle;
      return get ? get.call(i).catch(() => null) : Promise.resolve(null);
    });
  const handles = (await Promise.all(requests)).filter(
    (h): h is Handle => h?.kind === "file",
  );
  return files.map((file) => ({ file, handle: handles.find((h) => h.name === file.name) }));
}

export default function UploadPage() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const uploads = useUploads();
  const [over, setOver] = useState(false);
  const [skipped, setSkipped] = useState<string>();
  const resumeAfterPick = useRef(false);
  const sessions = useData(() => api.sessions(), []);
  const library = useData(() => api.library(), []);
  const busy = uploads.phase === "uploading";
  const paused = uploads.phase === "paused";

  // Open the session once a batch started from this page is in.
  const seenFinish = useRef(uploads.finished?.seq);
  useEffect(() => {
    if (uploads.finished && uploads.finished.seq !== seenFinish.current) {
      navigate(`/sessions/${uploads.finished.session}`);
    }
  }, [uploads.finished, navigate]);

  const add = (picked: Picked[]) => {
    const videos = picked.filter((p) => VIDEO.test(p.file.name));
    const rejected = picked.length - videos.length;
    setSkipped(rejected ? `${rejected} file(s) skipped: not a video` : undefined);
    addFiles(videos);
    if (resumeAfterPick.current) {
      resumeAfterPick.current = false;
      resume();
    }
  };

  const choose = async () => {
    const picker = (window as { showOpenFilePicker?: FilePicker }).showOpenFilePicker;
    if (!picker) {
      input.current?.click();
      return;
    }
    try {
      const handles = await picker({
        multiple: true,
        types: [{ description: "Videos", accept: { "video/*": EXTENSIONS } }],
      });
      add(await Promise.all(handles.map(async (handle) => ({ file: await handle.getFile(), handle }))));
    } catch (err) {
      resumeAfterPick.current = false;
      // Cancelled, or a picker that refuses these types: fall back to the plain one.
      if (!(err instanceof DOMException && err.name === "AbortError")) input.current?.click();
    }
  };

  const onResume = async () => {
    // Without a kept handle the file has to be chosen again; the picker needs this click.
    if (pending(uploads).some((it) => !it.file && !it.handle)) {
      resumeAfterPick.current = true;
      choose();
      return;
    }
    await resume();
  };

  const discard = async (p: { name: string; size: number }) => {
    await discardUpload(p.name, p.size).catch(() => {});
    library.reload();
  };

  const inBatch = new Set(uploads.items.map((it) => it.key));
  const unused = library.data?.videos.filter((v) => !inBatch.has(v.path)) ?? [];
  const inBatchSafe = new Set(uploads.items.map((it) => fileKey(safeName(it.name), it.size)));
  const unfinished =
    library.data?.partial.filter((p) => !inBatchSafe.has(fileKey(p.name, p.size))) ?? [];
  const error = uploads.error ?? skipped;

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
        onClick={choose}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          droppedFiles(e.dataTransfer).then(add);
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
          onChange={(e) => {
            if (e.target.files) add(Array.from(e.target.files, (file) => ({ file })));
            e.target.value = "";
          }}
        />
      </div>

      {uploads.items.length > 0 && (
        <div className="card stack">
          {paused && (
            <div className="notice">
              Upload paused. What already arrived stays on the server:{" "}
              {uploads.items.some((it) => !it.file && !it.path && !it.handle)
                ? "choose the same files again and it continues from there."
                : "resume continues from there."}
            </div>
          )}
          <div>
            {uploads.items.map((it) => (
              <div className="file-row" key={it.key}>
                <div>
                  <div style={{ fontWeight: 500 }}>{it.name}</div>
                  <div className="small muted">
                    {formatBytes(it.size)}
                    {it.fromLibrary
                      ? " · already on the server"
                      : it.path
                        ? " · uploaded"
                        : it.sent > 0
                          ? ` · ${formatBytes(it.sent)} on the server`
                          : ""}
                    {busy && !it.path && it.rate > 0 ? ` · ${formatBytes(it.rate)}/s` : ""}
                    {paused && !it.path && !it.file && !it.handle ? " · choose this file again" : ""}
                  </div>
                </div>
                {busy ? (
                  <span className="small muted">{it.path ? "Ready" : pct(it.sent / it.size)}</span>
                ) : (
                  <button className="btn btn-ghost small" onClick={() => removeItem(it.key)}>
                    Remove
                  </button>
                )}
                {!it.fromLibrary && (
                  <div className="progress" style={{ gridColumn: "1 / -1" }}>
                    <div style={{ width: `${(it.sent / it.size) * 100}%` }} />
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="row" style={{ gap: 12 }}>
            <select
              className="select"
              value={uploads.target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={uploads.phase !== "draft"}
            >
              <option value="new">New session</option>
              {sessions.data?.map((s) => (
                <option key={s.id} value={s.id}>
                  Add to: {sessionTitle(s)} ({formatDate(s.recorded_at ?? s.created_at)})
                </option>
              ))}
            </select>
            {uploads.target === "new" && (
              <input
                className="input"
                placeholder="Name (optional)"
                value={uploads.name}
                onChange={(e) => setName(e.target.value)}
                disabled={uploads.phase !== "draft"}
                style={{ flex: "1 1 200px" }}
              />
            )}
            {busy ? (
              <button className="btn" onClick={pause}>
                Pause
              </button>
            ) : paused ? (
              <button className="btn btn-primary" onClick={onResume}>
                Resume upload
              </button>
            ) : (
              <button className="btn btn-primary" onClick={start}>
                Upload and analyse
              </button>
            )}
          </div>
        </div>
      )}

      {error && <div className="notice notice-error">{error}</div>}

      {unfinished.length > 0 && (
        <div className="stack">
          <h2>Unfinished uploads</h2>
          <p className="muted small">Drop the same file again to continue where it stopped.</p>
          <div className="card table-wrap" style={{ padding: 0 }}>
            <table>
              <tbody>
                {unfinished.map((p) => (
                  <tr key={fileKey(p.name, p.size)}>
                    <td>{p.name}</td>
                    <td className="muted small">
                      {pct(p.received / p.size)} of {formatBytes(p.size)}
                    </td>
                    <td className="muted small">{formatDate(new Date(p.modified * 1000).toISOString())}</td>
                    <td className="num">
                      <button className="btn btn-ghost small" onClick={() => discard(p)}>
                        Discard
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

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
                      <button
                        className="btn small"
                        onClick={() => addLibraryVideo(v)}
                        disabled={uploads.phase !== "draft"}
                      >
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
