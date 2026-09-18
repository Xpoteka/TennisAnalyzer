import { useEffect, useState } from "react";
import { api } from "../api";
import { formatBytes, formatDate } from "../format";
import { useData } from "../hooks";

export default function SettingsPage() {
  const library = useData(() => api.library(), []);
  const config = useData(() => api.config(), []);
  const [text, setText] = useState("");
  const [saved, setSaved] = useState<string>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (config.data) setText(config.data.text);
  }, [config.data]);

  const lib = library.data;
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <p>The videos kept on the server, and the analysis configuration.</p>
        </div>
      </div>

      <section className="stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Video library</h2>
          {lib && (
            <span className="small muted">
              {formatBytes(lib.disk.free)} free of {formatBytes(lib.disk.total)}
            </span>
          )}
        </div>
        <div className="card table-wrap" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Size</th>
                <th>Added</th>
                <th>Sessions</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {lib?.videos.map((v) => (
                <tr key={v.path}>
                  <td>{v.name}</td>
                  <td>{formatBytes(v.size)}</td>
                  <td>{formatDate(new Date(v.modified * 1000).toISOString())}</td>
                  <td>{v.sessions.length ? v.sessions.map((s) => `#${s}`).join(", ") : "—"}</td>
                  <td className="num">
                    <button
                      className="btn btn-ghost small btn-danger"
                      onClick={async () => {
                        const note = v.sessions.length
                          ? " Its sessions keep their results, but can no longer be re-analysed or played."
                          : "";
                        if (!confirm(`Delete ${v.name} from the server?${note}`)) return;
                        await api.deleteLibraryVideo(v.name);
                        library.reload();
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {lib?.partial.map((p) => (
                <tr key={p.name}>
                  <td>{p.name}</td>
                  <td colSpan={4} className="muted small">
                    unfinished upload: {formatBytes(p.received)} of {formatBytes(p.size)}; drop the
                    same file again to resume it
                  </td>
                </tr>
              ))}
              {lib && !lib.videos.length && !lib.partial.length && (
                <tr>
                  <td colSpan={5} className="muted">
                    No videos uploaded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="stack">
        <h2>Configuration</h2>
        <div className="small muted">
          {config.data?.path} · YAML · every key is optional; see config.example.yaml for the full
          list. Changes apply to the next analysis.
        </div>
        <textarea
          className="input mono"
          rows={16}
          value={text}
          spellCheck={false}
          onChange={(e) => {
            setText(e.target.value);
            setSaved(undefined);
          }}
        />
        {error && <div className="notice notice-error">{error}</div>}
        <div className="row">
          <button
            className="btn btn-primary"
            onClick={async () => {
              setError(undefined);
              try {
                await api.saveConfig(text);
                setSaved("Saved.");
              } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
              }
            }}
          >
            Save
          </button>
          {saved && <span className="muted small">{saved}</span>}
        </div>
      </section>
    </div>
  );
}
