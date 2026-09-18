import { useState } from "react";
import { api } from "../api";

export default function Login({ onLogin }: { onLogin: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(undefined);
    try {
      await api.login(password);
      onLogin();
    } catch (err) {
      setError(err instanceof Error ? err.message : "login failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card login stack" onSubmit={submit}>
      <div className="brand">
        <span className="brand-ball" aria-hidden />
        <span>Tennis Analyzer</span>
      </div>
      <label className="stack" style={{ gap: 6 }}>
        <span className="small muted">Password</span>
        <input
          className="input"
          type="password"
          autoFocus
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </label>
      {error && <div className="notice notice-error">{error}</div>}
      <button className="btn btn-primary" disabled={busy || !password}>
        Log in
      </button>
    </form>
  );
}
