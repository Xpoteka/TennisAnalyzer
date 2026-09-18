import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api, onUnauthorized } from "./api";
import JobsIndicator from "./components/JobsIndicator";
import Login from "./pages/Login";
import PlayerPage from "./pages/PlayerPage";
import PlayersPage from "./pages/PlayersPage";
import SessionPage from "./pages/SessionPage";
import SessionsPage from "./pages/SessionsPage";
import SettingsPage from "./pages/SettingsPage";
import UploadPage from "./pages/UploadPage";

type AuthState = "checking" | "in" | "out";

export default function App() {
  const [auth, setAuth] = useState<AuthState>("checking");
  const [needsPassword, setNeedsPassword] = useState(false);

  useEffect(() => {
    api
      .me()
      .then((me) => {
        setNeedsPassword(me.password_required);
        setAuth(me.logged_in ? "in" : "out");
      })
      .catch(() => setAuth("out"));
    return onUnauthorized(() => setAuth("out"));
  }, []);

  if (auth === "checking") return null;
  if (auth === "out") return <Login onLogin={() => setAuth("in")} />;

  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <span className="brand-ball" aria-hidden />
          <span>Tennis Analyzer</span>
        </NavLink>
        <nav className="nav">
          <NavLink to="/upload">Upload</NavLink>
          <NavLink to="/sessions">Sessions</NavLink>
          <NavLink to="/players">Players</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>
        <div className="topbar-right">
          <JobsIndicator />
          {needsPassword && (
            <button
              className="btn btn-ghost small"
              onClick={() => api.logout().then(() => setAuth("out"))}
            >
              Log out
            </button>
          )}
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/sessions" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/sessions/:id" element={<SessionPage />} />
          <Route path="/players" element={<PlayersPage />} />
          <Route path="/players/:id" element={<PlayerPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<div className="empty">Nothing here.</div>} />
        </Routes>
      </main>
    </div>
  );
}
