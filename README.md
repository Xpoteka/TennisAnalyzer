# Tennis Analyzer

Drop in the videos of a tennis session, training or match. The app works out which kind it is and measures what happened:

- every shot, by player and stroke;
- ball speed, height over the net, and where the ball landed;
- rallies, and for a match, the points, games and sets.

It recognises the players by itself and keeps a profile for each one. The profile holds their stats across sessions and an analysis of their technique.

**Status: v2 in progress** (branch `v2`). Working: uploads, the job queue, playback, court and camera, player and ball tracking, hits, strokes, ball speed/height/placement, training vs match, score keeping (point winners are still unreliable), player profiles recognised across sessions with technique analysis and trends, and several videos per session (lined up by sound). Accuracy on a labelled match is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Run it

You need [uv](https://docs.astral.sh/uv/), Node 22 and ffmpeg (`ffmpeg` and `ffprobe` on `PATH`).

```bash
brew install uv node ffmpeg              # macOS
uv sync                                  # Python 3.11 and dependencies into .venv
(cd web && npm ci && npm run build)      # the UI, built into tennis/api/static
uv run tennis serve                      # opens http://127.0.0.1:8731/
```

For UI work, run `uv run tennis serve --no-open` and `npm run dev` in `web/` together. Vite serves the UI on :5173 with hot reload and forwards `/api` to the server.

On a server, use the Docker image (`ghcr.io/xpoteka/tennisanalyzer`). It needs a password to listen on the network. See [docs/DEPLOY.md](docs/DEPLOY.md).

## Using it

- **Upload.** Drop one or more videos of one session: two cameras, or one recording in several parts. Uploads go up in 32 MB pieces and resume if cut off. Videos already on the server (copied into `data/uploads/`) can be picked from a list instead.
- **Sessions.** Each session shows its type (match or training, detected automatically, and you can override it), the players, and the stats. Click a shot or a point to watch it.
- **Players.** Everyone recognised, with stats and technique over time. Rename a profile, or merge two that are the same person.
- **Settings.** The video library, with free disk space, and `config.yaml`.

Analyses run one at a time in the background, in a separate process. They survive a restart of the server; finished steps are not redone.

## Recording tips

- Film from a **fixed, raised position behind a baseline** with the **whole court in view**. That gives the best results: ball speed, height and bounce spots need the court lines. From any other angle, the app still counts shots and analyses technique, and says which stats it could not measure.
- **50 fps or more** makes ball speeds and hit times more precise. 25–30 fps works.
- Keep the **sound**. Ball impacts are the most precise hit times, and two cameras are lined up by their audio.

## Command line

```text
tennis serve [--host H --port P --password-file F]   # the app and its analysis worker
tennis analyze VIDEO... [--session ID] [--force]     # analyse in this terminal, no server
```

Data lives in `data/` (or `paths.data_root` in `config.yaml`):

- `tennis.db` (SQLite): sessions, players, shots, rallies and the job queue;
- `uploads/`: the video library;
- `videos/<id>/`: per-video analysis files;
- `models/`: models downloaded on first use.

## Development

```bash
uv run pytest -q          # tests (ffmpeg needed for most)
uv run ruff check . && uv run ruff format --check . && uv run mypy
(cd web && npm run typecheck)
```
