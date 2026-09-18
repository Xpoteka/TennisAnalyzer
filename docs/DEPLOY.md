# Running the UI on a server (TrueNAS / HexOS, as a Docker app)

Every push to `main` builds a Docker image and publishes it as
`ghcr.io/xpoteka/tennisanalyzer:latest` (the `image` job in
[.github/workflows/ci.yml](../.github/workflows/ci.yml)). It is only published after lint,
type checks and tests pass. A pull request builds the image without publishing it. TrueNAS
runs that image as a custom app. Nothing needs to be installed on the NAS.

The image is **CPU only**: PyTorch without CUDA, so the image stays around 1.5 GB. Pose
extraction is therefore several times slower than on an Apple-silicon Mac.

## 1. A dataset for the data

Create a dataset, for example `tennis-data` (HexOS: a folder on your pool). It holds
everything that must survive an update: the config, the sessions, **every uploaded video**,
and the downloaded models.

The app runs as TrueNAS's `apps` user (uid 568), so give that user read and write access:
**Datasets → tennis-data → Permissions → Edit → Add item → User: apps → Modify**.

Tip: also share the dataset over SMB (with your own user added to the permissions). Then you
can copy videos from your Mac straight into `tennis-data/uploads/` in Finder. They appear in
the **Videos** view, and for multi-GB files this is much faster than uploading through the
browser.

## 2. Install the app

In TrueNAS: **Apps → Discover Apps → ⋮ (top right) → Install via YAML**. Name it `tennis`, and
paste [docker/truenas-app.yaml](../docker/truenas-app.yaml). Change two things:

- `TENNIS_UI_PASSWORD`: your login password, at least 10 characters. The app does not start
  without one. Changing it later logs every browser out.
- `/mnt/tank/tennis-data`: the path of your dataset (`/mnt/<pool>/<dataset>`).

Save. When the app is running, open `http://<nas-ip>:8731/` and log in.

The **Custom App** form works too, with the same settings: image
`ghcr.io/xpoteka/tennisanalyzer`, tag `latest`, the environment variable, port 8731, and a
host path mounted at `/data`.

On first start the app writes `/data/config.yaml`, which the UI's **Config** view edits.
Models (YOLO, Whisper) are downloaded into `/data/models/` the first time they are needed.

## 3. Updating

Push to `main`. When the workflow's `image` job has finished (the repository's **Actions**
tab), restart the app in TrueNAS: `pull_policy: always` makes it fetch the new image on
start. Your data is untouched.

## 4. Your domain, over HTTPS

The app speaks plain HTTP on port 8731. Put one of these in front of it, both available as
TrueNAS apps:

**Cloudflare Tunnel (no ports opened on your router).** Needs the domain on Cloudflare.
Install the **cloudflared** app, create a tunnel in the Cloudflare dashboard (Zero Trust →
Networks → Tunnels), and add a public hostname such as `tennis.example.com` → `HTTP` →
`<nas-ip>:8731`. Cloudflare refuses request bodies over 100 MB. The UI uploads videos in
32 MB pieces, so large files still work.

**Nginx Proxy Manager (ports 80 and 443 forwarded to the NAS).** Add a proxy host
`tennis.example.com` → `http://<nas-ip>:8731`, and request a Let's Encrypt certificate on the
SSL tab with **Force SSL** on. For plain nginx, allow request bodies of at least 64 MB
(`client_max_body_size 64m;`). Also pass the `Host` (or `X-Forwarded-Host`) and
`X-Forwarded-Proto` headers.

The login cookie is marked `Secure` whenever the proxy reports HTTPS.

## Moving your existing sessions from the Mac

1. Copy the Mac's `data/` folder into the dataset, so that `tennis-data/sessions/`,
   `tennis-data/models/` and `tennis-data/uploads/` exist.
2. Copy the raw videos of those sessions into `tennis-data/uploads/`, keeping their names.
   To keep their modification times, drag them in Finder, or use `rsync -t`. Otherwise
   every stage reruns the first time.
3. In the UI, run **All commands → Relink moved videos**. Every session whose video is
   missing is pointed at the file of the same name and size in `uploads/`. Videos uploaded
   from now on are linked relative to the data folder, so this is a one-time step.

## Good to know

- **Disk.** Uploaded videos are never removed automatically. The **Videos** view shows the
  free space, and deleting a video there keeps its sessions' results. Only rerunning the
  pipeline needs the video.
- **Security.** Anyone with the password can run the pipeline and read any file the app
  can. Use a long password, and keep the app behind HTTPS. Failed logins are slowed to
  about one per second.
- **The app stops right away.** Its log (**Apps → tennis → Logs**) says why. Usually the
  password is missing or too short, or `/data` is not writable by the `apps` user.
- **Running the image elsewhere:**
  `docker run -p 8731:8731 -e TENNIS_UI_PASSWORD=... -v /some/folder:/data ghcr.io/xpoteka/tennisanalyzer`.
  Add `--user $(id -u):$(id -g)` if that folder belongs to you rather than uid 568.
