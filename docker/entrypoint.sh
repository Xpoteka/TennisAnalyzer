#!/bin/sh
# Start the UI on every interface. The login password comes from TENNIS_UI_PASSWORD (at
# least 10 characters); without it the UI refuses to start.
set -eu

DATA="${TENNIS_DATA:-/data}"
if [ ! -w "$DATA" ]; then
    echo "error: $DATA is not writable by uid $(id -u). Give that user write access to the" \
        "host folder mounted there (TrueNAS: user 'apps', uid 568)." >&2
    exit 1
fi
if [ ! -f "$DATA/config.yaml" ]; then
    # Relative paths are resolved against the config's own folder, i.e. the data folder.
    cat >"$DATA/config.yaml" <<'YAML'
# Every key has a default; see config.example.yaml in the repository for the full list.
# Edit it in the UI's Config view.
paths:
  data_root: .
  labels_dir: ./labels
YAML
fi
mkdir -p "$DATA/labels" "$DATA/uploads"

exec tennis ui --host 0.0.0.0 --port "${TENNIS_PORT:-8731}" --no-open \
    --config "$DATA/config.yaml" "$@"
