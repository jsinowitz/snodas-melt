#!/usr/bin/env bash
set -u
set -o pipefail

PYTHON=/opt/conda/envs/snodas/bin/python
UVICORN=/opt/conda/envs/snodas/bin/uvicorn

"$PYTHON" - <<'PY' || true
import datetime as dt, requests, sys
try:
  from server import build_24h_cog, build_72h_cog, nsidc_daily_tar_url
except Exception as e:
  print(f"Prefetch import warning: {e}", file=sys.stderr)
else:
  def first_available(dates):
    for d in dates:
      try:
        url = nsidc_daily_tar_url(d)
        r = requests.head(url, timeout=30, allow_redirects=True)
        if r.status_code == 200: return d
      except Exception:
        pass
    return None

  utc_today = dt.date.today()
  candidates = [utc_today, utc_today-dt.timedelta(days=1), utc_today-dt.timedelta(days=2), utc_today-dt.timedelta(days=3)]
  avail = first_available(candidates)
  if avail is None:
    print("Prefetch: no available SNODAS day yet (today..today-3). Will build on first tile request.")
  else:
    for fn, label in ((build_24h_cog,"24h"), (build_72h_cog,"72h")):
      try:
        print(f"Prefetch: building {label} COG for {avail}")
        fn(avail)
      except Exception as e:
        print(f"Prefetch warning {label} {avail} -> {e}", file=sys.stderr)
sys.exit(0)
PY

exec "$UVICORN" server:app --host 0.0.0.0 --port 8000
