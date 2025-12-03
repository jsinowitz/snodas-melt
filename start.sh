#!/usr/bin/env bash
# safe startup: never die before uvicorn
set -u                       # (no -e; allow prefetch to fail without aborting)
set -o pipefail

PYTHON=/opt/conda/envs/snodas/bin/python
UVICORN=/opt/conda/envs/snodas/bin/uvicorn

# ---- Non-fatal prefetch (today..today-3) ----
"$PYTHON" - <<'PY' || true
import datetime as dt
import requests
import sys
try:
    from server import build_24h_cog, nsidc_daily_tar_url
except Exception as e:
    print(f"Prefetch import warning: {e}", file=sys.stderr)
else:
    def first_available(dates):
        for d in dates:
            try:
                url = nsidc_daily_tar_url(d)build_72h_cog(d)
                r = requests.head(url, timeout=30, allow_redirects=True)
                if r.status_code == 200:
                    return d
            except Exception:
                pass
        return None

    utc_today = dt.date.today()
    candidates = [utc_today,
                  utc_today - dt.timedelta(days=1),
                  utc_today - dt.timedelta(days=2),
                  utc_today - dt.timedelta(days=3)]

    avail = first_available(candidates)
    if avail is None:
        print("Prefetch: no available SNODAS day yet (today..today-3). Will build on first tile request.")
    else:
        print(f"Prefetch: building 24h COG for {avail}")
        try:
            build_24h_cog(avail)
        except Exception as e:
            print(f"Prefetch warning for {avail} -> {e}", file=sys.stderr)
# Always exit 0 so startup continues
sys.exit(0)
PY

# ---- Run API (always) ----
exec "$UVICORN" server:app --host 0.0.0.0 --port 8000
