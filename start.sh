#!/usr/bin/env bash
set -euo pipefail

# Optional: choose how many days to cache (D1=tomorrow, D2=day-after)
export PREFETCH_DAYS="2"   # we’ll prebuild 24h+72h ending for the next 2 days

python - << 'PY'
import datetime as dt
import os, sys
from server import build_24h_cog, build_72h_cog

# Dates: “tomorrow” and “day after tomorrow” in UTC calendar terms
today = dt.date.today()
dates = [today + dt.timedelta(days=1), today + dt.timedelta(days=2)]

print("Prefetching SNODAS melt COGs...")
for d in dates:
    try:
        print("  24h:", d)
        build_24h_cog(d)
        print("  72h:", d)
        build_72h_cog(d)
    except Exception as e:
        print("Prefetch warning for", d, "->", e, file=sys.stderr)

print("Prefetch done.")
PY

# Launch the API
exec uvicorn server:app --host 0.0.0.0 --port 8000
