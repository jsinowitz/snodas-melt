from __future__ import annotations
import calendar, gzip, io, os, random, re, sys, tarfile, tempfile, threading, time, requests, json, rasterio, rioxarray
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import COGReader
from rio_tiler.utils import render
from starlette.responses import PlainTextResponse
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub._commit_api import CommitOperationAdd, CommitOperationDelete
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo(os.environ.get("LOCAL_TZ", "America/Chicago"))
LOWRES_Z = int(os.environ.get("LOWRES_Z", "6"))         # serve off-track as z=6 tile
LOWRES_MAX_REQUEST_Z = int(os.environ.get("LOWRES_MAX_REQUEST_Z", "14"))
LOWRES_BOX_ZMIN = int(os.environ.get("LOWRES_BOX_ZMIN", "4"))  # pre-bake lowres tiles at these zooms
LOWRES_BOX_ZMAX = int(os.environ.get("LOWRES_BOX_ZMAX", "6"))
HF_MAX_OPS_PER_COMMIT = int(os.environ.get("HF_MAX_OPS_PER_COMMIT", "800"))
HF_PUSH_SLEEP_SEC = float(os.environ.get("HF_PUSH_SLEEP_SEC", "0.0"))

ROLL_WINDOW_DAYS = int(os.environ.get("ROLL_WINDOW_DAYS", "4"))  # yesterday..+2
SCHEDULE_HOUR_LOCAL = int(os.environ.get("SCHEDULE_HOUR_LOCAL", "7"))  # 7am local
SCHEDULE_MIN_LOCAL = int(os.environ.get("SCHEDULE_MIN_LOCAL", "0"))

WARM_ZMIN = int(os.environ.get("WARM_ZMIN", "6"))
WARM_ZMAX = int(os.environ.get("WARM_ZMAX", "11"))
WARM_RADIUS = int(os.environ.get("WARM_RADIUS", "1"))
WARM_CAP_PER_ZOOM = int(os.environ.get("WARM_CAP_PER_ZOOM", "6000"))
WARM_MAX_TILES_TOTAL = int(os.environ.get("WARM_MAX_TILES_TOTAL", "40000"))
WARM_DOM = os.environ.get("WARM_DOM", "zz").lower()

# Corridor prewarm hours (24 and 72)
WARM_HOURS_LIST = [24, 72]

# How long to keep tile PNGs and forecast COGs (days of YYYYMMDD date keys)
KEEP_TILE_DAYS = int(os.environ.get("KEEP_TILE_DAYS", "6"))     # keeps a little extra buffer
KEEP_FORECAST_COG_DAYS = int(os.environ.get("KEEP_FORECAST_COG_DAYS", "10"))

CACHE_CONTROL_BY_DATE = os.environ.get(
    "CACHE_CONTROL_BY_DATE",
    "public, max-age=31536000, immutable",
)
CACHE_CONTROL_LATEST = os.environ.get(
    "CACHE_CONTROL_LATEST",
    "public, max-age=900",
)

def _tile_cache_dir() -> Path:
    d = CACHE / "tilecache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _tile_cache_key(dom: str, hours: int, ymd: str, z: int, x: int, y: int) -> str:
    dom = (dom or "zz").lower()
    return f"bydate_{dom}_h{int(hours)}_{ymd}_z{int(z)}_{int(x)}_{int(y)}.png"

def _tile_cache_path(dom: str, hours: int, ymd: str, z: int, x: int, y: int) -> Path:
    return _tile_cache_dir() / _tile_cache_key(dom, hours, ymd, z, x, y)

_LOCK_ROOT = Path(os.environ.get("SNODAS_LOCK_DIR", "/tmp/snodas-locks")).expanduser()
_LOCK_ROOT.mkdir(parents=True, exist_ok=True)

def _acquire_singleton_lock(name: str = "scheduler") -> bool:
    lock_path = _LOCK_ROOT / f"{name}.lock"
    fd = None
    try:
        fd = os.open(lock_path.as_posix(), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        return True
    except FileExistsError:
        return False
    except Exception:
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass

def _pick_cache_root() -> Path:
    env = (os.environ.get("SNODAS_CACHE") or "").strip()
    candidates = ([env] if env else []) + ["/data/snodas-cache", "/home/user/snodas-cache", "/tmp/snodas-cache", "./cache"]
    for c in candidates:
        p = Path(c).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".wtest"
            t.write_text("ok")
            t.unlink(missing_ok=True)
            return p.resolve()
        except Exception:
            pass
    return Path(tempfile.mkdtemp(prefix="snodas-cache-")).resolve()
CACHE = _pick_cache_root()

CFGRIB_DIR = CACHE / "cfgrib"
CFGRIB_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CFGRIB_INDEXPATH", str(CFGRIB_DIR))

COLLAB_BASE = "https://www.nohrsc.noaa.gov/pub/products/collaborators"
NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02158"

_DIRLIST = {"ts": 0.0, "txt": ""}
_COLLAB_LAST = {"ts": 0.0, "status": None, "error": None, "len": 0}

_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_LOCK = threading.Lock()

_LOG = deque(maxlen=2000)
TILE_CACHE_DIR = (CACHE / "tile_png_cache" / "forecast_by_date")
TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cog_sig(p: Path) -> str:
    st = p.stat()
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"

@lru_cache(maxsize=8000)
def _tile_png_cached(sig: str, z: int, x: int, y: int, max_in: float | None) -> bytes:
    name = sig.split(":", 1)[0]
    p = _forecast_dir().parent / "forecast" / name if name.endswith(".tif") else None
    if p is None or not p.exists():
        raise HTTPException(status_code=500, detail="cached tile missing COG")
    return _tile_png_from_cog(p, z, x, y, max_in)
    
def _lock(key: str) -> threading.Lock:
    with _BUILD_LOCKS_LOCK:
        l = _BUILD_LOCKS.get(key)
        if l is None:
            l = _BUILD_LOCKS[key] = threading.Lock()
        return l

def _tile_cache_get(dom: str, hours: int, ymd: str, z: int, x: int, y: int) -> bytes | None:
    p = _tile_cache_path(dom, hours, ymd, z, x, y)
    try:
        if p.exists() and p.stat().st_size > 0:
            return p.read_bytes()
    except Exception:
        return None
    return None

def _tile_cache_put(dom: str, hours: int, ymd: str, z: int, x: int, y: int, png: bytes) -> None:
    p = _tile_cache_path(dom, hours, ymd, z, x, y)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(png)
        tmp.replace(p)
    except Exception:
        pass

def _generate_forecast_by_date_png(
    *,
    z: int,
    x: int,
    y: int,
    date_yyyymmdd: str,
    hours: int,
    dom: str,
    max_in: float | None,
) -> tuple[bytes, dict]:
    dom = (dom or "zz").lower()
    valid = _norm_ts(f"{date_yyyymmdd}05")

    if int(hours) == 24:
        sel = _pick_forecast_for_valid(valid, dom_prefer=dom)
        if not sel.get("ok"):
            detail = dict(sel)
            detail["collab_last"] = dict(_COLLAB_LAST)
            raise HTTPException(status_code=503, detail=detail)

        # Build melt + snow mask, then render png
        melt_cogs = _build_forecast_melt_cogs(sel["run_init"], sel["valid"], "t0024", sel["melt_urls"])

        snow = snow_valid = None
        snow_info = "snowpack_unavailable"
        if sel.get("snowpack_urls"):
            try:
                snow, snow_valid, snow_info = _read_mask_tile(sel.get("snowpack_ts") or sel["valid"], sel["snowpack_urls"], z, x, y)
            except Exception as e:
                snow_info = f"snowpack_unavailable:{e!r}"

        melt_mm, melt_valid, melt_info = _union_tile(melt_cogs, z, x, y, label="melt")
        if melt_mm is None or melt_valid is None:
            return _transparent_png_256(), {
                "X-Allowed": "1",
                "X-OOB": "1",
                "X-Melt-Info": melt_info,
                "X-SnowMask-Info": snow_info,
                "X-Forecast-Valid": sel["valid"],
                "X-Forecast-RunInit-TT": sel["run_init"],
                "X-Forecast-Hours": "24",
            }

        png = _melt_to_png(
            melt_mm,
            (melt_valid.astype("uint8") * 255),
            max_in,
            snow,
            snow_valid,
            snow_allow_min_mm=0.0,
            snow_underlay_min_mm=0.0001,
            dilate_px=2,
            bin_edges_in=BIN_EDGES_IN,
        )

        return png, {
            "X-Allowed": "1",
            "X-OOB": "0",
            "X-Melt-Info": melt_info,
            "X-SnowMask-Info": snow_info,
            "X-Forecast-Valid": sel["valid"],
            "X-Forecast-RunInit-TT": sel["run_init"],
            "X-Forecast-Hours": "24",
        }

    # 72h
    run_init = _pick_best_run_init_for_valid_t0024(valid, dom_prefer=dom)
    if not run_init:
        raise HTTPException(status_code=503, detail={"error": "no_run_init_for_valid", "valid": valid, "dom": dom})

    melt72_cog = _forecast_melt72h_end_cog(valid, dom_prefer=dom)
    melt_mm, melt_mask, oob = _tile_arrays_from_cog(melt72_cog, z, x, y)
    if oob or melt_mm is None or melt_mask is None:
        return _transparent_png_256(), {
            "X-Allowed": "1",
            "X-OOB": "1",
            "X-Forecast-Hours": "72",
            "X-Forecast-Valid": valid,
            "X-Forecast-RunInit-TT": run_init,
            "X-Forecast-72h-COG": melt72_cog.name,
        }

    snow_urls = _find_fcst_11034_grib2_tt_ts(run_init, valid, dom_prefer=dom)
    snow = snow_valid = None
    snow_info = "snowpack_unavailable"
    if snow_urls:
        try:
            snow_cogs = _build_forecast_snowpack_cogs(valid, snow_urls)
            snow, snow_valid, snow_info = _union_tile(snow_cogs, z, x, y, label="snowpack")
        except Exception as e:
            snow_info = f"snowpack_unavailable:{e!r}"

    png = _melt_to_png(
        melt_mm,
        melt_mask,
        max_in,
        snow,
        snow_valid,
        snow_allow_min_mm=0.0,
        snow_underlay_min_mm=0.0001,
        dilate_px=2,
        bin_edges_in=BIN_EDGES_72H_IN,
    )

    return png, {
        "X-Allowed": "1",
        "X-OOB": "0",
        "X-Forecast-Hours": "72",
        "X-Forecast-Valid": valid,
        "X-Forecast-RunInit-TT": run_init,
        "X-Forecast-72h-COG": melt72_cog.name,
        "X-SnowPack-INFO": snow_info,
    }

def _log(level: str, msg: str) -> None:
    _LOG.append({"ts": int(datetime.utcnow().timestamp() * 1000), "level": level, "msg": msg})
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | {level} | {msg}", flush=True)


_HTTP_SEM = threading.BoundedSemaphore(int(os.environ.get("HTTP_MAX_INFLIGHT", "12")))

_SESSION = requests.Session()
_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
_SESSION.mount("http://", _ADAPTER)
_SESSION.mount("https://", _ADAPTER)

def _retry_get(
    url: str,
    *,
    timeout=(10, 120),
    tries: int = 6,
    backoff: float = 1.6,
    headers=None,
    stream: bool = False,
    allow_redirects: bool = True,
) -> requests.Response:
    hdr = {"User-Agent": "snodas/1.3"}
    if headers:
        hdr.update(headers)

    last_exc = None
    for i in range(int(tries)):
        with _HTTP_SEM:
            try:
                r = _SESSION.get(
                    url,
                    timeout=timeout,
                    allow_redirects=allow_redirects,
                    headers=hdr,
                    stream=stream,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                r = None

        if r is None:
            if i < tries - 1:
                time.sleep((backoff ** i) + random.uniform(0, 0.6))
                continue
            raise last_exc

        if r.status_code in (429, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else None
            except Exception:
                wait = None
            if i < tries - 1:
                if stream:
                    try:
                        r.close()
                    except Exception:
                        pass
                time.sleep((wait if wait is not None else (backoff ** i)) + random.uniform(0, 0.8))
                continue
            return r

        return r

    raise last_exc

app = FastAPI(title="SNODAS Snowmelt Tiles", version="1.3.0")
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

_TS10 = re.compile(r"^\d{10}$")
_TS8U2 = re.compile(r"^(\d{8})[_-](\d{2})$")
_HREF_RE = re.compile(r'href\s*=\s*(?P<q>[\'"])(?P<u>.*?)(?P=q)', re.I)
_TS_IN_NAME = re.compile(r"(?:TS|NATS)(\d{10})", re.I)
_TT_IN_NAME = re.compile(r"TT(\d{10})", re.I)

def _norm_ts(ts: str) -> str:
    ts = (ts or "").strip()
    if _TS10.fullmatch(ts):
        return ts
    m = _TS8U2.fullmatch(ts)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    raise HTTPException(status_code=404, detail="unknown forecast key")

US_BORDER_LAT = 49.0

def _tile2lon(x: int, z: int) -> float:
    return (float(x) / (1 << int(z))) * 360.0 - 180.0

def _tile2lat(y: int, z: int) -> float:
    n = np.pi - (2.0 * np.pi * float(y) / (1 << int(z)))
    return (180.0 / np.pi) * np.arctan(0.5 * (np.exp(n) - np.exp(-n)))

# def _tile_allowed(z: int, x: int, y: int) -> bool:
#     z = int(z); x = int(x); y = int(y)
#     n = 1 << z

#     if y < 0 or y >= n:
#         return True
#     if x < 0 or x >= n:
#         return True

#     lon = _tile2lon(x + 0.5, z)
#     lat = _tile2lat(y + 0.5, z)

#     # Box A: everything north of US border, broad longitudes to capture CN track
#     if (lat >= 49.0) and (-170.0 <= lon <= -40.0):
#         return True

#     # Box B: southern Ontario + Midwest/Great Lakes band (Minnesota->Kansas->Atlantic->Nova Scotia)
#     if (37.0 <= lat <= 49.0) and (-97.0 <= lon <= -63.0):
#         return True

#     return False
def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    z = int(z); x = int(x); y = int(y)
    west = _tile2lon(x, z)
    east = _tile2lon(x + 1, z)
    north = _tile2lat(y, z)
    south = _tile2lat(y + 1, z)
    return west, south, east, north  # (w,s,e,n)

def _box_intersects(a, b) -> bool:
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return not (ae < bw or be < aw or an < bs or bn < as_)

def _tile_allowed(z: int, x: int, y: int) -> bool:
    z = int(z); x = int(x); y = int(y)
    n = 1 << z
    if y < 0 or y >= n or x < 0 or x >= n:
        return True

    tb = _tile_bounds(z, x, y)

    box_a = (-170.0, 49.0, -40.0, 90.0)
    box_b = (-97.0, 37.0, -63.0, 49.0)

    return _box_intersects(tb, box_a) or _box_intersects(tb, box_b)


@lru_cache(maxsize=1)
def _transparent_png_256() -> bytes:
    rgb = np.zeros((3, 256, 256), dtype="uint8")
    a = np.zeros((256, 256), dtype="uint8")
    return render(rgb, mask=a, img_format="PNG")


_REMOTE_KEEP_DAYS = 4

def _hf_cfg() -> tuple[Optional[str], Optional[str]]:
    tok = (os.environ.get("HF_TOKEN") or "").strip()
    repo = (os.environ.get("HF_DATASET_REPO") or "").strip()
    if not tok or not repo:
        return None, None
    return tok, repo

def _hf_api() -> Optional[HfApi]:
    tok, repo = _hf_cfg()
    if not tok or not repo:
        return None
    return HfApi(token=tok)

def _date_ymd_utc(d: datetime.date | None = None) -> str:
    if d is None:
        d = datetime.utcnow().date()
    return d.strftime("%Y%m%d")

def _keep_cutoff_ymd(keep_days: int = _REMOTE_KEEP_DAYS) -> str:
    cut = datetime.utcnow().date() - timedelta(days=int(keep_days) - 1)
    return cut.strftime("%Y%m%d")

def _file_ymd_from_name(name: str) -> Optional[str]:
    m = re.search(r"(?:^|_)(\d{8})(?:_|\.|$)", name)
    return m.group(1) if m else None

def _is_cache_file_we_manage(rel: str) -> bool:
    n = rel.replace("\\", "/")
    if n.startswith("forecast/") and n.endswith("_cog.tif"):
        return True
    if (n.startswith("melt24h_") or n.startswith("melt72h_end_")) and n.endswith("_cog.tif"):
        return True
    if n.startswith("tilecache/") and n.endswith(".png"):
        return True
    return False

def _local_cache_paths_to_consider() -> list[Path]:
    out: list[Path] = []
    for p in CACHE.glob("melt24h_*_cog.tif"):
        out.append(p)
    for p in CACHE.glob("melt72h_end_*_cog.tif"):
        out.append(p)
    fc = _forecast_dir()
    if fc.exists():
        for p in fc.glob("*_cog.tif"):
            out.append(p)
    return out

def _hf_pull_cache() -> None:
    tok, repo = _hf_cfg()
    if not tok or not repo:
        _log("INFO", "[hf_cache] pull skipped (HF_TOKEN/HF_DATASET_REPO unset)")
        return

    try:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(CACHE),
            local_dir_use_symlinks=False,
            token=tok,
            allow_patterns=[
                "melt24h_*_cog.tif",
                "melt72h_end_*_cog.tif",
                "forecast/*_cog.tif",
                "tilecache/*.png",
                "tilecache/**.png",
            ],
        )
        _log("INFO", f"[hf_cache] pull OK repo={repo}")
    except Exception as e:
        _log("INFO", f"[hf_cache] pull FAIL repo={repo} err={e!r}")

def _hf_commit(ops: list, message: str) -> None:
    api = _hf_api()
    tok, repo = _hf_cfg()
    if api is None or not repo:
        return
    if not ops:
        return
    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=ops,
        commit_message=message,
    )

def _hf_commit_with_backoff(ops: list, message: str) -> None:
    api = _hf_api()
    tok, repo = _hf_cfg()
    if api is None or not repo or not ops:
        return

    tries = 6
    base_sleep = 2.0
    for i in range(tries):
        try:
            api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                operations=ops,
                commit_message=message,
            )
            if HF_PUSH_SLEEP_SEC > 0:
                time.sleep(HF_PUSH_SLEEP_SEC)
            return
        except Exception as e:
            s = repr(e).lower()
            is_rate = ("429" in s) or ("rate" in s and "limit" in s) or ("too many requests" in s)
            if (i < tries - 1) and is_rate:
                time.sleep(base_sleep * (2 ** i) + random.uniform(0, 0.25))
                continue
            raise

def _hf_push_tilecache_for_window(window_ymds: list[str], dom: str, hours_list: list[int]) -> dict:
    tok, repo = _hf_cfg()
    api = _hf_api()
    if not tok or not repo or api is None:
        return {"ok": False, "skipped": True, "reason": "hf_not_configured"}

    root = _tile_cache_dir()
    if not root.exists():
        return {"ok": True, "pushed": 0, "ops": 0}

    dom = (dom or "zz").lower()
    want_ymds = set(window_ymds)
    want_hours = set(int(h) for h in hours_list)

    # Collect candidate pngs
    files: list[Path] = []
    for p in root.rglob("bydate_*.png"):
        try:
            n = p.name
            m = re.search(r"bydate_([a-z0-9]+)_h([0-9]+)_([0-9]{8})_z", n)
            if not m:
                continue
            p_dom = m.group(1)
            p_h = int(m.group(2))
            p_ymd = m.group(3)
            if p_dom != dom:
                continue
            if p_h not in want_hours:
                continue
            if p_ymd not in want_ymds:
                continue
            if p.stat().st_size <= 0:
                continue
            files.append(p)
        except Exception:
            continue

    if not files:
        return {"ok": True, "pushed": 0, "ops": 0}

    # Build ops in chunks to avoid massive commits
    pushed = 0
    ops_total = 0
    files.sort(key=lambda p: p.name)

    i = 0
    while i < len(files):
        chunk = files[i : i + HF_MAX_OPS_PER_COMMIT]
        ops = []
        for p in chunk:
            try:
                rel = p.relative_to(CACHE).as_posix()
                if not _is_cache_file_we_manage(rel):
                    continue
                ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p.as_posix()))
            except Exception:
                continue

        if ops:
            _hf_commit_with_backoff(ops, message=f"tilecache: update dom={dom} files={len(ops)}")
            pushed += len(ops)
            ops_total += len(ops)

        i += HF_MAX_OPS_PER_COMMIT

    _log("INFO", f"[hf_cache] tilecache push OK dom={dom} files={pushed}")
    return {"ok": True, "pushed": pushed, "ops": ops_total}

def _hf_push_files(paths: list[Path]) -> None:
    tok, repo = _hf_cfg()
    if not tok or not repo:
        return

    ops = []
    for p in paths:
        try:
            if not p.exists() or p.stat().st_size <= 0:
                continue
            rel = p.relative_to(CACHE).as_posix()
            if not _is_cache_file_we_manage(rel):
                continue
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p.as_posix()))
        except Exception:
            continue

    try:
        _hf_commit(ops, message=f"cache: add/update {len(ops)} files")
        if ops:
            _log("INFO", f"[hf_cache] push OK files={len(ops)}")
    except Exception as e:
        _log("INFO", f"[hf_cache] push FAIL err={e!r}")

def _hf_prune_remote_and_local(keep_days: int = _REMOTE_KEEP_DAYS) -> None:
    cutoff = _keep_cutoff_ymd(keep_days)
    api = _hf_api()
    tok, repo = _hf_cfg()

    local_del: list[Path] = []
    for p in _local_cache_paths_to_consider():
        ymd = _file_ymd_from_name(p.name)
        if ymd and ymd < cutoff:
            local_del.append(p)

    for p in local_del:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    if api is None or not repo:
        if local_del:
            _log("INFO", f"[hf_cache] local prune OK deleted={len(local_del)} cutoff={cutoff}")
        return

    try:
        tree = api.list_repo_tree(repo_id=repo, repo_type="dataset", recursive=True)
    except Exception as e:
        _log("INFO", f"[hf_cache] remote prune list FAIL err={e!r}")
        return

    ops = []
    for item in tree:
        try:
            if getattr(item, "type", "") != "file":
                continue
            rel = getattr(item, "path", "")
            if not rel or not _is_cache_file_we_manage(rel):
                continue
            ymd = _file_ymd_from_name(Path(rel).name)
            if ymd and ymd < cutoff:
                ops.append(CommitOperationDelete(path_in_repo=rel))
        except Exception:
            continue

    try:
        _hf_commit(ops, message=f"cache: prune < {cutoff}")
        _log("INFO", f"[hf_cache] prune OK local_deleted={len(local_del)} remote_deleted={len(ops)} cutoff={cutoff}")
    except Exception as e:
        _log("INFO", f"[hf_cache] remote prune FAIL err={e!r}")

def _ts_to_dt(ts10: str) -> datetime:
    return datetime.strptime(ts10, "%Y%m%d%H").replace(tzinfo=timezone.utc)


def _yyyymmdd(d) -> str:
    return d.strftime("%Y%m%d")
    
def _track_geojson_path() -> Optional[Path]:
    p1 = Path("web") / "cn_tracks.geojson"
    if p1.exists():
        return p1
    p2 = Path(__file__).resolve().parent / "web" / "cn_tracks.geojson"
    if p2.exists():
        return p2
    return None

def _load_track_points(max_points: int = 35000) -> list[tuple[float, float]]:
    p = _track_geojson_path()
    if p is None:
        _log("INFO", "[prewarm] cn_tracks.geojson not found")
        return []

    try:
        gj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _log("INFO", f"[prewarm] failed to read cn_tracks.geojson err={e!r}")
        return []

    pts: list[tuple[float, float]] = []

    def add_coords(coords):
        nonlocal pts
        if not coords:
            return
        if isinstance(coords[0], (int, float)) and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
            if np.isfinite(lon) and np.isfinite(lat):
                pts.append((lon, lat))
            return
        for c in coords:
            add_coords(c)

    feats = gj.get("features") or []
    for f in feats:
        g = f.get("geometry") or {}
        add_coords(g.get("coordinates"))

    if not pts:
        return []

    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    return pts


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    lon = float(lon)
    n = 2 ** int(z)
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = np.deg2rad(lat)
    y = int((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y

def _tiles_near_points(points: list[tuple[float, float]], z: int, radius: int = 1, cap: int = 8000) -> list[tuple[int, int, int]]:
    if not points:
        return []
    r = max(0, int(radius))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, int]] = []
    n = 2 ** int(z)

    for lon, lat in points:
        tx, ty = _lonlat_to_tile(lon, lat, z)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x = tx + dx
                y = ty + dy
                if x < 0 or y < 0 or x >= n or y >= n:
                    continue
                key = (x, y)
                if key in seen:
                    continue
                seen.add(key)
                out.append((z, x, y))
                if len(out) >= cap:
                    return out
    return out

def _prewarm_union_tile(cogs: list[Path], z: int, x: int, y: int) -> bool:
    ok = False
    for cog in sorted(cogs, key=lambda p: _dom_pri(_cog_dom(p))):
        try:
            with COGReader(cog.as_posix()) as r:
                r.tile(x, y, z, tilesize=256, resampling_method="nearest")
            ok = True
            break
        except TileOutsideBounds:
            continue
        except Exception:
            continue
    return ok

def _prewarm_cn_corridor_tiles(
    *,
    days: int = 2,
    hours: int = 24,
    dom: str = "zz",
    zmin: int = 6,
    zmax: int = 11,
    radius: int = 1,
    cap_per_zoom: int = 6000,
    max_tiles_total: int = 20000,
) -> dict:
    dom = (dom or "zz").lower()
    pts = _load_track_points()
    if not pts:
        return {"ok": False, "error": "no_cn_points"}

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    valid = _valid_ts_for_days(int(days), now_utc=now)

    sel = _pick_forecast_for_valid(valid, dom_prefer=dom)
    if not sel.get("ok"):
        return {"ok": False, "error": "forecast_selection_failed", "detail": sel}

    melt_cogs: list[Path] = []
    snow_cogs: list[Path] = []

    if int(hours) == 72:
        melt72 = _forecast_melt72h_end_cog(sel["valid"], dom_prefer=dom)
        melt_cogs = [melt72]
        snow_urls = _find_fcst_11034_grib2_tt_ts(sel["run_init"], sel["valid"], dom_prefer=dom)
        if snow_urls:
            snow_cogs = _build_forecast_snowpack_cogs(sel["valid"], snow_urls)
    else:
        melt_cogs = _build_forecast_melt_cogs(sel["run_init"], sel["valid"], "t0024", sel["melt_urls"])
        snow_urls = _find_fcst_11034_grib2_tt_ts(sel["run_init"], sel["valid"], dom_prefer=dom)
        if snow_urls:
            snow_cogs = _build_forecast_snowpack_cogs(sel["valid"], snow_urls)

    total = 0
    warmed_melt = 0
    warmed_snow = 0

    for z in range(int(zmin), int(zmax) + 1):
        tiles = _tiles_near_points(pts, z, radius=radius, cap=cap_per_zoom)
        for (zz, x, y) in tiles:
            if total >= int(max_tiles_total):
                return {
                    "ok": True,
                    "stopped": "max_tiles_total",
                    "total": total,
                    "warmed_melt": warmed_melt,
                    "warmed_snow": warmed_snow,
                    "zmin": zmin,
                    "zmax": zmax,
                    "radius": radius,
                    "days": days,
                    "hours": hours,
                    "valid": sel["valid"],
                }

            if _prewarm_union_tile(melt_cogs, zz, x, y):
                warmed_melt += 1
            if snow_cogs:
                if _prewarm_union_tile(snow_cogs, zz, x, y):
                    warmed_snow += 1
            total += 1

    return {
        "ok": True,
        "total": total,
        "warmed_melt": warmed_melt,
        "warmed_snow": warmed_snow,
        "zmin": zmin,
        "zmax": zmax,
        "radius": radius,
        "days": days,
        "hours": hours,
        "valid": sel["valid"],
    }


def _collab_dirlist(ttl_sec: float = 240.0) -> str:
    now = datetime.utcnow().timestamp()
    if _DIRLIST["txt"] and (now - float(_DIRLIST["ts"])) < float(ttl_sec):
        return _DIRLIST["txt"]

    url = COLLAB_BASE + "/"
    try:
        r = _retry_get(
            url,
            timeout=(10, 90),
            tries=3,
            headers={"User-Agent": "snodas-forecast/1.3"},
        )

        _COLLAB_LAST.update({"ts": now, "status": r.status_code, "error": None, "len": len(r.text or "")})
        if r.status_code == 200 and (r.text or "").strip():
            _DIRLIST["ts"] = now
            _DIRLIST["txt"] = r.text or ""
            return _DIRLIST["txt"]
        _COLLAB_LAST["error"] = f"HTTP {r.status_code} (empty={not bool((r.text or '').strip())})"
        _log("INFO", f"[collab] dirlist bad: {_COLLAB_LAST['error']}")
        return ""
    except Exception as e:
        _COLLAB_LAST.update({"ts": now, "status": None, "error": repr(e), "len": 0})
        _log("INFO", f"[collab] dirlist EXC: {repr(e)}")
        return ""


def _hrefs(html: str) -> list[str]:
    return [m.group("u").strip() for m in _HREF_RE.finditer(html or "") if (m.group("u") or "").strip()]


def _ts_from_name(name: str) -> Optional[str]:
    m = _TS_IN_NAME.search(Path(name).name)
    return m.group(1) if m else None


def _tt_from_name(name: str) -> Optional[str]:
    m = _TT_IN_NAME.search(Path(name).name)
    return m.group(1) if m else None

def _dom_from_url(url: str) -> str:
    n = Path(url).name.lower()
    return (n.split("_", 1)[0] if "_" in n else n[:2]).strip() or "xx"


def _dom_pri(dom: str) -> int:
    d = (dom or "").lower()
    if d == "zz":
        return 0
    if d == "dp":
        return 1
    if d == "qc":
        return 2
    if d == "us":
        return 3
    return 9


def _as_url(n: str) -> str:
    return n if n.startswith("http") else f"{COLLAB_BASE}/{n}"


def _score_dom(url: str, dom_prefer: str) -> int:
    d = _dom_from_url(url)
    return 0 if d == dom_prefer else (_dom_pri(d) + 10)


def _valid_ts_for_days(days: int, now_utc: Optional[datetime] = None) -> str:
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    d = (now_utc.date() + timedelta(days=int(days)))
    return f"{d.strftime('%Y%m%d')}05"


def _is_data_name(low: str) -> bool:
    low = low.lower()
    return low.endswith(".nc") or low.endswith(".grz") or low.endswith(".grib2")

def _list_available_valid_ts_t0024(dom_prefer: str = "zz") -> list[str]:
    dom_prefer = (dom_prefer or "zz").lower()
    html = _collab_dirlist()
    names = _hrefs(html)

    ts_set: set[str] = set()
    for n in names:
        base = n.split("?", 1)[0].strip()
        low = base.lower()
        if "ssmv11044" not in low:
            continue
        if "__t0024tt" not in low:
            continue
        if not low.endswith(".nc"):
            continue
        ts = _ts_from_name(base)
        if ts and _TS10.fullmatch(ts):
            ts_set.add(ts)

    out = sorted(ts_set)
    return out

def _lon0360_to_180(x: np.ndarray) -> np.ndarray:
    return ((x + 180.0) % 360.0) - 180.0

def _best_state_nc_near_valid(
    code: str,
    valid: str,
    dom_prefer: str = "zz",
    max_after_h: float = 18.0,
    max_before_h: float = 240.0,
) -> dict:
    code = (code or "").lower()
    dom_prefer = (dom_prefer or "zz").lower()
    valid = _norm_ts(valid)

    html = _collab_dirlist()
    names = _hrefs(html)

    cands = []
    for n in names:
        base = n.split("?", 1)[0].strip()
        low = base.lower()
        if code not in low:
            continue
        if "__t0001ttnats" not in low:
            continue
        if "hp001" not in low:
            continue
        if not low.endswith(".nc"):
            continue

        ts = _ts_from_name(base)
        if not ts or not _TS10.fullmatch(ts):
            continue

        url = _as_url(base)
        try:
            dh = (_ts_to_dt(ts) - _ts_to_dt(valid)).total_seconds() / 3600.0
        except Exception:
            continue

        dom = _dom_from_url(url)
        cands.append((dh, ts, dom, url))

    if not cands:
        return {"picked_ts": None, "url": None, "delta_h": None}

    # 1) Prefer matching domain if any exist
    preferred = [r for r in cands if (r[2] or "").lower() == dom_prefer]
    pool = preferred if preferred else cands

    before = [r for r in pool if r[0] <= 0]
    after = [r for r in pool if r[0] > 0]

    pick = None
    if before:
        before.sort(key=lambda t: abs(t[0]))
        if abs(before[0][0]) <= float(max_before_h):
            pick = before[0]
    if pick is None and after:
        after.sort(key=lambda t: t[0])
        if after[0][0] <= float(max_after_h):
            pick = after[0]

    if pick is None:
        pool.sort(key=lambda t: abs(t[0]))
        pick = pool[0]

    dh, ts, dom, url = pick
    return {"picked_ts": ts, "url": url, "delta_h": dh, "dom": dom}

def _find_fcst_11034_grib2_tt_ts(run_init: str, valid: str, dom_prefer: str = "zz") -> list[str]:
    run_init = _norm_ts(run_init)
    valid = _norm_ts(valid)
    dom_prefer = (dom_prefer or "zz").lower()

    html = _collab_dirlist()
    names = _hrefs(html)

    want_tt = f"tt{run_init}"
    want_ts = f"ts{valid}"

    out = []
    for n in names:
        base = n.split("?", 1)[0].strip()
        low = base.lower()

        if "ssmv11034" not in low:
            continue
        if "__t0001tt" not in low:
            continue
        if "hp001" not in low:
            continue
        if want_tt not in low or want_ts not in low:
            continue
        if not low.endswith(".grib2"):
            continue

        out.append(_as_url(base))

    return sorted(
        set(out),
        key=lambda u: (0 if _dom_from_url(u) == dom_prefer else _dom_pri(_dom_from_url(u)) + 10),
    )

def _pick_best_run_init_for_valid_t0024(valid: str, dom_prefer: str = "zz") -> Optional[str]:
    valid = _norm_ts(valid)
    dom_prefer = (dom_prefer or "zz").lower()
    html = _collab_dirlist()
    names = _hrefs(html)
    want_ts = f"ts{valid}"
    rows = []
    for n in names:
        base = n.split("?", 1)[0].strip()
        low = base.lower()
        if "ssmv11044" not in low:
            continue
        if "__t0024tt" not in low:
            continue
        if want_ts not in low:
            continue
        if not low.endswith(".nc"):
            continue
        tt = _tt_from_name(base)
        if not tt:
            continue
        url = _as_url(base)
        rows.append((tt, _score_dom(url, dom_prefer)))
    if not rows:
        return None
    rows.sort(key=lambda t: (t[0], -t[1]))
    return rows[-1][0]


def _find_melt_11044_t0024(run_init: str, valid: str, dom_prefer: str = "zz") -> list[str]:
    run_init = _norm_ts(run_init)
    valid = _norm_ts(valid)
    dom_prefer = (dom_prefer or "zz").lower()
    html = _collab_dirlist()
    names = _hrefs(html)
    want_tt = f"tt{run_init}"
    want_ts = f"ts{valid}"
    out = []
    for n in names:
        base = n.split("?", 1)[0].strip()
        low = base.lower()
        if "ssmv11044" not in low:
            continue
        if "__t0024tt" not in low:
            continue
        if want_tt not in low or want_ts not in low:
            continue
        if not low.endswith(".nc"):
            continue
        out.append(_as_url(base))
    out = sorted(
        set(out),
        key=lambda u: (0 if _dom_from_url(u) == dom_prefer else _dom_pri(_dom_from_url(u)) + 10),
    )
    return out

def _pick_forecast_for_valid(valid: str, dom_prefer: str = "zz") -> dict:
    valid = _norm_ts(valid)
    dom_prefer = (dom_prefer or "zz").lower()

    run_init = _pick_best_run_init_for_valid_t0024(valid, dom_prefer=dom_prefer)
    if not run_init:
        return {"ok": False, "valid": valid, "error": "no melt run_init found (t0024)"}

    melt_urls = _find_melt_11044_t0024(run_init, valid, dom_prefer=dom_prefer)
    if not melt_urls:
        return {"ok": False, "run_init": run_init, "valid": valid, "error": "melt file not found for run_init+valid"}

    snowpack_urls = _find_fcst_11034_grib2_tt_ts(run_init, valid, dom_prefer=dom_prefer)
    if not snowpack_urls:
        return {
            "ok": False,
            "run_init": run_init,
            "valid": valid,
            "melt_urls": melt_urls,
            "error": "no forecasted snowpack (11034 grib2) found for TT+TS",
        }

    return {
        "ok": True,
        "run_init": run_init,
        "valid": valid,
        "lead": "t0024",
        "melt_urls": melt_urls,
        "snowpack_ts": valid,
        "snowpack_urls": snowpack_urls,
    }

def _pick_forecast_for_days(days: int, dom_prefer: str = "zz") -> dict:
    if days not in (2, 3):
        raise HTTPException(status_code=400, detail="days must be 2 or 3")
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    valid = _valid_ts_for_days(days, now_utc=now)
    sel = _pick_forecast_for_valid(valid, dom_prefer=(dom_prefer or "zz").lower())
    if sel.get("ok"):
        sel["days"] = days
        sel["now_utc"] = now.isoformat()
    return sel


def _nsidc_month_folder(d) -> str:
    m = d.month
    return f"{m:02d}_{calendar.month_abbr[m].title()}"


def _nsidc_url(date, masked: bool) -> str:
    sub = "masked" if masked else "unmasked"
    dstr = _yyyymmdd(date)
    return f"{NSIDC_BASE}/{date.year}/{_nsidc_month_folder(date)}/{dstr}/SNODAS_{sub}_{dstr}.tar"


def _nsidc_exists(date) -> tuple[bool, str]:
    hdr = {"User-Agent": "snodas/1.3"}
    for masked in (False, True):
        url = _nsidc_url(date, masked)
        try:
            r = requests.head(url, timeout=20, allow_redirects=True, headers=hdr)
            if r.status_code == 200:
                return True, url
            if r.status_code in (403, 405):
                rg = _retry_get(url, stream=True, timeout=(10, 60), tries=2, headers=hdr)
                rg.close()
                if rg.status_code == 200:
                    return True, url
        except Exception:
            pass
    return False, _nsidc_url(date, masked=False)

@lru_cache(maxsize=512)
def _nsidc_exists_cached(dstr: str) -> tuple[bool, str]:
    d = datetime.strptime(dstr, "%Y%m%d").date()
    return _nsidc_exists(d)
    
def _http_get(url: str, timeout: int = 180) -> bytes:
    r = _retry_get(url, timeout=(10, timeout), tries=3, headers={"User-Agent": "snodas/1.3"})
    r.raise_for_status()
    return r.content


@lru_cache(maxsize=1)
def _resolve_latest_nsidc_day(days_back: int = 120) -> Optional[str]:
    utc_today = datetime.utcnow().date()
    for i in range(days_back + 1):
        d = utc_today - timedelta(days=i)
        ok, _ = _nsidc_exists(d)
        if ok:
            return _yyyymmdd(d)
    return None


def _download_day_tar(date) -> Path:
    dstr = _yyyymmdd(date)
    out = CACHE / f"SNODAS_{dstr}.tar"
    if out.exists() and out.stat().st_size > 0:
        return out
    ok, url = _nsidc_exists_cached(dstr)
    if not ok:
        raise HTTPException(status_code=404, detail=f"NSIDC not available for {dstr}")
    with _lock(f"nsidc_tar_{dstr}"):
        if out.exists() and out.stat().st_size > 0:
            return out
        out.write_bytes(_http_get(url, timeout=180))
    return out

_KV_RE = re.compile(r"^\s*(?P<k>[^:=#;]+?)\s*(?:[:=]\s*|\s+)(?P<v>.+?)\s*$")


def _parse_hdr(path: Path) -> tuple[dict, str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    meta: dict[str, str] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        m = _KV_RE.match(line)
        if not m:
            continue
        k = m.group("k").strip().lower()
        v = m.group("v").strip().split("#", 1)[0].split(";", 1)[0].strip()
        meta[k] = v
        meta[re.sub(r"[^a-z0-9]+", "", k)] = v
    return meta, txt


def _dtype_from_hdr(nbits: int, pixeltype: str, byteorder: str) -> np.dtype:
    pixeltype = (pixeltype or "").upper()
    byteorder = (byteorder or "I").upper()
    endian = "<" if byteorder in ("I", "LSBFIRST", "LITTLEENDIAN") else ">"
    if pixeltype == "FLOAT":
        return np.dtype(endian + ("f4" if nbits <= 32 else "f8"))
    if pixeltype in ("UNSIGNEDINT", "UNSIGNED", "UINTEGER"):
        return np.dtype(endian + ("u1" if nbits <= 8 else "u2" if nbits <= 16 else "u4"))
    if pixeltype in ("SIGNEDINT", "SIGNED", "INTEGER"):
        return np.dtype(endian + ("i1" if nbits <= 8 else "i2" if nbits <= 16 else "i4"))
    return np.dtype(endian + "f4")


def _bil_to_geotiff(hdr: Path, bil: Path, out_tif: Path) -> Path:
    meta, raw = _parse_hdr(hdr)
    raw_l = raw.lower()

    def gi(*keys, default=None):
        for k in keys:
            kk = re.sub(r"[^a-z0-9]+", "", k.lower())
            for cand in (k.lower(), kk):
                v = meta.get(cand)
                if v is None:
                    continue
                try:
                    return int(float(str(v).strip()))
                except Exception:
                    pass
        return default

    def gf(*keys, default=None):
        for k in keys:
            kk = re.sub(r"[^a-z0-9]+", "", k.lower())
            for cand in (k.lower(), kk):
                v = meta.get(cand)
                if v is None:
                    continue
                try:
                    return float(str(v).strip())
                except Exception:
                    pass
        return default

    def re_float(pat: str) -> Optional[float]:
        m = re.search(pat, raw, re.I)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    ncols = gi("ncols", "samples", "nsamples", "columns")
    nrows = gi("nrows", "lines", "nlines", "rows")
    if ncols is None:
        m = re.search(r"Number of columns:\s*([0-9]+)", raw, re.I)
        ncols = int(m.group(1)) if m else None
    if nrows is None:
        m = re.search(r"Number of rows:\s*([0-9]+)", raw, re.I)
        nrows = int(m.group(1)) if m else None
    if ncols is None or nrows is None:
        raise HTTPException(status_code=502, detail="BIL header missing ncols/nrows")

    bands = gi("bands", "nbands", "number of bands", default=1) or 1
    nbits = gi("nbits", "bitspersample", "bits per sample", default=32) or 32
    dbpp = gi("data bytes per pixel", default=None)
    if dbpp:
        nbits = int(dbpp) * 8

    pixeltype = meta.get("pixeltype") or meta.get("datatype") or "FLOAT"
    byteorder = meta.get("byteorder") or "I"
    slope = gf("data slope", "slope", default=1.0) or 1.0
    intercept = gf("data intercept", "intercept", default=0.0) or 0.0
    nodata_raw = gf("no data value", "nodata", "nodata_value", default=None)

    ulx = gf("ulxmap", "upper left x", "ulx", default=None)
    uly = gf("ulymap", "upper left y", "uly", default=None)
    xll = gf("xllcorner", "xllcenter", "lower left x", "xll", default=None)
    yll = gf("yllcorner", "yllcenter", "lower left y", "yll", default=None)

    xres = gf("x-axis resolution", "x axis resolution", "xdim", "x dim", "cellsize", default=None)
    yres = gf("y-axis resolution", "y axis resolution", "ydim", "y dim", "cellsize", default=None)
    if xres is None:
        xres = re_float(r"X[- ]axis resolution:\s*([0-9.eE+-]+)")
    if yres is None:
        yres = re_float(r"Y[- ]axis resolution:\s*([0-9.eE+-]+)")

    xmin = gf("minimum x-axis coordinate", "min x", "minx", default=None)
    xmax = gf("maximum x-axis coordinate", "max x", "maxx", default=None)
    ymin = gf("minimum y-axis coordinate", "min y", "miny", default=None)
    ymax = gf("maximum y-axis coordinate", "max y", "maxy", "maximumy", default=None)
    if xmin is None:
        xmin = re_float(r"Minimum x[- ]axis coordinate:\s*([0-9.eE+-]+)")
    if xmax is None:
        xmax = re_float(r"Maximum x[- ]axis coordinate:\s*([0-9.eE+-]+)")
    if ymin is None:
        ymin = re_float(r"Minimum y[- ]axis coordinate:\s*([0-9.eE+-]+)")
    if ymax is None:
        ymax = re_float(r"Maximum y[- ]axis coordinate:\s*([0-9.eE+-]+)")

    if xres is None and xmin is not None and xmax is not None:
        xres = (xmax - xmin) / float(ncols)
    if yres is None and ymin is not None and ymax is not None:
        yres = (ymax - ymin) / float(nrows)

    if ulx is None and xll is not None:
        ulx = float(xll)
    if uly is None and yll is not None and yres is not None:
        uly = float(yll) + abs(float(yres)) * float(nrows)
    if ulx is None and xmin is not None:
        ulx = float(xmin)
    if uly is None and ymax is not None:
        uly = float(ymax)

    if ulx is None or uly is None or xres is None or yres is None:
        raise HTTPException(status_code=502, detail="BIL header missing geo params")

    units_line = (meta.get("data units") or meta.get("dataunits") or "").strip()
    denom = None
    m = re.search(r"Data units:\s*Meters\s*/\s*([0-9.]+)", raw, re.I)
    if m:
        try:
            denom = float(m.group(1))
        except Exception:
            denom = None
    if denom is None and units_line and "/" in units_line:
        try:
            denom = float(units_line.split("/", 1)[1].strip())
        except Exception:
            denom = None

    is_nohrsc_gisrs = "nohrsc gis/rs raster file" in raw_l

    file_bytes = bil.stat().st_size
    bytes_per_sample = max(1, int(nbits) // 8)
    expected_samples = int(nrows) * int(ncols) * max(1, int(bands))
    expected_bytes = expected_samples * bytes_per_sample

    if is_nohrsc_gisrs:
        dtp = np.dtype(">i2")
        expected_bytes = expected_samples * 2
    else:
        dtp = _dtype_from_hdr(int(nbits), str(pixeltype), str(byteorder))

    if file_bytes != expected_bytes:
        inferred = None
        for bps in (1, 2, 4):
            if file_bytes == expected_samples * bps:
                inferred = bps
                break
        if inferred is None:
            raise HTTPException(status_code=502, detail=f"BIL byte mismatch file={file_bytes} expected={expected_bytes}")
        if is_nohrsc_gisrs and inferred != 2:
            raise HTTPException(status_code=502, detail="NOHRSC GIS/RS expected 2 bytes/sample")
        if not is_nohrsc_gisrs:
            if inferred == 2:
                dtp = np.dtype("<i2")
            elif inferred == 4:
                dtp = np.dtype("<f4")

    arr = np.fromfile(bil, dtype=dtp)
    if arr.size != expected_samples:
        raise HTTPException(status_code=502, detail=f"BIL size mismatch expected={expected_samples} got={arr.size}")
    if int(bands) == 1:
        arr = arr.reshape((int(nrows), int(ncols)))
    else:
        arr = arr.reshape((int(bands), int(nrows), int(ncols)))

    arr = arr.astype("float32")
    if slope != 1.0 or intercept != 0.0:
        arr = arr * float(slope) + float(intercept)

    if is_nohrsc_gisrs:
        if denom is None or denom <= 0:
            raise HTTPException(status_code=502, detail="NOHRSC GIS/RS missing units denom")
        arr = arr * (1000.0 / float(denom))
        nodata = (float(nodata_raw) * (1000.0 / float(denom))) if nodata_raw is not None else None
    else:
        den = 1.0
        if units_line and "/" in units_line:
            try:
                den = float(units_line.split("/", 1)[1].strip()) or 1.0
            except Exception:
                den = 1.0
        if den != 1.0:
            arr = arr / float(den)
        if "meter" in (units_line or "").lower():
            arr = arr * 1000.0
        nodata = float(nodata_raw) if nodata_raw is not None else None

    arr[arr <= -9990.0] = nodata if nodata is not None else np.nan

    crs = CRS.from_epsg(4326)
    transform = Affine(float(xres), 0.0, float(ulx), 0.0, -abs(float(yres)), float(uly))

    profile = {
        "driver": "GTiff",
        "height": int(nrows),
        "width": int(ncols),
        "count": 1 if arr.ndim == 2 else int(arr.shape[0]),
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
        "tiled": True,
    }
    if nodata is not None:
        profile["nodata"] = float(nodata)

    with rasterio.open(out_tif, "w", **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
        else:
            for b in range(arr.shape[0]):
                dst.write(arr[b], b + 1)

    return out_tif


def _sanitize_mm(da: xr.DataArray) -> xr.DataArray:
    da = da.astype("float32")
    da = da.where(np.isfinite(da)).where(da > -9990.0)
    da = xr.where(da < 0.0, 0.0, da)
    da = xr.where(da > 100000.0, np.nan, da)
    return da

def _open_grib2(path: Path) -> xr.DataArray:
    idx = (CFGRIB_DIR / (path.name + ".idx")).as_posix()
    ds = xr.open_dataset(path.as_posix(), engine="cfgrib", backend_kwargs={"indexpath": idx})

    try:
        da = _pick_grib_var(ds)

        if "step" in da.dims:
            da = da.isel(step=0)

        attrs = dict(da.attrs or {})
        for k in ("_FillValue", "missingValue", "missing_value"):
            if k in attrs:
                try:
                    mv = float(attrs[k])
                    da = da.where(da != mv)
                except Exception:
                    pass

        da = da.where(np.isfinite(da)).where(da < 1.0e20).astype("float32")

        units = str(attrs.get("units", "")).strip().lower()
        if units in ("m", "meter", "metre", "m of water equivalent", "meters"):
            da = da * 1000.0

        dims_l = {d.lower(): d for d in da.dims}
        rename = {}
        if "longitude" in dims_l:
            rename[dims_l["longitude"]] = "x"
        if "latitude" in dims_l:
            rename[dims_l["latitude"]] = "y"
        if rename:
            da = da.rename(rename)

        if "x" in da.coords:
            try:
                x = np.asarray(da["x"].values)
                if x.ndim == 1 and x.size > 2:
                    xmin = float(np.nanmin(x))
                    xmax = float(np.nanmax(x))
                    if xmin >= 0.0 and xmax > 180.0:
                        da = da.assign_coords(x=_lon0360_to_180(x))
                        da = da.sortby("x")
            except Exception:
                pass

        if "x" in da.dims and "y" in da.dims:
            try:
                da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            except Exception:
                pass

            try:
                da = da.rio.write_crs("EPSG:4326", inplace=False)
            except Exception:
                pass

            try:
                da = _ensure_xy_transform(da)
            except Exception:
                pass

        da = _sanitize_mm(da)
        return da

    finally:
        try:
            ds.close()
        except Exception:
            pass



def _open_bil(hdr: Path, dat: Path) -> xr.DataArray:
    tmp_tif = dat.with_suffix(".tif")
    if not tmp_tif.exists() or tmp_tif.stat().st_size == 0:
        _bil_to_geotiff(hdr, dat, tmp_tif)
    da = xr.open_dataarray(tmp_tif.as_posix(), engine="rasterio").astype("float32")
    try:
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326", inplace=False)
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    except Exception:
        pass
    return _sanitize_mm(da)


def _pick_netcdf_raster_var(ds: xr.Dataset) -> xr.DataArray:
    bad = ("bounds", "bnds", "crs", "grid_mapping", "lat_bounds", "lon_bounds", "time_bounds")
    bad2 = ("mask", "qc", "quality", "flag", "prob", "probability", "index", "idx")
    good = ("melt", "snowmelt", "smelt", "we", "water", "equiv", "mm", "kg", "swe")

    best = None
    best_score = -10**9

    for name, da in ds.data_vars.items():
        n = name.lower()
        if any(t in n for t in bad) or any(t in n for t in bad2):
            continue
        if da.ndim < 2:
            continue

        score = 0
        if any(t in n for t in good):
            score += 300
        score += 200 if da.ndim == 2 else 50

        dims = [d.lower() for d in da.dims]
        if ("y" in dims and "x" in dims) or ("south_north" in dims and "west_east" in dims):
            score += 200
        if ("lat" in dims and "lon" in dims) or ("latitude" in dims and "longitude" in dims):
            score += 150

        try:
            score += int((da.shape[-1] * da.shape[-2]) // 50000)
        except Exception:
            pass

        if score > best_score:
            best_score = score
            best = da

    if best is None:
        raise HTTPException(status_code=502, detail=f"No suitable raster variable in NetCDF. vars={list(ds.data_vars)[:50]}")
    return best


def _open_netcdf(path: Path) -> xr.DataArray:
    last = None
    ds = None
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds = xr.open_dataset(path.as_posix(), engine=eng)
            break
        except Exception as e:
            last = e
            ds = None
    if ds is None:
        raise HTTPException(status_code=502, detail=f"NetCDF backend unavailable/failed. last_err={last!r}")

    try:
        da = _pick_netcdf_raster_var(ds)

        dims_l = {d.lower(): d for d in da.dims}
        rename = {}
        if "south_north" in dims_l and "west_east" in dims_l:
            rename[dims_l["south_north"]] = "y"
            rename[dims_l["west_east"]] = "x"
        elif "lat" in dims_l and "lon" in dims_l:
            rename[dims_l["lat"]] = "y"
            rename[dims_l["lon"]] = "x"
        elif "latitude" in dims_l and "longitude" in dims_l:
            rename[dims_l["latitude"]] = "y"
            rename[dims_l["longitude"]] = "x"
        elif "rlat" in dims_l and "rlon" in dims_l:
            rename[dims_l["rlat"]] = "y"
            rename[dims_l["rlon"]] = "x"
        if rename:
            da = da.rename(rename)

        spatial = {"x", "y"}
        for d in list(da.dims):
            if d not in spatial:
                n = int(da.sizes.get(d, 1) or 1)
                da = da.isel({d: n - 1})

        da = da.astype("float32")
        units = str(da.attrs.get("units", "")).strip().lower()
        if units in ("m", "meter", "metre", "m of water equivalent", "meters"):
            da = da * 1000.0

        try:
            lon = da.coords.get("longitude") or da.coords.get("lon")
            lat = da.coords.get("latitude") or da.coords.get("lat")
            if lon is not None and lat is not None:
                if lon.ndim == 2 and lat.ndim == 2:
                    lon_min, lon_max = float(np.nanmin(lon.values)), float(np.nanmax(lon.values))
                    lat_min, lat_max = float(np.nanmin(lat.values)), float(np.nanmax(lat.values))
                    h, w = da.shape[-2], da.shape[-1]
                    tfm = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
                    da = da.rio.write_crs("EPSG:4326", inplace=False).rio.write_transform(tfm, inplace=False)
                else:
                    da = da.rio.write_crs("EPSG:4326", inplace=False)
        except Exception:
            pass

        try:
            if "x" in da.dims and "y" in da.dims:
                da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        except Exception:
            pass

        return _sanitize_mm(da)
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _open_grid(desc: dict) -> xr.DataArray:
    t = desc.get("type")
    if t == "grib2":
        return _open_grib2(desc["path"])
    if t == "bil":
        return _open_bil(desc["hdr"], desc["dat"])
    if t == "netcdf":
        return _open_netcdf(desc["path"])
    raise HTTPException(status_code=500, detail="unknown grid type")


def _is_probably_html(blob: bytes) -> bool:
    head = blob[:512].lstrip()
    return head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or head.startswith(b"<HTML")


def _is_netcdf_bytes(blob: bytes) -> bool:
    if len(blob) >= 4 and blob[:3] == b"CDF" and blob[3] in (1, 2):
        return True
    return len(blob) >= 8 and blob[:8] == b"\x89HDF\r\n\x1a\n"
_HF_OPS_LOCK = threading.Lock()
_HF_PENDING: dict[str, CommitOperationAdd] = {}

def _hf_enqueue_files(paths: list[Path]) -> int:
    tok, repo = _hf_cfg()
    if not tok or not repo:
        return 0

    added = 0
    with _HF_OPS_LOCK:
        for p in paths:
            try:
                if (not p.exists()) or p.stat().st_size <= 0:
                    continue
                rel = p.relative_to(CACHE).as_posix()
                if not _is_cache_file_we_manage(rel):
                    continue
                _HF_PENDING[rel] = CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p.as_posix())
                added += 1
            except Exception:
                continue
    return added

def _hf_flush_pending(message: str) -> None:
    tok, repo = _hf_cfg()
    api = _hf_api()
    if not tok or not repo or api is None:
        return

    with _HF_OPS_LOCK:
        ops = list(_HF_PENDING.values())
        _HF_PENDING.clear()

    if not ops:
        return

    for attempt in range(6):
        try:
            api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message=message)
            _log("INFO", f"[hf_cache] flush OK files={len(ops)}")
            return
        except Exception as e:
            wait = (1.8 ** attempt) + random.uniform(0, 1.0)
            _log("INFO", f"[hf_cache] flush retry attempt={attempt+1} wait={wait:.1f} err={e!r}")
            time.sleep(wait)

    _log("INFO", f"[hf_cache] flush FAILED files={len(ops)}")


def _download_url_to_grid(url: str, workdir: Path) -> tuple[dict, str]:
    hdrs = {"User-Agent": "snodas/1.3"}
    name = Path(url).name.lower()

    last_err = None
    for attempt in range(6):
        r = _retry_get(url, timeout=(10, 180), tries=1, headers=hdrs)
        sc = int(r.status_code)

        if sc in (429, 502, 503, 504):
            last_err = f"HTTP {sc}"
            if attempt < 5:
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else None
                except Exception:
                    wait = None
                time.sleep((wait if wait is not None else (1.6 ** attempt)) + random.uniform(0, 0.8))
                continue
            raise HTTPException(status_code=502, detail=f"{url} -> {last_err} after retries")

        if sc not in (200, 206):
            raise HTTPException(status_code=404, detail=f"{url} -> HTTP {sc}")

        blob = r.content
        ctype = (r.headers.get("Content-Type") or "").lower()

        if _is_probably_html(blob) or "text/html" in ctype:
            sample = blob[:300].decode("utf-8", "ignore")
            raise HTTPException(
                status_code=502,
                detail=f"Upstream returned HTML (not data). url={url} content_type={ctype!r} sample={sample!r}",
            )

        if len(blob) >= 2 and blob[:2] == b"\x1f\x8b":
            try:
                blob = gzip.decompress(blob)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"gzip decompress failed: {e!r}")

        if blob.startswith(b"GRIB"):
            p = workdir / "data.grib2"
            p.write_bytes(blob)
            return {"type": "grib2", "path": p}, name

        if _is_netcdf_bytes(blob):
            p = workdir / "data.nc"
            p.write_bytes(blob)
            return {"type": "netcdf", "path": p}, name

        looks_tar = (len(blob) % 512 == 0) or (b"ustar" in blob[:4096])
        if looks_tar:
            try:
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
                    grib_member = hdr_member = dat_member = nc_member = None
                    for m in tf.getmembers():
                        n = m.name.lower()
                        if n.endswith(".grib2") and grib_member is None:
                            grib_member = m
                        elif n.endswith(".nc") and nc_member is None:
                            nc_member = m
                        elif n.endswith(".hdr") and hdr_member is None:
                            hdr_member = m
                        elif (n.endswith(".dat") or n.endswith(".bil")) and dat_member is None:
                            dat_member = m

                    if grib_member is not None:
                        p = workdir / "data.grib2"
                        f = tf.extractfile(grib_member)
                        if f is None:
                            raise HTTPException(status_code=502, detail="TAR member unreadable")
                        p.write_bytes(f.read(grib_member.size))
                        return {"type": "grib2", "path": p}, name

                    if nc_member is not None:
                        p = workdir / "data.nc"
                        f = tf.extractfile(nc_member)
                        if f is None:
                            raise HTTPException(status_code=502, detail="TAR member unreadable")
                        p.write_bytes(f.read(nc_member.size))
                        return {"type": "netcdf", "path": p}, name

                    if hdr_member is not None and dat_member is not None:
                        hdr = workdir / "data.hdr"
                        dat = workdir / "data.bil"
                        fh = tf.extractfile(hdr_member)
                        fd = tf.extractfile(dat_member)
                        if fh is None or fd is None:
                            raise HTTPException(status_code=502, detail="TAR member unreadable")
                        hdr.write_bytes(fh.read(hdr_member.size))
                        dat.write_bytes(fd.read(dat_member.size))
                        return {"type": "bil", "hdr": hdr, "dat": dat}, name
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"TAR parse failed: {e!r}")

        head_hex = blob[:16].hex()
        sample_txt = blob[:200].decode("utf-8", "ignore")
        raise HTTPException(
            status_code=502,
            detail=f"payload unrecognized. url={url} content_type={ctype!r} head_hex={head_hex} sample={sample_txt!r}",
        )

    raise HTTPException(status_code=504, detail=f"upstream retries exhausted. url={url} last_err={last_err}")

def _da_to_cog(da: xr.DataArray, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        _ = da.rio.transform()
        ok_transform = True
    except Exception:
        ok_transform = False

    if not ok_transform:
        if "x" in da.coords and "y" in da.coords:
            try:
                da = _ensure_xy_transform(da)
            except Exception:
                pass

    nodata = np.float32(-9999.0)
    try:
        crs = da.rio.crs
    except Exception:
        crs = None
    if crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)
    da = da.astype("float32").where(np.isfinite(da)).where(da > -9990.0)
    da = da.where(~da.isnull(), nodata)

    with tempfile.NamedTemporaryFile(suffix=".tif", dir=str(out.parent), delete=False) as tf:
        tmp = Path(tf.name)
    try:
        da.rio.to_raster(tmp.as_posix(), dtype="float32", compress="DEFLATE", nodata=float(nodata), tiled=True)
        prof = cog_profiles.get("deflate")
        if prof is None:
            raise HTTPException(status_code=500, detail="rio_cogeo deflate profile missing")
        cog_translate(tmp.as_posix(), out.as_posix(), prof, in_memory=False, quiet=True)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _extract_var(tar_path: Path, code: str) -> tuple[Path, Path]:
    outdir = CACHE / f"{tar_path.stem}_x"
    outdir.mkdir(exist_ok=True)
    hdr_path: Optional[Path] = None
    dat_path: Optional[Path] = None
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            low = name.lower()
            if code not in low:
                continue
            if low.endswith(".hdr"):
                outp = outdir / name
                if not outp.exists():
                    f = tf.extractfile(m)
                    if f is not None:
                        outp.write_bytes(f.read())
                hdr_path = outp
            elif low.endswith(".dat") or low.endswith(".bil"):
                outp = outdir / name
                if not outp.exists():
                    f = tf.extractfile(m)
                    if f is not None:
                        outp.write_bytes(f.read())
                dat_path = outp
    if not hdr_path or not dat_path:
        raise HTTPException(status_code=500, detail=f"Missing .hdr or .dat for var {code}")
    return hdr_path, dat_path


def build_24h_cog(date) -> Path:
    dstr = _yyyymmdd(date)
    out = CACHE / f"melt24h_{dstr}_cog.tif"
    if out.exists() and out.stat().st_size > 0:
        return out
    with _lock(f"melt24h_{dstr}"):
        if out.exists() and out.stat().st_size > 0:
            return out
        tar = _download_day_tar(date)
        hdr, dat = _extract_var(tar, "1044")
        da = _open_bil(hdr, dat)
        p = _da_to_cog(da, out)

        try:
            _hf_enqueue_files([p])
        except Exception:
            pass
        return p


def build_72h_cog(end_date) -> Path:
    dstr = _yyyymmdd(end_date)
    out = CACHE / f"melt72h_end_{dstr}_cog.tif"
    if out.exists() and out.stat().st_size > 0:
        return out
    with _lock(f"melt72h_{dstr}"):
        if out.exists() and out.stat().st_size > 0:
            return out
        cogs = [
            build_24h_cog(end_date - timedelta(days=2)),
            build_24h_cog(end_date - timedelta(days=1)),
            build_24h_cog(end_date),
        ]
        with rasterio.open(cogs[0]) as base:
            acc = base.read(1).astype("float32")
            for c in cogs[1:]:
                with rasterio.open(c) as src:
                    arr = src.read(1).astype("float32")
                    if (src.transform != base.transform) or (src.crs != base.crs) or (src.width != base.width) or (src.height != base.height):
                        tmp = np.full_like(acc, np.nan, dtype="float32")
                        reproject(
                            arr,
                            tmp,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=base.transform,
                            dst_crs=base.crs,
                            resampling=Resampling.nearest,
                        )
                        arr = tmp
                    m = np.isfinite(arr)
                    acc[m] = np.nan_to_num(acc[m], nan=0.0) + arr[m]
            tmp_tif = out.with_suffix(".tmp.tif")
            prof = base.profile.copy()
            prof.update({"compress": "DEFLATE", "tiled": True})
            with rasterio.open(tmp_tif, "w", **prof) as ds:
                ds.write(acc, 1)
        cog_translate(tmp_tif.as_posix(), out.as_posix(), cog_profiles.get("deflate"), in_memory=False, quiet=True)
        tmp_tif.unlink(missing_ok=True)

        try:
            _hf_enqueue_files([out])
        except Exception:
            pass

        return out



def _forecast_dir() -> Path:
    d = CACHE / "forecast"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _forecast_melt_cog(run_init: str, valid: str, lead: str, dom: str) -> Path:
    return _forecast_dir() / f"fcst_melt_{lead}_{run_init}_{valid}_{dom}_cog.tif"


def _forecast_snow_cog(ts: str, dom: str, code: str) -> Path:
    return _forecast_dir() / f"fcst_snow_{code}_{ts}_{dom}_cog.tif"


def _forecast_melt72h_end_cog(valid: str, dom_prefer: str = "zz") -> Path:
    valid = _norm_ts(valid)
    dom_prefer = (dom_prefer or "zz").lower()
    out = _forecast_dir() / f"fcst_melt72h_end_{valid}_{dom_prefer}_cog.tif"
    if out.exists() and out.stat().st_size > 0:
        return out

    with _lock(f"fcst_melt72h_{valid}_{dom_prefer}"):
        if out.exists() and out.stat().st_size > 0:
            return out

        v0 = valid
        v1 = _shift_ts(v0, -24)
        v2 = _shift_ts(v0, -48)

        sels = []
        for v in (v2, v1, v0):
            s = _pick_forecast_for_valid(v, dom_prefer=dom_prefer)
            sels.append(s)
        if not all(s.get("ok") for s in sels):
            raise HTTPException(status_code=503, detail={"error": "forecast_selection_unavailable_for_72h", "need": [v2, v1, v0], "sels": sels})

        day_cogs: list[Path] = []
        for s, v in zip(sels, (v2, v1, v0)):
            melt_cogs = _build_forecast_melt_cogs(s["run_init"], v, "t0024", s["melt_urls"])
            pick = None
            for p in melt_cogs:
                if _cog_dom(p) == dom_prefer:
                    pick = p
                    break
            if pick is None:
                pick = sorted(melt_cogs, key=lambda p: _dom_pri(_cog_dom(p)))[0]
            day_cogs.append(pick)

        with rasterio.open(day_cogs[0]) as base:
            acc = base.read(1).astype("float32")
            for c in day_cogs[1:]:
                with rasterio.open(c) as src:
                    arr = src.read(1).astype("float32")
                    if (src.transform != base.transform) or (src.crs != base.crs) or (src.width != base.width) or (src.height != base.height):
                        tmp = np.full_like(acc, np.nan, dtype="float32")
                        reproject(
                            arr,
                            tmp,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=base.transform,
                            dst_crs=base.crs,
                            resampling=Resampling.nearest,
                        )
                        arr = tmp
                    m = np.isfinite(arr) & (arr > -9990.0)
                    acc[m] = np.nan_to_num(acc[m], nan=0.0) + arr[m]

            tmp_tif = out.with_suffix(".tmp.tif")
            prof = base.profile.copy()
            prof.update({"compress": "DEFLATE", "tiled": True})
            with rasterio.open(tmp_tif, "w", **prof) as ds:
                ds.write(acc, 1)

        cog_translate(tmp_tif.as_posix(), out.as_posix(), cog_profiles.get("deflate"), in_memory=False, quiet=True)
        tmp_tif.unlink(missing_ok=True)

        _log("INFO", f"[build_forecast] melt72h OK valid={valid} dom_prefer={dom_prefer} -> {out.name} (from {', '.join([p.name for p in day_cogs])})")
        return out
def _pick_grib_var(ds: xr.Dataset) -> xr.DataArray:
    best = None
    best_score = -10**9
    for name, da in ds.data_vars.items():
        a = dict(da.attrs or {})
        units = str(a.get("units", "")).lower()
        grib_name = str(a.get("GRIB_name", "")).lower()
        std = str(a.get("standard_name", "")).lower()
        longn = str(a.get("long_name", "")).lower()
        txt = " ".join([name.lower(), grib_name, std, longn])

        score = 0
        if "snow" in txt or "swe" in txt or "water equivalent" in txt:
            score += 500
        if "depth" in txt:
            score += 250
        if units in ("m", "meter", "metre", "mm"):
            score += 120
        if units in ("k", "kelvin", "degc", "c", "degree_celsius"):
            score -= 400
        if da.ndim >= 2:
            score += 50

        if score > best_score:
            best_score = score
            best = da

    if best is None:
        var = next(iter(ds.data_vars))
        return ds[var]
    return best

def _ensure_xy_transform(da: xr.DataArray) -> xr.DataArray:
    if not ("x" in da.coords and "y" in da.coords):
        return da

    try:
        x = np.asarray(da["x"].values)
        y = np.asarray(da["y"].values)
        if x.ndim != 1 or y.ndim != 1:
            return da
        if x.size < 2 or y.size < 2:
            return da

        dx = float(np.nanmedian(np.diff(x)))
        dy = float(np.nanmedian(np.diff(y)))
        if not np.isfinite(dx) or not np.isfinite(dy) or dx == 0.0 or dy == 0.0:
            return da

        left = float(x[0]) - dx / 2.0

        if dy < 0:
            top = float(y[0]) + (-dy) / 2.0
        else:
            top = float(y[0]) - dy / 2.0

        tfm = Affine(dx, 0.0, left, 0.0, dy, top)
        return da.rio.write_transform(tfm, inplace=False)
    except Exception:
        return da


def _build_forecast_melt_cogs(run_init: str, valid: str, lead: str, melt_urls: list[str]) -> list[Path]:
    run_init = _norm_ts(run_init)
    valid = _norm_ts(valid)
    lead = (lead or "").lower()
    dom2urls: dict[str, list[str]] = {}
    for u in melt_urls:
        dom2urls.setdefault(_dom_from_url(u), []).append(u)

    built: list[Path] = []
    with _lock(f"melt_{lead}_{run_init}_{valid}"):
        for dom in sorted(dom2urls.keys(), key=_dom_pri):
            out = _forecast_melt_cog(run_init, valid, lead, dom)
            if out.exists() and out.stat().st_size > 0:
                built.append(out)
                continue
            last_err = None
            for url in dom2urls[dom]:
                low = Path(url).name.lower()
                if lead and lead not in low:
                    continue
                with tempfile.TemporaryDirectory() as td:
                    try:
                        desc, _ = _download_url_to_grid(url, Path(td))
                        da = _open_grid(desc)
                        _da_to_cog(da, out)
                        if out.exists() and out.stat().st_size > 0:
                            _log("INFO", f"[build_forecast] melt OK run_init={run_init} valid={valid} lead={lead} dom={dom} url={url} -> {out.name}")
                            built.append(out)
                            break
                        last_err = "empty COG"
                    except HTTPException as e:
                        last_err = f"{e.status_code} {e.detail}"
                    except Exception as e:
                        last_err = repr(e)
            if out not in built:
                _log("INFO", f"[build_forecast] melt FAIL run_init={run_init} valid={valid} lead={lead} dom={dom} last_err={last_err}")
    if not built:
        raise HTTPException(status_code=502, detail=f"No usable MELT field for run_init={run_init} valid={valid} lead={lead}")
    return built


# def _build_forecast_swe_cogs(ts: str, urls: list[str]) -> list[Path]:
#     ts = _norm_ts(ts) if ts and _TS10.fullmatch(ts) else ts
#     dom2urls: dict[str, list[str]] = {}
#     for u in urls:
#         dom2urls.setdefault(_dom_from_url(u), []).append(u)

#     built: list[Path] = []
#     with _lock(f"snow_{ts}"):
#         for dom in sorted(dom2urls.keys(), key=_dom_pri):
#             last_err = None
#             for url in dom2urls[dom]:
#                 low = Path(url).name.lower()
#                 code = "ssmv11036" if "ssmv11036" in low else ("ssmv11036" if "ssmv11036" in low else "ssmv11005")
#                 out = _forecast_snow_cog(ts or "na", dom, code)
#                 if out.exists() and out.stat().st_size > 0:
#                     built.append(out)
#                     break
#                 with tempfile.TemporaryDirectory() as td:
#                     try:
#                         desc, _ = _download_url_to_grid(url, Path(td))
#                         da = _open_grid(desc)
#                         _da_to_cog(da, out)
#                         if out.exists() and out.stat().st_size > 0:
#                             _log("INFO", f"[build_forecast] snow OK ts={ts} dom={dom} code={code} url={url} -> {out.name}")
#                             built.append(out)
#                             break
#                         last_err = "empty COG"
#                     except HTTPException as e:
#                         last_err = f"{e.status_code} {e.detail}"
#                     except Exception as e:
#                         last_err = repr(e)
#             if not any(f"_{dom}_" in p.name for p in built):
#                 _log("INFO", f"[build_forecast] snow FAIL ts={ts} dom={dom} last_err={last_err}")
#     if not built:
#         raise HTTPException(status_code=502, detail=f"No usable snow field for {ts}")
#     return built

def _build_forecast_snowpack_cogs(ts: str, urls: list[str]) -> list[Path]:
    ts = _norm_ts(ts) if ts and _TS10.fullmatch(ts) else ts
    dom2urls: dict[str, list[str]] = {}
    for u in urls:
        dom2urls.setdefault(_dom_from_url(u), []).append(u)

    built: list[Path] = []
    with _lock(f"snow_{ts}"):
        for dom in sorted(dom2urls.keys(), key=_dom_pri):
            last_err = None
            for url in dom2urls[dom]:
                low = Path(url).name.lower()
                if "ssmv11036" in low:
                    code = "ssmv11036"
                elif "ssmv11034" in low:
                    code = "ssmv11034"
                elif "ssmv11005" in low:
                    code = "ssmv11005"
                else:
                    code = "ssmv_unknown"

                out = _forecast_snow_cog(ts or "na", dom, code)
                if out.exists() and out.stat().st_size > 0:
                    built.append(out)
                    break

                with tempfile.TemporaryDirectory() as td:
                    try:
                        desc, _ = _download_url_to_grid(url, Path(td))
                        da = _open_grid(desc)
                        _da_to_cog(da, out)
                        with rasterio.open(out) as ds:
                            _log("INFO", f"[snow_cog] {out.name} crs={ds.crs} bounds={tuple(ds.bounds)} size=({ds.width}x{ds.height})")

                        if out.exists() and out.stat().st_size > 0:
                            _log("INFO", f"[build_forecast] snow OK ts={ts} dom={dom} code={code} url={url} -> {out.name}")
                            built.append(out)
                            break
                        last_err = "empty COG"
                    except HTTPException as e:
                        last_err = f"{e.status_code} {e.detail}"
                    except Exception as e:
                        last_err = repr(e)

            if not any(f"_{dom}_" in p.name for p in built):
                _log("INFO", f"[build_forecast] snow FAIL ts={ts} dom={dom} last_err={last_err}")

    if not built:
        raise HTTPException(status_code=502, detail=f"No usable snow field for {ts}")
    return built

def _cog_dom(p: Path) -> str:
    parts = p.name.split("_")
    return parts[-2] if len(parts) >= 2 else "xx"


def _union_tile(
    cogs: list[Path],
    z: int,
    x: int,
    y: int,
    *,
    label: str,
    valid_pred: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if not cogs:
        return None, None, f"{label}_no_cogs"
    out_arr: Optional[np.ndarray] = None
    out_valid: Optional[np.ndarray] = None
    used: list[str] = []
    for cog in sorted(cogs, key=lambda p: _dom_pri(_cog_dom(p))):
        try:
            with COGReader(cog.as_posix()) as r:
                data, mask = r.tile(x, y, z, tilesize=256, resampling_method="nearest")
            arr = data[0].astype("float32")
            m = mask[0]
            valid = (m > 0) & np.isfinite(arr) & (arr > -9990.0)
            if valid_pred is not None:
                try:
                    valid = valid & valid_pred(arr, m)
                except Exception:
                    pass
            if out_arr is None:
                out_arr = arr.copy()
                out_valid = valid.copy()
            else:
                fill = valid & (~out_valid)
                if fill.any():
                    out_arr[fill] = arr[fill]
                    out_valid[fill] = True
            used.append(cog.name)
        except TileOutsideBounds:
            continue
        except Exception:
            continue
    if out_arr is None or out_valid is None:
        return None, None, f"{label}_unavailable_all_domains"
    info = f"{label}_union:" + ",".join(used[:4]) + ("..." if len(used) > 4 else "")
    return out_arr, out_valid, info


def _read_mask_tile(ts: str, urls: list[str], z: int, x: int, y: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    ts = _norm_ts(ts) if ts else ts
    if not urls:
        return None, None, "mask_no_urls"
    try:
        cogs = _build_forecast_snowpack_cogs(ts or "0000000000", urls)
    except HTTPException as e:
        return None, None, f"mask_build_fail:{e.status_code}:{e.detail}"
    except Exception as e:
        return None, None, f"mask_build_fail:{e!r}"
    arr, valid, info = _union_tile(cogs, z, x, y, label="mask")
    return arr, valid, info

def _cog_point_data_mask(r: COGReader, lon: float, lat: float) -> tuple[np.ndarray, np.ndarray]:
    out = r.point(lon, lat)

    # rio-tiler older behavior: (data, mask)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        data, mask = out
        return np.asarray(data), np.asarray(mask)

    # rio-tiler newer behavior: ImageData-like object with .data/.mask
    data = getattr(out, "data", None)
    mask = getattr(out, "mask", None)
    if data is None or mask is None:
        raise ValueError(f"Unexpected COGReader.point() return type: {type(out)!r}")

    return np.asarray(data), np.asarray(mask)

INCH_TO_MM = 25.4
BIN_EDGES_IN = np.array(
    [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.39, 0.59, 0.79, 0.98, 1.4, 2.0, 3.0, 4.0, 5.0, 9999.0],
    dtype="float32",
)
BIN_EDGES_72H_IN = np.array(
    [0.01, 0.04, 0.08, 0.16, 0.24, 0.31, 0.39, 0.79, 1.4, 2.0, 3.0, 3.9, 5.0, 6.5, 8.0, 9.8, 9999.0],
    dtype="float32",
)
MELT_COLORS = np.array(
    [
        [0xDC, 0xDC, 0xDC],
        [0x99, 0x00, 0x00],
        [0xCC, 0x33, 0x33],
        [0xFF, 0x66, 0x66],
        [0xCC, 0xCC, 0x66],
        [0xFF, 0x8C, 0x00],
        [0xFF, 0xFF, 0x66],
        [0x99, 0xFF, 0x99],
        [0x00, 0xFF, 0x00],
        [0x66, 0xCC, 0x66],
        [0x66, 0x66, 0xFF],
        [0x33, 0x33, 0xCC],
        [0x00, 0x00, 0x99],
        [0x99, 0x00, 0x99],
        [0x4B, 0x00, 0x82],
        [0x2F, 0x00, 0x4F],
        [0x00, 0x00, 0x00],
    ],
    dtype="uint8",
)

@lru_cache(maxsize=32)
def _corridor_tiles_set_for_z(z: int) -> set[tuple[int, int]]:
    pts = _load_track_points()
    if not pts:
        return set()
    tiles = _tiles_near_points(pts, int(z), radius=WARM_RADIUS, cap=WARM_CAP_PER_ZOOM)
    return set((x, y) for (_, x, y) in tiles)

def _is_corridor_tile(z: int, x: int, y: int) -> bool:
    s = _corridor_tiles_set_for_z(int(z))
    return (int(x), int(y)) in s

def _effective_request_tile(z: int, x: int, y: int) -> tuple[int, int, int, str]:
    z = int(z); x = int(x); y = int(y)

    if not _tile_allowed(z, x, y):
        return z, x, y, "none"

    if _is_corridor_tile(z, x, y):
        return z, x, y, "hi"

    # off-track but allowed: serve low-res parent tile
    z_lr = min(int(LOWRES_Z), z)
    if z <= z_lr:
        return z, x, y, "lo"

    shift = z - z_lr
    return z_lr, (x >> shift), (y >> shift), "lo"
def _tiles_in_allowed_boxes(z: int) -> list[tuple[int, int, int]]:
    z = int(z)
    n = 1 << z

    box_a = (-170.0, 49.0, -40.0, 90.0)  # Canada north band
    box_b = (-97.0, 37.0, -63.0, 49.0)   # southern band

    def tile_range_for_box(box):
        w, s, e, nlat = box
        x0, y0 = _lonlat_to_tile(w, nlat, z)
        x1, y1 = _lonlat_to_tile(e, s, z)
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        xa = max(0, min(n - 1, xa)); xb = max(0, min(n - 1, xb))
        ya = max(0, min(n - 1, ya)); yb = max(0, min(n - 1, yb))
        out = []
        for xx in range(xa, xb + 1):
            for yy in range(ya, yb + 1):
                out.append((z, xx, yy))
        return out

    seen = set()
    out = []
    for t in tile_range_for_box(box_a) + tile_range_for_box(box_b):
        _, x, y = t
        if (x, y) in seen:
            continue
        seen.add((x, y))
        out.append(t)
    return out

def _bake_lowres_box_tiles_for_date(
    *,
    date_yyyymmdd: str,
    dom: str,
    hours_list: list[int],
    zmin: int,
    zmax: int,
    conc: int = 6,
) -> dict:
    dom = (dom or "zz").lower()

    tiles: list[tuple[int, int, int]] = []
    for z in range(int(zmin), int(zmax) + 1):
        tiles.extend(_tiles_in_allowed_boxes(z))

    if not tiles:
        return {"ok": False, "error": "no_lowres_tiles"}

    q = deque()
    for hours in hours_list:
        for (z, x, y) in tiles:
            q.append((int(hours), z, x, y))

    ok = 0
    fail = 0
    lk = threading.Lock()

    def worker():
        nonlocal ok, fail
        while True:
            with lk:
                if not q:
                    return
                hours, z, x, y = q.popleft()
            try:
                if _tile_cache_get(dom, hours, date_yyyymmdd, z, x, y) is not None:
                    ok += 1
                    continue
                png, _ = _generate_forecast_by_date_png(
                    z=z, x=x, y=y,
                    date_yyyymmdd=date_yyyymmdd,
                    hours=hours,
                    dom=dom,
                    max_in=None,
                )
                _tile_cache_put(dom, hours, date_yyyymmdd, z, x, y, png)
                ok += 1
            except Exception:
                fail += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(int(conc))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {
        "ok": True,
        "date": date_yyyymmdd,
        "dom": dom,
        "hours_list": hours_list,
        "tiles": len(tiles),
        "jobs": len(tiles) * len(hours_list),
        "ok_count": ok,
        "fail_count": fail,
        "zmin": zmin,
        "zmax": zmax,
    }

def _png(rgb: np.ndarray, a: np.ndarray) -> bytes:
    return render(rgb, mask=a.astype("uint8"), img_format="PNG")


def _dilate_bool_1px(m: np.ndarray) -> np.ndarray:
    m = m.astype(bool, copy=False)
    if m.size == 0:
        return m
    p = np.pad(m, 1, mode="constant", constant_values=False)
    out = p[1:-1, 1:-1].copy()
    out |= p[:-2, 1:-1]
    out |= p[2:, 1:-1]
    out |= p[1:-1, :-2]
    out |= p[1:-1, 2:]
    out |= p[:-2, :-2]
    out |= p[:-2, 2:]
    out |= p[2:, :-2]
    out |= p[2:, 2:]
    return out


def _melt_to_png(
    melt_mm: np.ndarray,
    melt_mask: np.ndarray,
    max_in: Optional[float],
    snow_mm: Optional[np.ndarray],
    snow_valid: Optional[np.ndarray],
    *,
    snow_allow_min_mm: float = 0.0,
    snow_underlay_min_mm: float = 0.01,
    dilate_px: int = 1,
    bin_edges_in: Optional[np.ndarray] = None,
) -> bytes:
    mm = melt_mm.astype("float32")
    melt_valid = (melt_mask > 0) & np.isfinite(mm) & (mm > -9990.0)
    mm = np.where(melt_valid, np.maximum(mm, 0.0), np.nan).astype("float32")

    snow_underlay = np.zeros_like(melt_valid, dtype=bool)
    snow_allow = None

    if snow_mm is not None and snow_valid is not None:
        snow = snow_mm.astype("float32")
        snow_ok = snow_valid & np.isfinite(snow) & (snow > -9990.0)

        snow_underlay = snow_ok & (snow >= float(snow_underlay_min_mm))
        snow_allow = snow_ok & (snow >= float(snow_allow_min_mm))

        if dilate_px and dilate_px > 0:
            for _ in range(int(dilate_px)):
                snow_allow = _dilate_bool_1px(snow_allow)

        melt_valid = melt_valid & snow_allow

    inches = mm / INCH_TO_MM
    if max_in is not None:
        inches = np.minimum(inches, float(max_in))

    melt_vis = np.isfinite(inches) & (inches > 0.0) & melt_valid
    snow_only = snow_underlay & (~melt_vis)

    h, w = mm.shape
    rgb = np.zeros((3, h, w), dtype="uint8")
    alpha = np.zeros((h, w), dtype="uint8")

    if snow_only.any():
        rgb[:, snow_only] = np.array(MELT_COLORS[0], dtype="uint8").reshape(3, 1)
        alpha[snow_only] = 255

    if melt_vis.any():
        filled = np.where(melt_vis, inches, 0.0).astype("float32")
        edges = bin_edges_in if bin_edges_in is not None else BIN_EDGES_IN
        idx = np.digitize(filled, edges, right=True)
        idx = np.clip(idx, 0, len(MELT_COLORS) - 1)
        melt_rgb = np.moveaxis(MELT_COLORS[idx], -1, 0)
        rgb[:, melt_vis] = melt_rgb[:, melt_vis]
        alpha[melt_vis] = 255

    return _png(rgb, alpha)
    
# def build_forecast_72h_melt_cog(run_init: str, valid: str, dom_prefer: str = "zz") -> Path:
#     run_init = _norm_ts(run_init)
#     valid = _norm_ts(valid)
#     dom_prefer = (dom_prefer or "zz").lower()

#     out = _forecast_melt_t0072_cog(run_init, valid, dom_prefer)
#     if out.exists() and out.stat().st_size > 0:
#         return out

#     lock_key = f"melt72h_{run_init}_{valid}_{dom_prefer}"
#     with _lock(lock_key):
#         if out.exists() and out.stat().st_size > 0:
#             return out

#         v0 = valid
#         v1 = _shift_ts(v0, -24)
#         v2 = _shift_ts(v0, -48)

#         u0 = _find_melt_11044_t0024(run_init, v0, dom_prefer=dom_prefer)
#         u1 = _find_melt_11044_t0024(run_init, v1, dom_prefer=dom_prefer)
#         u2 = _find_melt_11044_t0024(run_init, v2, dom_prefer=dom_prefer)
#         if not (u0 and u1 and u2):
#             raise HTTPException(
#                 status_code=503,
#                 detail={"error": "melt72_inputs_missing", "tt": run_init, "need": [v2, v1, v0], "have": [bool(u2), bool(u1), bool(u0)]},
#             )

#         cogs2 = _build_forecast_melt_cogs(run_init, v2, "t0024", u2)
#         cogs1 = _build_forecast_melt_cogs(run_init, v1, "t0024", u1)
#         cogs0 = _build_forecast_melt_cogs(run_init, v0, "t0024", u0)

#         def _best_dom_cog(cogs: list[Path], dom: str) -> Path:
#             dom = (dom or "zz").lower()
#             same = [p for p in cogs if _cog_dom(p).lower() == dom]
#             if same:
#                 same.sort(key=lambda p: _dom_pri(_cog_dom(p)))
#                 return same[0]
#             cogs = sorted(cogs, key=lambda p: _dom_pri(_cog_dom(p)))
#             return cogs[0]

#         p2 = _best_dom_cog(cogs2, dom_prefer)
#         p1 = _best_dom_cog(cogs1, dom_prefer)
#         p0 = _best_dom_cog(cogs0, dom_prefer)

#         with rasterio.open(p2) as base:
#             acc = base.read(1).astype("float32")
#             base_prof = dict(base.profile)

#             for src_path in (p1, p0):
#                 with rasterio.open(src_path) as src:
#                     arr = src.read(1).astype("float32")
#                     if (src.transform != base.transform) or (src.crs != base.crs) or (src.width != base.width) or (src.height != base.height):
#                         tmp = np.full_like(acc, np.nan, dtype="float32")
#                         reproject(
#                             arr,
#                             tmp,
#                             src_transform=src.transform,
#                             src_crs=src.crs,
#                             dst_transform=base.transform,
#                             dst_crs=base.crs,
#                             resampling=Resampling.nearest,
#                         )
#                         arr = tmp

#                     m = np.isfinite(arr) & (arr > -9990.0)
#                     if m.any():
#                         acc[m] = np.nan_to_num(acc[m], nan=0.0) + arr[m]

#         tmp_tif = out.with_suffix(".tmp.tif")
#         base_prof.update({"dtype": "float32", "count": 1, "compress": "DEFLATE", "tiled": True})
#         with rasterio.open(tmp_tif, "w", **base_prof) as ds:
#             ds.write(acc.astype("float32"), 1)

#         cog_translate(tmp_tif.as_posix(), out.as_posix(), cog_profiles.get("deflate"), in_memory=False, quiet=True)
#         tmp_tif.unlink(missing_ok=True)

#         _log("INFO", f"[build_forecast] melt72 OK tt={run_init} valid={valid} dom={dom_prefer} -> {out.name}")
#         return out


CACHE_CONTROL_TILES = os.environ.get("TILES_CACHE_CONTROL", "public, max-age=604800, immutable")

def _resp_png(p: bytes, cache_control: str | None = None, **headers) -> Response:
    resp = Response(p, media_type="image/png")
    resp.headers["Cache-Control"] = cache_control or CACHE_CONTROL_TILES
    for k, v in headers.items():
        if v is not None:
            resp.headers[k] = str(v)
    return resp


def _tile_arrays_from_cog(cog: Path, z: int, x: int, y: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    if not cog.exists() or cog.stat().st_size <= 0:
        raise HTTPException(status_code=500, detail=f"COG missing: {cog.name}")
    try:
        with COGReader(cog.as_posix()) as r:
            data, mask = r.tile(x, y, z, tilesize=256, resampling_method="nearest")
        return data[0].astype("float32"), mask[0].astype("uint8"), False
    except TileOutsideBounds:
        return None, None, True

def _tile_png_from_cog(cog: Path, z: int, x: int, y: int, max_in: Optional[float]) -> bytes:
    arr, mask, oob = _tile_arrays_from_cog(cog, z, x, y)
    if oob or arr is None or mask is None:
        return _transparent_png_256()
    return _melt_to_png(arr, mask, max_in, None, None)


def _serve_forecast_tile(
    run_init: str,
    valid: str,
    lead: str,
    melt_urls: list[str],
    snow_ts: str,
    snow_urls: list[str],
    z: int,
    x: int,
    y: int,
    max_in: float | None,
) -> Response:
    run_init = _norm_ts(run_init)
    valid = _norm_ts(valid)
    lead = (lead or "").lower()

    melt_cogs = _build_forecast_melt_cogs(run_init, valid, lead, melt_urls)

    snow = snow_valid = None
    snow_info = "mask_unavailable"
    if snow_urls:
        snow, snow_valid, snow_info = _read_mask_tile(snow_ts or valid, snow_urls, z, x, y)

    melt_mm, melt_valid, melt_info = _union_tile(melt_cogs, z, x, y, label="melt")
    if melt_mm is None or melt_valid is None:
        return _resp_png(
            _transparent_png_256(),
            **{
                "X-Melt-Info": melt_info,
                "X-NoData": "1",
            },
        )


    melt_mask_u8 = (melt_valid.astype("uint8") * 255)

    png = _melt_to_png(
        melt_mm,
        melt_mask_u8,
        max_in,
        snow,
        snow_valid,
        snow_allow_min_mm=0.0,
        snow_underlay_min_mm=0.0001,
        dilate_px=2,
    )

    return _resp_png(
        png,
        **{
            "X-Forecast-RunInit-TT": run_init,
            "X-Forecast-Valid-TS": valid,
            "X-Forecast-Lead": lead,
            "X-Melt-Info": melt_info,
            "X-SnowMask-Info": snow_info,
            "X-SnowMask-TS": (snow_ts or valid or ""),
        },
    )

def _corridor_tiles_for_zoom_range(
    zmin: int,
    zmax: int,
    *,
    radius: int,
    cap_per_zoom: int,
) -> list[tuple[int, int, int]]:
    pts = _load_track_points()
    if not pts:
        return []
    out: list[tuple[int, int, int]] = []
    total_seen: set[tuple[int, int, int]] = set()

    for z in range(int(zmin), int(zmax) + 1):
        tiles = _tiles_near_points(pts, z, radius=radius, cap=cap_per_zoom)
        for (zz, x, y) in tiles:
            if not _tile_allowed(zz, x, y):
                continue
            t = (zz, x, y)
            if t in total_seen:
                continue
            total_seen.add(t)
            out.append(t)
    return out
    
def _point_mm_from_cog(cog: Path, lon: float, lat: float) -> tuple[Optional[float], str]:
    if not cog.exists() or cog.stat().st_size <= 0:
        return None, f"missing:{cog.name}"
    try:
        with COGReader(cog.as_posix()) as r:
            data, mask = _cog_point_data_mask(r, lon, lat)

        # data/mask are typically shaped (bands,) for point queries
        v = float(np.ravel(data)[0])
        m = int(np.ravel(mask)[0])

        if m <= 0 or (not np.isfinite(v)) or (v <= -9990.0):
            return None, f"nodata:{cog.name}"
        return v, f"ok:{cog.name}"
    except TileOutsideBounds:
        return None, f"oob:{cog.name}"
    except Exception as e:
        return None, f"err:{cog.name}:{e!r}"


def _union_point_mm(cogs: list[Path], lon: float, lat: float) -> tuple[Optional[float], str]:
    if not cogs:
        return None, "no_cogs"
    used = []
    for cog in sorted(cogs, key=lambda p: _dom_pri(_cog_dom(p))):
        v, info = _point_mm_from_cog(cog, lon, lat)
        used.append(info)
        if v is not None:
            return v, "union:" + info
    return None, "union_fail:" + ",".join(used[:6])

def _bake_corridor_tiles_for_date(
    *,
    date_yyyymmdd: str,
    dom: str,
    hours_list: list[int],
    zmin: int,
    zmax: int,
    radius: int,
    cap_per_zoom: int,
    max_tiles_total: int,
    conc: int = 6,
) -> dict:
    dom = (dom or "zz").lower()

    tiles = _corridor_tiles_for_zoom_range(zmin, zmax, radius=radius, cap_per_zoom=cap_per_zoom)
    if not tiles:
        return {"ok": False, "error": "no_tiles_or_no_track_points"}

    # Global cap across all hours
    tiles = tiles[: int(max_tiles_total)]

    q = deque()
    for hours in hours_list:
        for (z, x, y) in tiles:
            q.append((hours, z, x, y))

    ok = 0
    fail = 0
    lk = threading.Lock()

    def worker():
        nonlocal ok, fail
        while True:
            with lk:
                if not q:
                    return
                hours, z, x, y = q.popleft()
            try:
                # If already cached on disk, skip
                if _tile_cache_get(dom, int(hours), date_yyyymmdd, z, x, y) is not None:
                    ok += 1
                    continue

                png, _ = _generate_forecast_by_date_png(
                    z=z, x=x, y=y,
                    date_yyyymmdd=date_yyyymmdd,
                    hours=int(hours),
                    dom=dom,
                    max_in=None,
                )
                _tile_cache_put(dom, int(hours), date_yyyymmdd, z, x, y, png)
                ok += 1
            except Exception:
                fail += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(int(conc))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {
        "ok": True,
        "date": date_yyyymmdd,
        "dom": dom,
        "hours_list": hours_list,
        "tiles": len(tiles),
        "jobs": len(hours_list) * len(tiles),
        "ok_count": ok,
        "fail_count": fail,
        "zmin": zmin,
        "zmax": zmax,
        "radius": radius,
        "cap_per_zoom": cap_per_zoom,
        "max_tiles_total": max_tiles_total,
    }

def _rolling_dates_local_window(today_local: datetime.date | None = None, days: int = 4) -> list[str]:
    if today_local is None:
        today_local = datetime.now(LOCAL_TZ).date()
    # yesterday..+2 for days=4
    start = today_local - timedelta(days=1)
    return [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(int(days))]

def _prune_tile_cache(keep_ymds: set[str], keep_days_fallback: int) -> int:
    deleted = 0
    cutoff = (datetime.utcnow().date() - timedelta(days=int(keep_days_fallback) - 1)).strftime("%Y%m%d")
    root = _tile_cache_dir() / "tilecache"
    if not root.exists():
        return 0

    for p in root.rglob("bydate_*.png"):
        try:
            name = p.name
            m = re.search(r"_([0-9]{8})_z", name)
            ymd = m.group(1) if m else None
            if not ymd:
                continue
            if (ymd in keep_ymds) or (ymd >= cutoff):
                continue
            p.unlink(missing_ok=True)
            deleted += 1
        except Exception:
            continue
    return deleted


def _prune_forecast_cogs(keep_days: int) -> int:
    deleted = 0
    cutoff = (datetime.utcnow().date() - timedelta(days=int(keep_days) - 1)).strftime("%Y%m%d")
    fc = _forecast_dir()
    for p in fc.glob("*_cog.tif"):
        ymd = _file_ymd_from_name(p.name)
        if ymd and ymd < cutoff:
            try:
                p.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass
    return deleted

def _ensure_forecast_cogs_for_date(date_yyyymmdd: str, dom: str) -> dict:
    dom = (dom or "zz").lower()
    valid = _norm_ts(f"{date_yyyymmdd}05")

    # 24h (build melt + snowpack COGs opportunistically)
    sel = _pick_forecast_for_valid(valid, dom_prefer=dom)
    if not sel.get("ok"):
        return {"ok": False, "date": date_yyyymmdd, "error": "pick_failed", "detail": sel}

    melt_cogs = _build_forecast_melt_cogs(sel["run_init"], sel["valid"], "t0024", sel["melt_urls"])

    snow_cogs = []
    if sel.get("snowpack_urls"):
        try:
            snow_cogs = _build_forecast_snowpack_cogs(sel["valid"], sel["snowpack_urls"])
        except Exception:
            snow_cogs = []

    # 72h (reuses built 24h cogs for v0,v1,v2)
    melt72 = _forecast_melt72h_end_cog(sel["valid"], dom_prefer=dom)

    return {
        "ok": True,
        "date": date_yyyymmdd,
        "valid": sel["valid"],
        "tt": sel["run_init"],
        "dom": dom,
        "melt24_cogs": [p.name for p in melt_cogs[:3]],
        "snow_cogs": [p.name for p in snow_cogs[:3]],
        "melt72_cog": melt72.name,
    }

def _next_run_local(now: datetime | None = None) -> datetime:
    now = now or datetime.now(LOCAL_TZ)
    target = now.replace(hour=SCHEDULE_HOUR_LOCAL, minute=SCHEDULE_MIN_LOCAL, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target

def _run_daily_roll_job_once() -> dict:
    started = datetime.utcnow().isoformat()
    today_local = datetime.now(LOCAL_TZ).date()
    window = _rolling_dates_local_window(today_local=today_local, days=ROLL_WINDOW_DAYS)
    keep_set = set(window)

    _log("INFO", f"[roll] start local_today={today_local.isoformat()} window={window}")

    # in _run_daily_roll_job_once
    try:
        _hf_pull_cache()
    except Exception:
        pass


    del_tiles = _prune_tile_cache(keep_set, KEEP_TILE_DAYS)
    del_fcst = _prune_forecast_cogs(KEEP_FORECAST_COG_DAYS)

    try:
        _tile_png_cached.cache_clear()
    except Exception:
        pass

    ensured = []
    for ymd in window:
        try:
            ensured.append(_ensure_forecast_cogs_for_date(ymd, WARM_DOM))
        except Exception as e:
            ensured.append({"ok": False, "date": ymd, "error": f"ensure_exc:{e!r}"})

    baked_corridor = []
    for ymd in window:
        try:
            baked_corridor.append(
                _bake_corridor_tiles_for_date(
                    date_yyyymmdd=ymd,
                    dom=WARM_DOM,
                    hours_list=WARM_HOURS_LIST,
                    zmin=WARM_ZMIN,
                    zmax=WARM_ZMAX,
                    radius=WARM_RADIUS,
                    cap_per_zoom=WARM_CAP_PER_ZOOM,
                    max_tiles_total=WARM_MAX_TILES_TOTAL,
                    conc=int(os.environ.get("BAKE_CONCURRENCY", "2")),
                )
            )
        except Exception as e:
            baked_corridor.append({"ok": False, "date": ymd, "error": f"bake_exc:{e!r}"})

    baked_lowres = []
    for ymd in window:
        try:
            baked_lowres.append(
                _bake_lowres_box_tiles_for_date(
                    date_yyyymmdd=ymd,
                    dom=WARM_DOM,
                    hours_list=WARM_HOURS_LIST,
                    zmin=LOWRES_BOX_ZMIN,
                    zmax=LOWRES_BOX_ZMAX,
                    conc=max(1, min(6, int(os.environ.get("BAKE_CONCURRENCY", "2")))),
                )
            )
        except Exception as e:
            baked_lowres.append({"ok": False, "date": ymd, "error": f"bake_lowres_exc:{e!r}"})

    pushed_tiles = None
    try:
        pushed_tiles = _hf_push_tilecache_for_window(window, WARM_DOM, WARM_HOURS_LIST)
    except Exception as e:
        pushed_tiles = {"ok": False, "error": f"push_tilecache_exc:{e!r}"}

    try:
        _hf_prune_remote_and_local(keep_days=max(KEEP_FORECAST_COG_DAYS, KEEP_TILE_DAYS))
    except Exception:
        pass

    try:
        _hf_flush_pending(message=f"cache: roll bake {today_local.isoformat()} ({len(window)} days)")
    except Exception:
        pass

    finished = datetime.utcnow().isoformat()
    _log("INFO", f"[roll] done del_tiles={del_tiles} del_fcst={del_fcst}")

    return {
        "ok": True,
        "started_utc": started,
        "finished_utc": finished,
        "local_today": today_local.isoformat(),
        "window": window,
        "deleted_tiles": del_tiles,
        "deleted_forecast_cogs": del_fcst,
        "ensured": ensured,
        "baked_corridor": baked_corridor,
        "baked_lowres": baked_lowres,
        "pushed_tiles": pushed_tiles,
    }



def _daily_scheduler_loop() -> None:
    time.sleep(0.25)
    _log("INFO", "[scheduler] start")
    while True:
        try:
            now = datetime.now(LOCAL_TZ)
            nxt = _next_run_local(now)
            sleep_s = max(1.0, (nxt - now).total_seconds())
            _log("INFO", f"[scheduler] next_run_local={nxt.isoformat()} sleep_s={int(sleep_s)}")
            time.sleep(sleep_s)

            _run_daily_roll_job_once()
        except Exception as e:
            _log("INFO", f"[scheduler] crashed err={e!r}")
            time.sleep(30.0)


def _shift_ts(ts10: str, hours: int) -> str:
    dt = _ts_to_dt(_norm_ts(ts10)) + timedelta(hours=hours)
    return dt.strftime("%Y%m%d%H")

def _forecast_melt_t0072_cog(run_init: str, valid: str, dom: str) -> Path:
    run_init = _norm_ts(run_init)
    valid = _norm_ts(valid)
    dom = (dom or "zz").lower()
    return _forecast_dir() / f"fcst_melt_t0072_{run_init}_{valid}_{dom}_cog.tif"

# def _warm_cache_background() -> None:
#     time.sleep(0.25)
#     try:
#         _log("INFO", "[warm] start")

#         ZMIN = 8
#         ZMAX = 11

#         BOX_CANADA = {
#             "name": "canada_north",
#             "lat_min": 49.0,
#             "lat_max": 54,
#             "lon_w": -170.0,
#             "lon_e": -40.0,
#         }

#         BOX_SOUTH = {
#             "name": "south_band",
#             "lat_min": 39.0,
#             "lat_max": 49.0,
#             "lon_w": -97.0,
#             "lon_e": -63.0,
#         }

#         def _clamp(v: int, lo: int, hi: int) -> int:
#             return lo if v < lo else hi if v > hi else v

#         def _range_for_box(z: int, box: dict) -> list[tuple[int, int, int]]:
#             n = 1 << int(z)

#             xw, _ = _lonlat_to_tile(float(box["lon_w"]), float(box["lat_max"]), z)
#             xe, _ = _lonlat_to_tile(float(box["lon_e"]), float(box["lat_max"]), z)
#             x0 = _clamp(int(min(xw, xe)), 0, n - 1)
#             x1 = _clamp(int(max(xw, xe)), 0, n - 1)

#             _, y_top = _lonlat_to_tile(float(box["lon_w"]), float(box["lat_max"]), z)
#             _, y_bot = _lonlat_to_tile(float(box["lon_w"]), float(box["lat_min"]), z)
#             y0 = _clamp(int(min(y_top, y_bot)), 0, n - 1)
#             y1 = _clamp(int(max(y_top, y_bot)), 0, n - 1)

#             out = []
#             for x in range(x0, x1 + 1):
#                 for y in range(y0, y1 + 1):
#                     out.append((z, x, y))
#             return out

#         seen: set[tuple[int, int, int]] = set()
#         allowed: list[tuple[int, int, int]] = []

#         for z in range(int(ZMIN), int(ZMAX) + 1):
#             for box in (BOX_CANADA, BOX_SOUTH):
#                 for t in _range_for_box(z, box):
#                     if t in seen:
#                         continue
#                     seen.add(t)
#                     allowed.append(t)

#         if not allowed:
#             _log("INFO", f"[warm] no tiles computed z={ZMIN}-{ZMAX}")
#             _log("INFO", "[warm] done")
#             return

#         sel = _pick_forecast_for_days(2, dom_prefer="zz")
#         if not sel.get("ok"):
#             _log("INFO", f"[warm] forecast pick FAIL days=2 detail={sel.get('error')}")
#             _log("INFO", "[warm] done")
#             return

#         ymd = sel["valid"][:8]
#         hours_list = [24, 72]

#         def _tile_url(z: int, x: int, y: int, hours: int) -> str:
#             return f"http://127.0.0.1:8000/tiles/forecast/by_date/{z}/{x}/{y}.png?date_yyyymmdd={ymd}&hours={hours}&dom=zz"

#         def _run_batch(urls: list[str], conc: int = 6) -> tuple[int, int]:
#             ok = 0
#             fail = 0
#             q = deque(urls)
#             lk = threading.Lock()

#             def worker():
#                 nonlocal ok, fail
#                 while True:
#                     with lk:
#                         if not q:
#                             return
#                         u = q.popleft()
#                     try:
#                         r = _retry_get(u, timeout=(5, 60), tries=1, headers={"User-Agent": "snodas-warm/1.3"})
#                         if r.status_code == 200:
#                             ok += 1
#                         else:
#                             fail += 1
#                     except Exception:
#                         fail += 1

#             threads = [threading.Thread(target=worker, daemon=True) for _ in range(int(conc))]
#             for t in threads:
#                 t.start()
#             for t in threads:
#                 t.join()
#             return ok, fail

#         urls: list[str] = []
#         for hours in hours_list:
#             for (z, x, y) in allowed:
#                 urls.append(_tile_url(z, x, y, hours))

#         random.shuffle(urls)

#         _log(
#             "INFO",
#             f"[warm] prewarm valid={sel['valid']} tiles={len(allowed)} urls={len(urls)} z={ZMIN}-{ZMAX} "
#             f"boxes=[{BOX_CANADA['name']}, {BOX_SOUTH['name']}]",
#         )
#         ok, fail = _run_batch(urls, conc=6)
#         _log("INFO", f"[warm] prewarm done ok={ok} fail={fail}")

#         _log("INFO", "[warm] done")
#     except Exception as e:
#         _log("INFO", f"[warm] crashed err={e!r}")

def _union_point(cogs: list[Path], lon: float, lat: float) -> tuple[Optional[float], str]:
    if not cogs:
        return None, "no_cogs"
    used = []
    for cog in sorted(cogs, key=lambda p: _dom_pri(_cog_dom(p))):
        try:
            with COGReader(cog.as_posix()) as r:
                data, mask = _cog_point_data_mask(r, lon, lat)

            v = float(np.ravel(data)[0])
            m = int(np.ravel(mask)[0])
            used.append(cog.name)

            if m > 0 and np.isfinite(v) and v > -9990.0:
                return v, "ok:" + cog.name
        except Exception:
            continue
    return None, "nodata:" + ",".join(used[:3])


@app.get("/forecast/sample_melt")
def forecast_sample_melt(
    date_yyyymmdd: str = Query(..., regex=r"^\d{8}$"),
    hours: int = Query(24, ge=24, le=72),
    lon: float = Query(..., ge=-180.0, le=180.0),
    lat: float = Query(..., ge=-90.0, le=90.0),
    dom: str = Query("zz"),
):
    dom = (dom or "zz").lower()
    valid = _norm_ts(f"{date_yyyymmdd}05")

    mm = None
    info = ""

    if int(hours) == 72:
        cog = _forecast_melt72h_end_cog(valid, dom_prefer=dom)
        try:
            with COGReader(cog.as_posix()) as r:
                data, mask = _cog_point_data_mask(r, lon, lat)
            v = float(np.ravel(data)[0])
            m = int(np.ravel(mask)[0])

            if m > 0 and np.isfinite(v) and v > -9990.0:
                mm = v
                info = "ok:" + cog.name
            else:
                info = "nodata:" + cog.name
        except Exception as e:
            info = f"err:{e!r}"
    else:
        sel = _pick_forecast_for_valid(valid, dom_prefer=dom)
        if not sel.get("ok"):
            detail = dict(sel)
            detail["collab_last"] = dict(_COLLAB_LAST)
            raise HTTPException(status_code=503, detail=detail)

        melt_cogs = _build_forecast_melt_cogs(sel["run_init"], sel["valid"], "t0024", sel["melt_urls"])
        mm, info = _union_point(melt_cogs, lon, lat)

    if mm is None:
        return {"ok": True, "value_in": None, "info": info, "hours": int(hours), "date_yyyymmdd": date_yyyymmdd}

    inches = max(0.0, float(mm) / INCH_TO_MM)
    return {
        "ok": True,
        "value_in": round(inches, 1),
        "info": info,
        "hours": int(hours),
        "date_yyyymmdd": date_yyyymmdd,
    }
    
@app.on_event("startup")
def _startup() -> None:
    if _acquire_singleton_lock("hf_pull"):
        try:
            _hf_pull_cache()
        except Exception:
            pass
    else:
        _log("INFO", "[startup] hf pull skipped (another worker holds lock)")

    if _acquire_singleton_lock("daily_scheduler"):
        t = threading.Thread(target=_daily_scheduler_loop, daemon=True)
        t.start()
        _log("INFO", "[startup] daily scheduler enabled (singleton)")
    else:
        _log("INFO", "[startup] daily scheduler disabled (another worker holds lock)")

    if (os.environ.get("RUN_BOOT_ROLL", "0") == "1") and _acquire_singleton_lock("boot_roll"):
        t2 = threading.Thread(target=_run_daily_roll_job_once, daemon=True)
        t2.start()
        _log("INFO", "[startup] boot roll job kicked off (RUN_BOOT_ROLL=1)")
    else:
        _log("INFO", "[startup] boot roll job skipped")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "SNODAS Snowmelt Tiles. See /web for demo and /docs for API."


@app.get("/healthz", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/config.js", response_class=PlainTextResponse)
def config_js():
    token = os.environ.get("MAPBOX_TOKEN", "")
    return f"window.__SNODAS_CONFIG__ = {{ MAPBOX_TOKEN: '{token}' }};"

@app.get("/tiles/24h/latest/{z}/{x}/{y}.png")
def tiles_24h_latest(z: int, x: int, y: int, max: float | None = Query(None)):
    d = _resolve_latest_nsidc_day()
    if d is None:
        raise HTTPException(status_code=503, detail="No recent NSIDC day available")

    # Optional server-side tile cache for "latest" (short-lived).
    # We key by resolved day so that when "latest" advances, it naturally becomes a new key.
    if max is None:
        try:
            hit = _tile_cache_get("nsidc", 24, d, z, x, y)  # dom="nsidc" just namespacing
            if hit is not None:
                return _resp_png(
                    hit,
                    cache_control=CACHE_CONTROL_LATEST,
                    **{"X-Route": "24h-latest", "X-Cache": "tilecache-hit", "X-Latest-Day": d},
                )
        except Exception:
            pass

    cog = build_24h_cog(datetime.strptime(d, "%Y%m%d").date())
    png = _tile_png_from_cog(cog, z, x, y, max)

    if max is None:
        try:
            _tile_cache_put("nsidc", 24, d, z, x, y, png)
        except Exception:
            pass

    return _resp_png(
        png,
        cache_control=CACHE_CONTROL_LATEST,
        **{"X-Route": "24h-latest", "X-Cache": ("tilecache-store" if max is None else "bypass-max"), "X-Latest-Day": d},
    )

@app.get("/tiles/24h/{date_yyyymmdd:regex(\\d{8})}/{z}/{x}/{y}.png")
def tiles_24h(date_yyyymmdd: str, z: int, x: int, y: int, max: float | None = Query(None)):
    try:
        d = datetime.strptime(date_yyyymmdd, "%Y%m%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be YYYYMMDD")

    png = _tile_png_from_cog(build_24h_cog(d), z, x, y, max)
    return _resp_png(
        png,
        cache_control=CACHE_CONTROL_BY_DATE,
        **{"X-Route": "24h-date"},
    )

@app.get("/tiles/72h/{end_yyyymmdd:regex(\\d{8})}/{z}/{x}/{y}.png")
def tiles_72h(end_yyyymmdd: str, z: int, x: int, y: int, max: float | None = Query(None)):
    try:
        d = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be YYYYMMDD")

    png = _tile_png_from_cog(build_72h_cog(d), z, x, y, max)
    return _resp_png(
        png,
        cache_control=CACHE_CONTROL_BY_DATE,
        **{"X-Route": "72h"},
    )

@app.get("/forecast/ending")
def forecast_ending(days: int = Query(2, ge=2, le=3), hours: int = Query(24, ge=24, le=72)):
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    end_dt = _ts_to_dt(_valid_ts_for_days(days, now_utc=now))
    return {
        "days": int(days),
        "hours": int(hours),
        "valid": end_dt.strftime("%Y%m%d%H"),
        "end_iso": end_dt.isoformat(),
        "label": f"{int(hours)} Hour snowmelt ending on {end_dt.strftime('%Y-%m-%d %HZ')}",
        "now_utc": now.isoformat(),
        "anchor_date_05z": now.date().isoformat(),
    }

@app.get("/tiles/forecast/latest/{z}/{x}/{y}.png")
def tiles_forecast_latest(
    z: int,
    x: int,
    y: int,
    max: float | None = Query(None),
    days: int = Query(2, ge=2, le=3),
    hours: int = Query(24, ge=24, le=72),
):
    if not _tile_allowed(z, x, y):
        return _resp_png(
            _transparent_png_256(),
            cache_control=CACHE_CONTROL_LATEST,
            **{"X-Route": "forecast-latest", "X-Allowed": "0"},
        )

    sel = _pick_forecast_for_days(days, dom_prefer="zz")
    if not sel.get("ok"):
        detail = dict(sel)
        detail["collab_last"] = dict(_COLLAB_LAST)
        raise HTTPException(status_code=503, detail=detail)

    # Optional server-side tile cache for latest.
    # Key is the resolved valid day (sel["valid"]) + hours + days + tile.
    # This avoids serving old pixels after "latest" advances.
    valid_ymd = (sel["valid"] or "")[:8]  # YYYYMMDD
    if max is None and valid_ymd:
        try:
            hit = _tile_cache_get("zz", int(hours), f"fcstlatest_{int(days)}_{valid_ymd}", z, x, y)
            if hit is not None:
                return _resp_png(
                    hit,
                    cache_control=CACHE_CONTROL_LATEST,
                    **{
                        "X-Route": "forecast-latest",
                        "X-Allowed": "1",
                        "X-Cache": "tilecache-hit",
                        "X-Forecast-Valid": sel["valid"],
                        "X-Forecast-Hours": str(int(hours)),
                        "X-Forecast-RunInit-TT": sel["run_init"],
                    },
                )
        except Exception:
            pass

    if int(hours) == 72:
        melt72_cog = _forecast_melt72h_end_cog(sel["valid"], dom_prefer="zz")

        melt_mm, melt_mask, oob = _tile_arrays_from_cog(melt72_cog, z, x, y)
        if oob or melt_mm is None or melt_mask is None:
            return _resp_png(
                _transparent_png_256(),
                cache_control=CACHE_CONTROL_LATEST,
                **{
                    "X-Route": "forecast-latest",
                    "X-Allowed": "1",
                    "X-OOB": "1",
                    "X-Forecast-Hours": "72",
                    "X-Forecast-Valid": sel["valid"],
                    "X-Forecast-RunInit-TT": sel["run_init"],
                    "X-Forecast-72h-COG": melt72_cog.name,
                },
            )

        snow_urls = _find_fcst_11034_grib2_tt_ts(sel["run_init"], sel["valid"], dom_prefer="zz")
        snow = snow_valid = None
        snow_info = "snowpack_unavailable"
        if snow_urls:
            try:
                snow_cogs = _build_forecast_snowpack_cogs(sel["valid"], snow_urls)
                snow, snow_valid, snow_info = _union_tile(snow_cogs, z, x, y, label="snowpack")
            except Exception as e:
                snow_info = f"snowpack_unavailable:{e!r}"

        png = _melt_to_png(
            melt_mm,
            melt_mask,
            max,
            snow,
            snow_valid,
            snow_allow_min_mm=0.0,
            snow_underlay_min_mm=0.0001,
            dilate_px=2,
            bin_edges_in=BIN_EDGES_72H_IN,
        )

        if max is None and valid_ymd:
            try:
                _tile_cache_put("zz", 72, f"fcstlatest_{int(days)}_{valid_ymd}", z, x, y, png)
            except Exception:
                pass

        return _resp_png(
            png,
            cache_control=CACHE_CONTROL_LATEST,
            **{
                "X-Route": "forecast-latest",
                "X-Forecast-Hours": "72",
                "X-Forecast-Valid": sel["valid"],
                "X-Forecast-RunInit-TT": sel["run_init"],
                "X-Forecast-72h-COG": melt72_cog.name,
                "X-SnowPack-TS": sel["valid"],
                "X-SnowPack-Info": snow_info,
                "X-Allowed": "1",
                "X-OOB": "0",
                "X-Cache": ("tilecache-store" if max is None else "bypass-max"),
            },
        )

    _log("INFO", f"[tiles] /tiles/forecast/latest days={days} tt={sel['run_init']} ts={sel['valid']} lead={sel['lead']}")
    resp = _serve_forecast_tile(
        sel["run_init"],
        sel["valid"],
        sel["lead"],
        sel["melt_urls"],
        sel.get("snowpack_ts") or "",
        sel.get("snowpack_urls") or [],
        z,
        x,
        y,
        max,
    )
    resp.headers["X-Allowed"] = "1"
    resp.headers["Cache-Control"] = CACHE_CONTROL_LATEST
    resp.headers["X-Route"] = "forecast-latest"
    resp.headers["X-Forecast-Hours"] = str(int(hours))

    if max is None and valid_ymd:
        try:
            body = resp.body
            if body:
                _tile_cache_put("zz", int(hours), f"fcstlatest_{int(days)}_{valid_ymd}", z, x, y, body)
                resp.headers["X-Cache"] = "tilecache-store"
        except Exception:
            pass
    else:
        resp.headers["X-Cache"] = "bypass-max"

    return resp
@app.get("/value/forecast/by_date")
def value_forecast_by_date(
    date_yyyymmdd: str = Query(..., regex=r"^\d{8}$"),
    hours: int = Query(24, ge=24, le=72),
    lon: float = Query(...),
    lat: float = Query(...),
    dom: str = Query("zz"),
):
    dom = (dom or "zz").lower()
    valid = _norm_ts(f"{date_yyyymmdd}05")

    if int(hours) == 24:
        sel = _pick_forecast_for_valid(valid, dom_prefer=dom)
        if not sel.get("ok"):
            detail = dict(sel)
            detail["collab_last"] = dict(_COLLAB_LAST)
            raise HTTPException(status_code=503, detail=detail)

        melt_cogs = _build_forecast_melt_cogs(sel["run_init"], sel["valid"], "t0024", sel["melt_urls"])
        mm, info = _union_point_mm(melt_cogs, float(lon), float(lat))
        inches = None if mm is None else (mm / INCH_TO_MM)
        return {"ok": True, "hours": 24, "valid": sel["valid"], "run_init": sel["run_init"], "mm": mm, "inches": inches, "info": info}

    run_init = _pick_best_run_init_for_valid_t0024(valid, dom_prefer=dom)
    if not run_init:
        raise HTTPException(status_code=503, detail={"error": "no_run_init_for_valid", "valid": valid, "dom": dom})

    melt72_cog = _forecast_melt72h_end_cog(valid, dom_prefer=dom)
    mm, info = _point_mm_from_cog(melt72_cog, float(lon), float(lat))
    inches = None if mm is None else (float(mm) / INCH_TO_MM)
    inches_1dp = None if inches is None else round(inches, 1)
    return {
        "ok": True,
        "hours": 24,
        "valid": sel["valid"],
        "run_init": sel["run_init"],
        "mm": None if mm is None else float(mm),
        "inches": inches,          # full precision
        "inches_1dp": inches_1dp,  # for display (2.4, etc.)
        "info": info,
    }


@app.get("/tiles/forecast/by_date/{z}/{x}/{y}.png")
def tiles_forecast_by_date(
    z: int,
    x: int,
    y: int,
    date_yyyymmdd: str = Query(..., regex=r"^\d{8}$"),
    hours: int = Query(24, ge=24, le=72),
    max: float | None = Query(None),
    dom: str = Query("zz"),
):
    if not _tile_allowed(z, x, y):
        return _resp_png(
            _transparent_png_256(),
            cache_control=CACHE_CONTROL_BY_DATE,
            **{"X-Route": "forecast-by-date", "X-Allowed": "0"},
        )

    dom = (dom or "zz").lower()
    valid = _norm_ts(f"{date_yyyymmdd}05")

    if max is None:
        hit = _tile_cache_get(dom, int(hours), date_yyyymmdd, z, x, y)
        if hit is not None:
            return _resp_png(
                hit,
                cache_control=CACHE_CONTROL_BY_DATE,
                **{"X-Route": "forecast-by-date", "X-Allowed": "1", "X-Cache": "tilecache-hit"},
            )

    if int(hours) == 24:
        sel = _pick_forecast_for_valid(valid, dom_prefer=dom)
        if not sel.get("ok"):
            detail = dict(sel)
            detail["collab_last"] = dict(_COLLAB_LAST)
            raise HTTPException(status_code=503, detail=detail)

        resp = _serve_forecast_tile(
            sel["run_init"],
            sel["valid"],
            "t0024",
            sel["melt_urls"],
            sel.get("snowpack_ts") or "",
            sel.get("snowpack_urls") or [],
            z,
            x,
            y,
            max,
        )
        resp.headers["X-Allowed"] = "1"
        resp.headers["X-Route"] = "forecast-by-date"
        resp.headers["X-Forecast-Hours"] = "24"
        resp.headers["Cache-Control"] = CACHE_CONTROL_BY_DATE

        if max is None:
            try:
                body = resp.body
                if body:
                    _tile_cache_put(dom, 24, date_yyyymmdd, z, x, y, body)
                    resp.headers["X-Cache"] = "tilecache-store"
            except Exception:
                pass

        return resp

    run_init = _pick_best_run_init_for_valid_t0024(valid, dom_prefer=dom)
    if not run_init:
        raise HTTPException(status_code=503, detail={"error": "no_run_init_for_valid", "valid": valid, "dom": dom})

    melt72_cog = _forecast_melt72h_end_cog(valid, dom_prefer=dom)

    melt_mm, melt_mask, oob = _tile_arrays_from_cog(melt72_cog, z, x, y)
    if oob or melt_mm is None or melt_mask is None:
        return _resp_png(
            _transparent_png_256(),
            cache_control=CACHE_CONTROL_BY_DATE,
            **{
                "X-Route": "forecast-by-date",
                "X-Allowed": "1",
                "X-OOB": "1",
                "X-Forecast-Hours": "72",
                "X-Forecast-Valid": valid,
                "X-Forecast-RunInit-TT": run_init,
                "X-Forecast-72h-COG": melt72_cog.name,
            },
        )

    snow_urls = _find_fcst_11034_grib2_tt_ts(run_init, valid, dom_prefer=dom)
    snow = snow_valid = None
    snow_info = "snowpack_unavailable"
    if snow_urls:
        try:
            snow_cogs = _build_forecast_snowpack_cogs(valid, snow_urls)
            snow, snow_valid, snow_info = _union_tile(snow_cogs, z, x, y, label="snowpack")
        except Exception as e:
            snow_info = f"snowpack_unavailable:{e!r}"

    png = _melt_to_png(
        melt_mm,
        melt_mask,
        max,
        snow,
        snow_valid,
        snow_allow_min_mm=0.0,
        snow_underlay_min_mm=0.0001,
        dilate_px=2,
        bin_edges_in=BIN_EDGES_72H_IN,
    )

    if max is None:
        _tile_cache_put(dom, 72, date_yyyymmdd, z, x, y, png)

    return _resp_png(
        png,
        cache_control=CACHE_CONTROL_BY_DATE,
        **{
            "X-Route": "forecast-by-date",
            "X-Forecast-Hours": "72",
            "X-Forecast-Valid": valid,
            "X-Forecast-RunInit-TT": run_init,
            "X-Forecast-72h-COG": melt72_cog.name,
            "X-SnowPack-TS": valid,
            "X-SnowPack-INFO": snow_info,
            "X-Allowed": "1",
            "X-OOB": "0",
            "X-Cache": ("tilecache-store" if max is None else "bypass-max"),
        },
    )

@app.get("/tiles/forecast/{ts_key:regex(\\d{8}(?:[_-]?\\d{2}))}/{z}/{x}/{y}.png")
def tiles_forecast(
    ts_key: str,
    z: int,
    x: int,
    y: int,
    max: float | None = Query(None),
    lead: str = Query("t0024"),
    valid: str | None = Query(None),
    dom: str = Query("zz"),
):
    if not _tile_allowed(z, x, y):
        return _resp_png(_transparent_png_256(), **{"X-Route": "forecast-direct", "X-Allowed": "0"})

    ts_key = _norm_ts(ts_key)
    lead = (lead or "").lower()
    if lead != "t0024":
        raise HTTPException(status_code=400, detail="This service now serves direct 24h melt files (t0024). Use lead=t0024.")
    if not valid:
        raise HTTPException(status_code=400, detail="valid is required (TSYYYYMMDDHH) for direct 24h melt")
    valid = _norm_ts(valid)

    melt_urls = _find_melt_11044_t0024(ts_key, valid, dom_prefer=dom)
    if not melt_urls:
        raise HTTPException(status_code=404, detail=f"No melt file for TT={ts_key} TS={valid} lead=t0024")

    sd = _best_state_nc_near_valid("ssmv11036", valid, dom_prefer=dom)
    if not sd.get("url"):
        raise HTTPException(status_code=404, detail="No snowdepth 11036 state .nc found")

    resp = _serve_forecast_tile(ts_key, valid, "t0024", melt_urls, sd.get("picked_ts") or "", [sd["url"]], z, x, y, max)
    resp.headers["X-Allowed"] = "1"
    return resp

@app.get("/forecast/latest_ts")
def forecast_latest_ts(dom: str = "zz"):
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    sel2 = _pick_forecast_for_days(2, dom_prefer=dom)
    sel3 = _pick_forecast_for_days(3, dom_prefer=dom)
    return {"now_utc": now.isoformat(), "d2": sel2, "d3": sel3, "collab_last": dict(_COLLAB_LAST)}


@app.get("/forecast/selection")
def forecast_selection(dom: str = "zz"):
    return forecast_latest_ts(dom=dom)


@app.get("/forecast/avail")
def forecast_avail(lead: str = "t0024"):
    html = _collab_dirlist()
    names = _hrefs(html)
    lead = (lead or "").lower()
    tts: list[str] = []
    for n in names:
        base = n.split("?", 1)[0].strip()
        low = base.lower()
        if "ssmv11044" not in low:
            continue
        if lead and lead not in low:
            continue
        if not _is_data_name(low):
            continue
        tt = _tt_from_name(base)
        if tt:
            tts.append(tt)
    tts = sorted(set(tts))
    return {"lead": lead, "count": len(tts), "min_tt": tts[0] if tts else None, "max_tt": tts[-1] if tts else None, "sample_tail": tts[-10:] if len(tts) > 10 else tts}


@app.get("/forecast/available_days")
def forecast_available_days(dom: str = "zz", lookback_days: int = 5):
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    min_dt = (now.date() - timedelta(days=int(lookback_days)))
    max_dt = None

    ts_all = _list_available_valid_ts_t0024(dom_prefer=dom)
    dates = []
    for ts10 in ts_all:
        dt = _ts_to_dt(ts10)
        d = dt.date()
        if d < min_dt:
            continue
        dates.append(d.strftime("%Y%m%d"))
        if max_dt is None or d > max_dt:
            max_dt = d

    dates = sorted(set(dates))
    return {
        "now_utc": now.isoformat(),
        "lookback_days": int(lookback_days),
        "min_date": min_dt.strftime("%Y%m%d"),
        "max_date": (max_dt.strftime("%Y%m%d") if max_dt else None),
        "available_dates": dates,
    }

@app.get("/__debug/prewarm_cn")
def __debug_prewarm_cn(
    days: int = Query(2, ge=2, le=3),
    hours: int = Query(24, ge=24, le=72),
    dom: str = Query("zz"),
    zmin: int = Query(7, ge=0, le=14),
    zmax: int = Query(11, ge=0, le=14),
    radius: int = Query(1, ge=0, le=4),
    cap_per_zoom: int = Query(6000, ge=100, le=20000),
    max_tiles_total: int = Query(20000, ge=100, le=100000),
):
    return _prewarm_cn_corridor_tiles(
        days=days,
        hours=hours,
        dom=dom,
        zmin=zmin,
        zmax=zmax,
        radius=radius,
        cap_per_zoom=cap_per_zoom,
        max_tiles_total=max_tiles_total,
    )
@app.get("/__debug/roll_now")
def __debug_roll_now():
    return _run_daily_roll_job_once()

@app.get("/__debug/tile_cache_stats")
def __debug_tile_cache_stats():
    total_files = 0
    total_bytes = 0
    if TILE_CACHE_DIR.exists():
        for root, _, files in os.walk(TILE_CACHE_DIR):
            for fn in files:
                if not fn.endswith(".png"):
                    continue
                total_files += 1
                try:
                    total_bytes += Path(root, fn).stat().st_size
                except Exception:
                    pass
    return {"dir": TILE_CACHE_DIR.as_posix(), "png_files": total_files, "bytes": total_bytes}

@app.get("/__logs")
def __logs(since_ms: int | None = None, limit: int = 200):
    try:
        limit = max(1, min(int(limit), 1000))
    except Exception:
        limit = 200
    rows = list(_LOG)
    if since_ms is not None:
        try:
            s = int(since_ms)
            rows = [r for r in rows if r["ts"] > s]
        except Exception:
            pass
    return {"count": len(rows[-limit:]), "items": rows[-limit:]}

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("UVICORN_WORKERS", "3") or "3")

    module = (os.environ.get("APP_MODULE", "") or "").strip()

    if workers > 1 and not module:
        raise SystemExit(
            "UVICORN_WORKERS>1 requires APP_MODULE like 'main:app'. "
            "Example: APP_MODULE=main:app UVICORN_WORKERS=3 python main.py"
        )

    if module:
        uvicorn.run(module, host=host, port=port, workers=workers, log_level="info")
    else:
        uvicorn.run(app, host=host, port=port, workers=1, log_level="info")
