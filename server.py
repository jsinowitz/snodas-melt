from __future__ import annotations

import io
import os
import re
import tarfile
import gzip
import tempfile
import calendar
import sys
import datetime as dt
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Iterable
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rio_cogeo.cogeo import cog_translate
from rio_tiler.io import COGReader
from rio_tiler.profiles import img_profiles
import xarray as xr
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.staticfiles import StaticFiles
from starlette.responses import PlainTextResponse, JSONResponse
from rio_tiler.utils import render
import logging
from collections import deque
from functools import lru_cache

# ------------ cache/env ------------
def pick_cache_root() -> Path:
    env = (os.environ.get("SNODAS_CACHE") or "").strip()
    for c in ([env] if env else []) + [
        "/home/user/snodas-cache",
        "/tmp/snodas-cache",
        "./cache",
    ]:
        p = Path(c).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".wtest"
            t.write_text("ok")
            t.unlink(missing_ok=True)
            return p.resolve()
        except Exception:
            pass
    return Path(tempfile.mkdtemp(prefix="snodas-cache-"))


CACHE = pick_cache_root()
CFGRIB_DIR = CACHE / "cfgrib"
CFGRIB_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CFGRIB_INDEXPATH", str(CFGRIB_DIR))

COLLAB_BASE = "https://www.nohrsc.noaa.gov/pub/products/collaborators"
NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02158"

# ------------ app ------------
app = FastAPI(title="SNODAS Snowmelt Tiles", version="1.0.0")
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# ------------ logging (ring buffer) ------------
_LOG_DEQUE = deque(maxlen=1000)


class _RingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = f"[logging-format-error] {record.__dict__!r}"
        ts = int(record.created * 1000)
        _LOG_DEQUE.append({"ts": ts, "level": record.levelname, "msg": msg})


def _setup_logger():
    logger = logging.getLogger("snodas")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    h1 = logging.StreamHandler(sys.stdout)
    h1.setFormatter(fmt)
    h2 = _RingHandler()
    h2.setFormatter(fmt)
    logger.addHandler(h1)
    logger.addHandler(h2)
    logger.propagate = False
    return logger


log = _setup_logger()

# ------------ small utils ------------
def yyyymmdd(date: dt.date) -> str:
    return date.strftime("%Y%m%d")


def _http_get(url: str, timeout=180) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "snodas/1.0"})
    r.raise_for_status()
    return r


_TS_10 = re.compile(r"^\d{10}$")
_TS_8u2 = re.compile(r"^(\d{8})[_-](\d{2})$")


def _normalize_ts_key(ts_key: str) -> str:
    """
    Accept YYYYMMDDHH or YYYYMMDD_HH / YYYYMMDD-HH and normalize to YYYYMMDDHH.
    """
    if _TS_10.fullmatch(ts_key):
        return ts_key
    m = _TS_8u2.fullmatch(ts_key)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    # 404 reduces aggressive retries vs 400
    raise HTTPException(status_code=404, detail="unknown forecast key")


# ------------ NSIDC daily ------------
def _nsidc_month_folder(d: dt.date) -> str:
    m = d.month
    return f"{m:02d}_{calendar.month_abbr[m].title()}"


def _nsidc_file_url(date: dt.date, masked: bool) -> str:
    sub = "masked" if masked else "unmasked"
    dstr = yyyymmdd(date)
    return (
        f"{NSIDC_BASE}/{date.year}/{_nsidc_month_folder(date)}/"
        f"{dstr}/SNODAS_{sub}_{dstr}.tar"
    )


def nsidc_exists(date: dt.date) -> tuple[bool, str]:
    for masked in (False, True):
        url = _nsidc_file_url(date, masked)
        try:
            r = requests.get(
                url, stream=True, timeout=30, headers={"User-Agent": "snodas/1.0"}
            )
            r.close()
            if r.status_code == 200:
                return True, url
        except Exception:
            pass
    return False, _nsidc_file_url(date, masked=False)


def nsidc_daily_tar_url(date: dt.date) -> str:
    return nsidc_exists(date)[1]


def _download_day_tar(date: dt.date) -> Path:
    dstr = yyyymmdd(date)
    tar_path = CACHE / f"SNODAS_{dstr}.tar"
    if tar_path.exists() and tar_path.stat().st_size > 0:
        return tar_path
    ok, url = nsidc_exists(date)
    if not ok:
        raise HTTPException(status_code=404, detail=f"NSIDC not available for {dstr}")
    r = _http_get(url)
    tar_path.write_bytes(r.content)
    return tar_path


def _extract_var1044(tar_path: Path) -> tuple[Path, Path]:
    tempdir = CACHE / (tar_path.stem + "_extracted")
    tempdir.mkdir(exist_ok=True)
    hdr_path = dat_path = None

    with tarfile.open(tar_path) as tf:
        members = [
            m
            for m in tf.getmembers()
            if "1044" in m.name and (m.name.endswith(".Hdr") or m.name.endswith(".dat"))
        ]
        if not members:
            raise HTTPException(
                status_code=500,
                detail="Var 1044 (.Hdr/.dat) not found in TAR.",
            )

        for m in members:
            outp = tempdir / Path(m.name).name
            if not outp.exists():
                tf.extract(m, path=tempdir)
            if outp.suffix.lower() == ".hdr":
                hdr_path = outp
            elif outp.suffix.lower() == ".dat":
                dat_path = outp

    if not hdr_path or not dat_path:
        raise HTTPException(
            status_code=500,
            detail="Missing .Hdr or .dat for var 1044.",
        )

    return hdr_path, dat_path


# ------------ ESRI BIL / GRIB helpers ------------
_KEYVAL_RE = re.compile(
    r"^\s*(?P<k>[A-Za-z0-9_ ].*?)[=:]?\s+(?P<v>.+?)\s*$"
)  # key = value lines


def _parse_esri_hdr(hdr_path: Path) -> dict:
    try:
        txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = hdr_path.read_text(errors="ignore")
    meta = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        m = _KEYVAL_RE.match(line)
        if not m:
            continue
        k = m.group("k").strip().lower()
        v = m.group("v").strip()
        v = v.split("#", 1)[0].split(";", 1)[0].strip()
        meta[k] = v
        meta[re.sub(r"[^a-z0-9]+", "", k)] = v
    return meta


def _to_int(meta, *keys, default=None):
    for k in keys:
        v = meta.get(k)
        if v is None:
            continue
        try:
            return int(float(v))
        except Exception:
            pass
    return default


def _to_float(meta, *keys, default=None):
    for k in keys:
        v = meta.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return default


def _dtype_from_esri(nbits: int, pixeltype: str, byteorder: str):
    pixeltype = (pixeltype or "").upper()
    byteorder = (byteorder or "I").upper()
    endian = "<" if byteorder in ("I", "LSBFIRST", "LITTLEENDIAN") else ">"
    if pixeltype == "FLOAT":
        if nbits == 32:
            return np.dtype(endian + "f4")
        if nbits == 64:
            return np.dtype(endian + "f8")
    if pixeltype in ("UNSIGNEDINT", "UNSIGNED", "UINTEGER"):
        if nbits == 8:
            return np.uint8
        if nbits == 16:
            return np.dtype(endian + "u2")
        if nbits == 32:
            return np.dtype(endian + "u4")
    if pixeltype in ("SIGNEDINT", "SIGNED", "INTEGER"):
        if nbits == 8:
            return np.int8
        if nbits == 16:
            return np.dtype(endian + "i2")
        if nbits == 32:
            return np.dtype(endian + "i4")
    return np.dtype(endian + "f4")


def _bil_to_geotiff(hdr_path: Path, bil_path: Path, out_tif: Path) -> Path:
    import math

    # Read header text once
    try:
        hdr_txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        hdr_txt = hdr_path.read_text(errors="ignore")

    meta = _parse_esri_hdr(hdr_path)

    # ---- NCOLS / NROWS from metadata ----
    ncols = _to_int(
        meta,
        "ncols",
        "samples",
        "nsamples",
        "columns",
        "number of columns",
        "numberofcolumns",
        "cols",
    )
    nrows = _to_int(
        meta,
        "nrows",
        "lines",
        "nlines",
        "rows",
        "number of rows",
        "numberofrows",
    )

    # ---- regex fallback for "Number of columns/rows:" ----
    if not ncols or not nrows:
        m_cols = re.search(r"Number of columns:\s*([0-9]+)", hdr_txt, re.IGNORECASE)
        m_rows = re.search(r"Number of rows:\s*([0-9]+)", hdr_txt, re.IGNORECASE)
        if m_cols and not ncols:
            try:
                ncols = int(m_cols.group(1))
            except Exception:
                pass
        if m_rows and not nrows:
            try:
                nrows = int(m_rows.group(1))
            except Exception:
                pass

    # ---- bands / bits / dtype ----
    bands = _to_int(
        meta,
        "bands",
        "nbands",
        "number of bands",
        "numberofbands",
        default=1,
    ) or 1

    nbits = _to_int(
        meta,
        "nbits",
        "bitspersample",
        "bits per sample",
        "bits",
        default=32,
    ) or 32

    if meta.get("data bytes per pixel"):
        try:
            nbits = int(float(meta["data bytes per pixel"])) * 8
        except Exception:
            pass

    pixeltype = (
        meta.get("pixeltype")
        or meta.get("pixel type")
        or meta.get("data type")
        or ("FLOAT" if nbits == 32 else "UNSIGNEDINT")
    )
    byteorder = meta.get("byteorder") or meta.get("byte order") or "I"

    # robust slope/intercept defaults; avoid NoneType issues
    intercept = _to_float(meta, "data intercept", "intercept", default=0.0)
    slope = _to_float(meta, "data slope", "slope", default=1.0)
    if intercept is None:
        intercept = 0.0
    if slope is None:
        slope = 1.0

    # ---- last-resort NCOLS/NROWS from extents/resolution ----
    if not (ncols and nrows):
        xres = _to_float(
            meta,
            "x-axis resolution",
            "x axis resolution",
            "xdim",
            "x dim",
            "x dimension",
            "cellsize",
        )
        yres = _to_float(
            meta,
            "y-axis resolution",
            "y axis resolution",
            "ydim",
            "y dim",
            "y dimension",
            "cellsize",
        )
        xmin = _to_float(
            meta,
            "minimum x-axis coordinate",
            "minimum x axis coordinate",
            "min x",
            "minx",
            "minimumx",
        )
        xmax = _to_float(
            meta,
            "maximum x-axis coordinate",
            "maximum x axis coordinate",
            "max x",
            "maxx",
            "maximumx",
        )
        ymin = _to_float(
            meta,
            "minimum y-axis coordinate",
            "minimum y axis coordinate",
            "min y",
            "miny",
            "minimumy",
        )
        ymax = _to_float(
            meta,
            "maximum y-axis coordinate",
            "maximum y axis coordinate",
            "max y",
            "maxy",
            "maximumy",
        )
        if xres and xmin is not None and xmax is not None and not ncols:
            try:
                ncols = int(round((xmax - xmin) / xres))
            except Exception:
                pass
        if yres and ymin is not None and ymax is not None and not nrows:
            try:
                nrows = int(round((ymax - ymin) / yres))
            except Exception:
                pass

    # ---- final fallback: bytes-based + brute force factor search ----
    bytes_per_sample = max(1, nbits // 8)
    file_bytes = bil_path.stat().st_size
    total_samples = file_bytes // bytes_per_sample

    header_ncols = ncols
    header_nrows = nrows

    bandrowbytes = _to_int(meta, "bandrowbytes", "band row bytes")
    totalrowbytes = _to_int(meta, "totalrowbytes", "total row bytes")
    if bandrowbytes and not ncols:
        try:
            ncols = bandrowbytes // bytes_per_sample
        except Exception:
            pass
    if totalrowbytes and not nrows:
        try:
            nrows = total_samples if ncols is None else total_samples // ncols
        except Exception:
            pass

    def _pick_dims_from_factors(total, hdr_cols, hdr_rows):
        best = None  # (score, rows, cols)
        root = int(math.sqrt(total)) + 1
        for c in range(1, root + 1):
            if total % c != 0:
                continue
            r = total // c
            score = 0.0
            if hdr_cols:
                score += abs(c - hdr_cols) / float(hdr_cols)
            if hdr_rows:
                score += abs(r - hdr_rows) / float(hdr_rows)
            if hdr_cols and hdr_rows:
                score += abs((r / c) - (hdr_rows / hdr_cols)) * 0.1
            if best is None or score < best[0]:
                best = (score, r, c)
        if best is None:
            return None, None
        return best[1], best[2]

    if header_ncols and header_nrows:
        expected = header_ncols * header_nrows * max(1, bands)
        if expected != total_samples:
            r_guess, c_guess = _pick_dims_from_factors(
                total_samples, header_ncols, header_nrows
            )
            if r_guess and c_guess:
                nrows, ncols = r_guess, c_guess

    if not (ncols and nrows):
        r_guess, c_guess = _pick_dims_from_factors(
            total_samples, header_ncols, header_nrows
        )
        if r_guess and c_guess:
            nrows, ncols = r_guess, c_guess

    if not (ncols and nrows):
        snippet = "\n".join(hdr_txt.splitlines()[:120])
        raise HTTPException(
            status_code=502,
            detail=f"BIL header indeterminate NCOLS/NROWS\n{snippet}",
        )

    # ---- georeferencing (more tolerant) ----
    # Try a variety of keys for upper-left or lower-left + cellsize
    ulx = _to_float(meta, "ulxmap", "upper left x", "ulx", default=None)
    uly = _to_float(meta, "ulymap", "upper left y", "uly", default=None)

    xmin = _to_float(
        meta,
        "minimum x-axis coordinate",
        "minimum x axis coordinate",
        "min x",
        "minx",
        "xllcorner",
        "xllcenter",
        "x origin",
        default=None,
    )
    xmax = _to_float(
        meta,
        "maximum x-axis coordinate",
        "maximum x axis coordinate",
        "max x",
        "maxx",
        default=None,
    )
    ymin = _to_float(
        meta,
        "minimum y-axis coordinate",
        "minimum y axis coordinate",
        "min y",
        "miny",
        "yllcorner",
        "yllcenter",
        "y origin",
        default=None,
    )
    ymax = _to_float(
        meta,
        "maximum y-axis coordinate",
        "maximum y axis coordinate",
        "max y",
        "maxy",
        "maximumy",
        default=None,
    )

    xres = _to_float(
        meta,
        "x-axis resolution",
        "x axis resolution",
        "xdim",
        "x dim",
        "x dimension",
        "cellsize",
        default=0.0083333333,
    )
    yres = _to_float(
        meta,
        "y-axis resolution",
        "y axis resolution",
        "ydim",
        "y dim",
        "y dimension",
        "cellsize",
        default=0.0083333333,
    )

    # Derive ULX/ULY from extents if needed
    if ulx is None and xmin is not None:
        ulx = xmin
    if uly is None and ymax is not None:
        uly = ymax

    # If only lower-left and resolution are known, compute upper-left
    if uly is None and ymin is not None:
        try:
            uly = ymin + abs(yres) * nrows
        except Exception:
            pass

    # Last-resort: if still missing, pick 0,0 so we can at least build a COG
    if ulx is None:
        ulx = 0.0
    if uly is None:
        uly = 0.0

    dtp = _dtype_from_esri(nbits, pixeltype, byteorder)
    count = ncols * nrows * max(1, bands)

    arr = np.fromfile(bil_path, dtype=dtp, count=count)

    if arr.size != count:
        total_samples = bil_path.stat().st_size // bytes_per_sample
        samples_per_band = total_samples // max(1, bands)
        r_guess, c_guess = _pick_dims_from_factors(samples_per_band, ncols, nrows)
        if (
            r_guess
            and c_guess
            and r_guess * c_guess * max(1, bands) == total_samples
        ):
            nrows, ncols = r_guess, c_guess
            count = ncols * nrows * max(1, bands)
            arr = np.fromfile(bil_path, dtype=dtp, count=count)
        if arr.size != count:
            raise HTTPException(
                status_code=502,
                detail=(
                    "BIL read size mismatch "
                    f"(ncols={ncols}, nrows={nrows}, header_bands={bands}, "
                    f"expected={count}, got={arr.size}, "
                    f"file_bytes={file_bytes}, total_samples={total_samples})"
                ),
            )

    # ---- reshape ----
    if bands == 1:
        arr = arr.reshape((nrows, ncols))
    else:
        layout = (meta.get("layout") or "bil").upper()
        if layout == "BSQ":
            arr = arr.reshape((bands, nrows, ncols))
        else:
            arr = arr.reshape((nrows, bands, ncols)).transpose(1, 0, 2)

    # ---- apply scale/offset safely ----
    try:
        s = float(slope)
    except Exception:
        s = 1.0
    try:
        b = float(intercept)
    except Exception:
        b = 0.0

    if (s != 1.0) or (b != 0.0):
        arr = arr.astype("float32") * s + b
    else:
        arr = arr.astype("float32")

    transform = rasterio.Affine(xres, 0.0, ulx, 0.0, -abs(yres), uly)
    profile = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1 if arr.ndim == 2 else arr.shape[0],
        "dtype": arr.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "DEFLATE",
        "tiled": True,
        "nodata": _to_float(
            meta,
            "no data value",
            "nodata",
            "nodata_value",
            default=None,
        ),
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
        else:
            for bidx in range(arr.shape[0]):
                dst.write(arr[bidx], bidx + 1)

    return out_tif


def _open_bil_as_da(hdr_path: Path, bil_path: Path) -> xr.DataArray:
    tmp_tif = bil_path.with_suffix(".tif")
    tiff = _bil_to_geotiff(hdr_path, bil_path, tmp_tif)
    da = xr.open_dataarray(tiff.as_posix(), engine="rasterio")
    try:
        if getattr(da, "rio", None) and da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326", inplace=False)
    except Exception:
        pass
    return da.astype("float32")


def _open_grib2_as_da(path: Path) -> xr.DataArray:
    idx = (CFGRIB_DIR / (path.name + ".idx")).as_posix()
    ds = xr.open_dataset(
        path.as_posix(),
        engine="cfgrib",
        backend_kwargs={"indexpath": idx},
    )
    pick = None
    for v in ds.data_vars:
        nm = v.lower()
        if any(k in nm for k in ("melt", "tsm", "smlt")):
            pick = v
            break
    if pick is None:
        for v in ds.data_vars:
            nm = v.lower()
            if any(k in nm for k in ("swe", "sdwe", "snow_water")):
                pick = v
                break
    if pick is None:
        pick = next(iter(ds.data_vars))
    da = ds[pick]
    if "step" in da.dims:
        da = da.isel(step=0)
    fv = da.attrs.get("_FillValue")
    if fv is not None:
        da = da.where(da != fv)
    return da.astype("float32")


def _candidate_melt_urls_for_ts(ts_key: str) -> list[str]:
    """
    Snowmelt-specific discovery, restricted to true melt products.

    We are targeting modeled snowmelt liquid water equivalent, which
    corresponds to product code ssmv11044. We explicitly exclude
    non-melt fields like ssmv11038 (temperature) etc.

    Strategy:
      1) Scan the collaborator directory listing for ssmv11044 files
         with archive extensions.
      2) Extract TS(YYYYMMDDHH) from names.
      3) Pick the subset whose TS is closest to `ts_key` in time,
         but only if within MAX_DELTA_HOURS.
    """
    try:
        r = requests.get(
            COLLAB_BASE + "/",
            timeout=15,
            headers={"User-Agent": "snodas-forecast/1.0"},
        )
    except Exception as e:
        log.info(f"[melt-discovery] dirlist exception {e!r}")
        return []

    if r.status_code != 200 or not (r.text or "").strip():
        log.info(
            f"[melt-discovery] dirlist HTTP {r.status_code}, "
            f"len={len(r.text or '')}"
        )
        return []

    txt = r.text or ""
    href_pat = re.compile(
        r'href\s*=\s*(?P<q>[\'"])(?P<u>.*?)(?P=q)', re.IGNORECASE
    )
    names = [m.group("u") for m in href_pat.finditer(txt)]
    if not names:
        bare = re.compile(r'href\s*=\s*([^ >]+)', re.IGNORECASE)
        names = [m.group(1) for m in bare.finditer(txt)]

    if not names:
        log.info("[melt-discovery] no hrefs in dir listing")
        return []

    allowed_exts = (".grz", ".tar", ".tar.gz", ".tgz")

    # STRICT melt product codes: only true snowmelt LE
    melt_codes = ("ssmv11044",)

    ts_re = re.compile(r"TS(\d{10})", re.IGNORECASE)

    try:
        target_dt = datetime.strptime(ts_key, "%Y%m%d%H").replace(
            tzinfo=timezone.utc
        )
    except Exception as e:
        log.info(f"[melt-discovery] bad ts_key={ts_key}: {e!r}")
        return []

    candidates: list[tuple[float, str, str]] = []
    # (abs_delta_hours, ts_found_str, url)

    for n in names:
        base = n.strip().split("?", 1)[0]
        low = base.lower()

        if not any(code in low for code in melt_codes):
            continue
        if not any(low.endswith(ext) for ext in allowed_exts):
            continue

        m = ts_re.search(base)
        if not m:
            continue

        ts_found = m.group(1)
        try:
            dt_found = datetime.strptime(ts_found, "%Y%m%d%H").replace(
                tzinfo=timezone.utc
            )
        except Exception:
            continue

        delta_hours = abs((dt_found - target_dt).total_seconds()) / 3600.0
        url = base if base.startswith("http") else f"{COLLAB_BASE}/{base}"
        candidates.append((delta_hours, ts_found, url))

    if not candidates:
        log.info(f"[melt-discovery] ts={ts_key} -> 0 MELT candidates (raw)")
        return []

    candidates.sort(key=lambda t: t[0])
    best_delta = candidates[0][0]
    best_ts = candidates[0][1]

    MAX_DELTA_HOURS = 48.0
    if best_delta > MAX_DELTA_HOURS:
        log.info(
            f"[melt-discovery] ts={ts_key} -> nearest MELT TS={best_ts} "
            f"but delta={best_delta:.1f}h (>{MAX_DELTA_HOURS:.0f}h); ignoring"
        )
        return []

    best_urls = [url for d, ts, url in candidates if ts == best_ts]

    log.info(
        f"[melt-discovery] ts={ts_key} -> {len(best_urls)} MELT candidates "
        f"using TS={best_ts} (Δ={best_delta:.1f}h)"
    )
    return sorted(set(best_urls))


def _da_to_cog_3857(da: xr.DataArray, out_cog: Path) -> Path:
    lon_name = next(
        (n for n in da.coords if n.lower() in ("lon", "longitude", "x")), None
    )
    lat_name = next(
        (n for n in da.coords if n.lower() in ("lat", "latitude", "y")), None
    )
    if getattr(da, "rio", None) is None:
        raise HTTPException(status_code=500, detail="rioxarray missing")
    try:
        crs = da.rio.crs
    except Exception:
        crs = None
    if crs is None:
        if lon_name and lat_name:
            da = da.rio.set_spatial_dims(
                x_dim=lon_name, y_dim=lat_name, inplace=False
            )
            da = da.rio.write_crs("EPSG:4326", inplace=False)
        else:
            raise HTTPException(status_code=500, detail="grid lacks lon/lat")
    da3857 = da.rio.reproject("EPSG:3857", nodata=np.nan)
    tmp = out_cog.with_suffix(".tmp.tif")
    da3857.rio.to_raster(
        tmp, dtype="float32", compress="DEFLATE", nodata=np.nan
    )
    cog_translate(
        tmp.as_posix(),
        out_cog.as_posix(),
        img_profiles.get("deflate"),
        in_memory=False,
        quiet=True,
    )
    Path(tmp).unlink(missing_ok=True)
    return out_cog


def _open_any_grid_as_da(desc: dict) -> xr.DataArray:
    t = desc.get("type")
    if t == "grib2":
        return _open_grib2_as_da(desc["path"])
    if t == "bil":
        return _open_bil_as_da(desc["hdr"], desc["dat"])
    raise HTTPException(status_code=500, detail=f"unknown grid type: {t}")


def _hdr_from_grz_bytes(blob: bytes) -> str | None:
    if len(blob) >= 2 and blob[0] == 0x1F and blob[1] == 0x8B:
        try:
            blob = gzip.decompress(blob)
        except Exception:
            return None
    if blob.startswith(b"GRIB"):
        return None
    try:
        bio = io.BytesIO(blob)
        with tarfile.open(fileobj=bio, mode="r:*") as tf:
            for m in tf.getmembers():
                if m.name.lower().endswith(".hdr"):
                    with tf.extractfile(m) as f:
                        return f.read().decode("utf-8", "ignore")
    except Exception:
        return None
    return None


def _desc_from_hdr_text(hdr_text: str | None) -> str:
    if not hdr_text:
        return ""
    for line in hdr_text.splitlines():
        if line.lower().startswith("description"):
            return line.split(":", 1)[-1].strip().lower()
    return ""


# ------------ horizon helpers ------------
_VALID_HOURS = ["05", "06", "11", "12", "17", "18"]
_RUN_HOURS = ["00", "06", "12", "18"]
# broaden prefixes slightly: both lower/upper
_PREFIXES = [
    "zz",
    "us",
    "nw",
    "ne",
    "nc",
    "sw",
    "se",
    "sc",
    "wc",
    "rc",
    "ZZ",
    "US",
    "NW",
    "NE",
    "NC",
    "SW",
    "SE",
    "SC",
    "WC",
    "RC",
]
_PRODUCTS = ["ssmv11044", "ssmv11034", "ssmv01020"]
_HP_SUFFIXES = [f"HP{n:03d}" for n in range(0, 5)] + [""]
_EXTS = [".grz", ".tar", ".tar.gz", ".tgz"]


def _candidate_urls_current(ts_key: str) -> Iterable[str]:
    ts_dt = datetime.strptime(ts_key, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    run_dates = [ts_dt.date(), (ts_dt - timedelta(days=1)).date()]
    seen = set()
    for run_date in run_dates:
        run_ds = run_date.strftime("%Y%m%d")
        for run_hh in _RUN_HOURS:
            for pre in _PREFIXES:
                for prod in _PRODUCTS:
                    for hp in _HP_SUFFIXES:
                        for ext in _EXTS:
                            for url in (
                                f"{COLLAB_BASE}/{pre}_{prod}STT{run_ds}{run_hh}TS{ts_key}{hp}{ext}",
                                f"{COLLAB_BASE}/{pre}_{prod}TS{ts_key}STT{run_ds}{run_hh}{hp}{ext}",
                            ):
                                if url not in seen:
                                    seen.add(url)
                                    yield url


def _candidate_ts_keys_for_horizon(
    days_ahead: int, now_utc: datetime | None = None
) -> list[str]:
    if days_ahead < 1:
        days_ahead = 1
    elif days_ahead > 3:
        days_ahead = 3
    if now_utc is None:
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    day0 = (now_utc + timedelta(days=days_ahead)).date()
    day1 = (now_utc + timedelta(days=days_ahead + 1)).date()
    out: list[str] = []
    for d in (day0, day1):
        ds = d.strftime("%Y%m%d")
        for hh in _VALID_HOURS:
            out.append(f"{ds}{hh}")
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


# ------------ collaborators discovery ------------
_WIDE_PREFIXES = [
    "zz",
    "us",
    "nw",
    "ne",
    "nc",
    "sw",
    "se",
    "sc",
    "wc",
    "rc",
    "ZZ",
    "US",
    "NW",
    "NE",
    "NC",
    "SW",
    "SE",
    "SC",
    "WC",
    "RC",
]
_WIDE_PRODUCTS = ["ssmv11044", "ssmv11034", "ssmv01020", "ssmv11005"]
_WIDE_HP = [f"HP{n:03d}" for n in range(0, 6)] + [""]
_WIDE_EXT = [".grz", ".tar", ".tar.gz", ".tgz"]
_WIDE_RUN_HH = ["00", "06", "12", "18"]


def _candidate_urls_wide(ts_key: str) -> Iterable[str]:
    ts_dt = datetime.strptime(ts_key, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    for off_days in (0, 1, 2, 3):
        run_date = (ts_dt - timedelta(days=off_days)).strftime("%Y%m%d")
        for run_hh in _WIDE_RUN_HH:
            for pre in _WIDE_PREFIXES:
                for prod in _WIDE_PRODUCTS:
                    for hp in _WIDE_HP:
                        for ext in _WIDE_EXT:
                            yield (
                                f"{COLLAB_BASE}/{pre}_{prod}STT{run_date}"
                                f"{run_hh}TS{ts_key}{hp}{ext}"
                            )
                            yield (
                                f"{COLLAB_BASE}/{pre}_{prod}TS{ts_key}"
                                f"STT{run_date}{run_hh}{hp}{ext}"
                            )


def _discover_candidates_via_listing(ts_key: str) -> list[str]:
    out = []
    try:
        r = requests.get(
            COLLAB_BASE + "/",
            timeout=15,
            headers={"User-Agent": "snodas-forecast/1.0"},
        )
        if r.status_code != 200 or not r.text:
            return out
        pat = re.compile(r'href="([^"]+)"')
        names = [m.group(1) for m in pat.finditer(r.text)]
        names = [
            n
            for n in names
            if n.endswith((".grz", ".tar", ".tar.gz", ".tgz")) and ts_key in n
        ]
        for n in names:
            out.append(n if n.startswith("http") else f"{COLLAB_BASE}/{n}")
    except Exception:
        return []
    return out


def _candidate_swe_urls_for_ts(ts_key: str) -> list[str]:
    """
    SWE-specific discovery:

    1) Scan the collaborator directory listing for SWE-like products
       (ssmv11034, ssmv11005) with archive extensions.
    2) Extract the TS(YYYYMMDDHH) time from each filename.
    3) Choose the subset whose TS is closest to `ts_key` in time.

    This allows us to pick SWE fields like TS=2025120417 for a target
    ts_key=2025120405 if there is no exact match, instead of returning
    zero candidates.
    """
    try:
        r = requests.get(
            COLLAB_BASE + "/",
            timeout=15,
            headers={"User-Agent": "snodas-forecast/1.0"},
        )
    except Exception as e:
        log.info(f"[swe-discovery] dirlist exception {e!r}")
        return []

    if r.status_code != 200 or not (r.text or "").strip():
        log.info(
            f"[swe-discovery] dirlist HTTP {r.status_code}, "
            f"len={len(r.text or '')}"
        )
        return []

    txt = r.text or ""
    href_pat = re.compile(r'href\s*=\s*(?P<q>[\'"])(?P<u>.*?)(?P=q)', re.IGNORECASE)
    names = [m.group("u") for m in href_pat.finditer(txt)]

    if not names:
        bare = re.compile(r'href\s*=\s*([^ >]+)', re.IGNORECASE)
        names = [m.group(1) for m in bare.finditer(txt)]

    if not names:
        log.info("[swe-discovery] no hrefs in dir listing")
        return []

    allowed_exts = (".grz", ".tar", ".tar.gz", ".tgz")
    swe_codes = ("ssmv11034", "ssmv11005")
    ts_re = re.compile(r"TS(\d{10})", re.IGNORECASE)

    # Parse requested ts_key into a datetime
    try:
        target_dt = datetime.strptime(ts_key, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except Exception as e:
        log.info(f"[swe-discovery] bad ts_key={ts_key}: {e!r}")
        return []

    candidates: list[tuple[float, str, str]] = []
    # (abs_delta_hours, ts_found_str, url)

    for n in names:
        base = n.strip().split("?", 1)[0]
        low = base.lower()
        if not any(code in low for code in swe_codes):
            continue
        if not any(low.endswith(ext) for ext in allowed_exts):
            continue

        m = ts_re.search(base)
        if not m:
            continue

        ts_found = m.group(1)
        try:
            dt_found = datetime.strptime(ts_found, "%Y%m%d%H").replace(
                tzinfo=timezone.utc
            )
        except Exception:
            continue

        delta_hours = abs((dt_found - target_dt).total_seconds()) / 3600.0
        url = base if base.startswith("http") else f"{COLLAB_BASE}/{base}"
        candidates.append((delta_hours, ts_found, url))

    if not candidates:
        log.info(f"[swe-discovery] ts={ts_key} -> 0 SWE candidates (raw)")
        return []

    # Prefer nearest in time; group by the best TS string we found
    candidates.sort(key=lambda t: t[0])
    best_delta = candidates[0][0]
    best_ts = candidates[0][1]

    # Safety: if nothing is within 48h, treat as "no SWE"
    MAX_DELTA_HOURS = 48.0
    if best_delta > MAX_DELTA_HOURS:
        log.info(
            f"[swe-discovery] ts={ts_key} -> nearest SWE TS={best_ts} "
            f"but delta={best_delta:.1f}h (>{MAX_DELTA_HOURS:.0f}h); ignoring"
        )
        return []

    best_urls = [url for d, ts, url in candidates if ts == best_ts]

    log.info(
        f"[swe-discovery] ts={ts_key} -> {len(best_urls)} SWE candidates "
        f"using TS={best_ts} (Δ={best_delta:.1f}h)"
    )
    return sorted(set(best_urls))


def _download_collab_to_grid_or_skip(
    url: str,
    workdir: Path,
    want=("melt", "snowmelt", "melt rate", "snow water equivalent", "swe"),
) -> tuple[dict, str]:
    """
    Download a single collaborator object and either:

    - Return a grid descriptor + description text
    - Or raise HTTPException with:
        204 -> "skip" (not the variable we want)
        404 -> object not found
        504 -> upstream/network unreachable
        502 -> payload not usable (bad format)

    In MELT mode we now restrict to true melt product(s) and ensure the
    description looks like melt/snowmelt (not temperature).
    """

    log.info(f"[collab] fetch {url}")
    fname = Path(url).name.lower()

    # Rough mode detection based on what the caller asked for.
    swe_mode = any("snow water equivalent" in w or w == "swe" for w in want)
    melt_mode = any("melt" in w for w in want)

    # --- quick filename-based screening BEFORE we even download ---
    # Product codes (heuristic, based on NOHRSC naming):
    #   ssmv11034, ssmv11005 -> SWE / SWE-like
    #   ssmv01020            -> surface temperature
    #   ssmv11044            -> modeled snowmelt liquid water equivalent
    #   ssmv11038            -> snow temperature (NOT melt)
    if swe_mode:
        # In SWE mode, only keep 11034 / 11005; skip known non-SWE.
        if not any(code in fname for code in ("ssmv11034", "ssmv11005")):
            raise HTTPException(status_code=204, detail="skip")
    elif melt_mode:
        # In MELT mode, *only* accept the true melt LE code(s).
        # Explicitly skip temperature code 11038 or others.
        if "ssmv11044" not in fname:
            raise HTTPException(status_code=204, detail="skip non-melt product")
        if "ssmv11038" in fname:
            raise HTTPException(status_code=204, detail="skip temperature product")
    else:
        # Fallback: allow anything, but we still may skip later based on header.
        pass

    # ---- actually download ----
    try:
        r = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "snodas-forecast/1.0"},
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=504,
            detail=f"upstream unreachable: {url} ({e})",
        )

    if r.status_code not in (200, 206):
        raise HTTPException(status_code=404, detail=f"{url} -> HTTP {r.status_code}")

    blob = r.content

    # Try to get a header text (from GRZ/TAR) and use it as a secondary filter.
    hdr_text = _hdr_from_grz_bytes(blob)
    desc_text = _desc_from_hdr_text(hdr_text).lower()

    # Secondary guard: if caller passed specific "want" words, check description.
    if desc_text:
        if swe_mode:
            # In SWE mode, if description clearly mentions temperature or melt
            # and not SWE, discard it (belt + suspenders).
            if any(t in desc_text for t in ("temperature", "temp", "melt")) and not any(
                t in desc_text for t in ("snow water equivalent", "swe")
            ):
                raise HTTPException(status_code=204, detail="skip")
        elif melt_mode:
            # In MELT mode we want *actual melt*, not just temperature:
            #  - Require "melt" or "snowmelt" in the description
            #  - Explicitly skip temperature-only descriptions
            if "temperature" in desc_text and not any(
                t in desc_text for t in ("melt", "snowmelt")
            ):
                raise HTTPException(
                    status_code=204, detail="skip temperature field in melt mode"
                )
            if not any(t in desc_text for t in ("melt", "snowmelt")):
                raise HTTPException(
                    status_code=204,
                    detail="skip non-melt field in melt mode (no melt keyword)",
                )
        else:
            # Generic case: if nothing from 'want' appears in description, skip.
            if not any(w in desc_text for w in want):
                raise HTTPException(status_code=204, detail="skip")

    # If gzipped, decompress
    if len(blob) >= 2 and blob[0] == 0x1F and blob[1] == 0x8B:
        blob = gzip.decompress(blob)

    # Raw GRIB
    if blob.startswith(b"GRIB"):
        grb = workdir / "data.grib2"
        grb.write_bytes(blob)
        return {"type": "grib2", "path": grb}, desc_text or fname

    # Try to interpret as TAR (BIL+HDR or GRIB)
    looks_tar = (len(blob) % 512 == 0) or (b"ustar" in blob[:4096])
    if looks_tar:
        bio = io.BytesIO(blob)
        with tarfile.open(fileobj=bio, mode="r:*") as tf:
            grib_member = hdr_member = dat_member = None
            for m in tf.getmembers():
                n = m.name.lower()
                if n.endswith(".grib2") and grib_member is None:
                    grib_member = m
                elif n.endswith(".hdr") and hdr_member is None:
                    hdr_member = m
                elif (n.endswith(".dat") or n.endswith(".bil")) and dat_member is None:
                    dat_member = m

            # GRIB inside TAR
            if grib_member is not None:
                grb = workdir / "data.grib2"
                with tf.extractfile(grib_member) as f:
                    grb.write_bytes(f.read())
                return {"type": "grib2", "path": grb}, desc_text or fname

            # BIL+HDR inside TAR
            if hdr_member is not None and dat_member is not None:
                bil = workdir / "data.bil"
                hdr = workdir / "data.hdr"
                with tf.extractfile(dat_member) as f:
                    bil.write_bytes(f.read())
                with tf.extractfile(hdr_member) as f:
                    hdr.write_bytes(f.read())
                log.info(f"[collab] using BIL/HDR-from-TAR for {url}")
                return {"type": "bil", "hdr": hdr, "dat": bil}, desc_text or fname

    # If we get here, payload isn't in a format we know how to use
    bad = CACHE / f"bad_{Path(url).name}.payload"
    bad.write_bytes(blob)
    raise HTTPException(
        status_code=502,
        detail=f"not GRIB/TAR saved {bad.name}",
    )


def _iter_discovery(ts_key: str) -> Iterable[str]:
    """
    Combined discovery: directory listing (exact names) then wide patterns.
    """
    seen = set()
    for u in _discover_candidates_via_listing(ts_key):
        if u not in seen:
            seen.add(u)
            yield u
    for u in _candidate_urls_wide(ts_key):
        if u not in seen:
            seen.add(u)
            yield u


def _has_usable_for_ts(ts_key: str, max_checks: int = 60) -> bool:
    """
    A TS is 'usable' if at least one collaborator object for that TS
    contains melt/SWE grid that we can open successfully.
    Uses the same discovery strategy as the real COG builder.
    """
    tried = 0
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for url in _iter_discovery(ts_key):
            try:
                _download_collab_to_grid_or_skip(
                    url,
                    tdir,
                    want=(
                        "melt",
                        "snowmelt",
                        "melt rate",
                        "snow water equivalent",
                        "swe",
                    ),
                )
                return True
            except HTTPException as e:
                if e.status_code == 204:
                    # hdr says not melt/SWE -> just skip
                    pass
            except Exception:
                pass
            tried += 1
            if tried >= max_checks:
                break
    return False


@lru_cache(maxsize=4)
def _find_latest_ts_for_horizon(days_ahead: int) -> str | None:
    for ts in _candidate_ts_keys_for_horizon(days_ahead):
        if _has_usable_for_ts(ts):
            return ts
    return None


def _build_forecast_cog_ts(ts_key: str) -> Path:
    """
    Build or retrieve a COG of 24h modeled snowmelt liquid equivalent for
    the given TS key, using the NOHRSC "modeled snowmelt" product
    (ssmv11044).

    This is *not* SWE(T-24h) - SWE(T); it is the direct melt liquid
    equivalent field from NOHRSC.
    """
    ts_key = _normalize_ts_key(ts_key)
    out = (CACHE / "forecast") / f"fcst_melt24_{ts_key}_3857_cog.tif"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        return out

    log.info(f"[build_forecast] starting MELT COG for ts={ts_key}")

    melt_cache_root = CACHE / "forecast_melt"
    melt_cache_root.mkdir(parents=True, exist_ok=True)

    # One directory per TS so we can keep payloads on disk
    ts_dir = melt_cache_root / ts_key
    ts_dir.mkdir(parents=True, exist_ok=True)

    melt_grid: dict | None = None

    # Discover melt products for this TS
    melt_urls = _candidate_melt_urls_for_ts(ts_key)

    for url in melt_urls:
        try:
            grid_desc, desc = _download_collab_to_grid_or_skip(
                url,
                ts_dir,
                want=(
                    "melt",
                    "snowmelt",
                    "melt rate",
                    "liquid water equivalent",
                    "melted water equivalent",
                ),
            )
            melt_grid = grid_desc
            log.info(
                f"[build_forecast] picked MELT ts={ts_key} "
                f"url={url} desc={desc}"
            )
            break
        except HTTPException as e:
            # 204 -> skip (not actually a melt field), 404/504 -> upstream issue
            if e.status_code in (204, 404, 504):
                log.info(
                    f"[build_forecast] MELT skip {url}: "
                    f"{e.status_code} {e.detail}"
                )
                continue
            log.info(f"[build_forecast] MELT HTTPException {url}: {e.detail}")
            continue
        except Exception as e:
            log.info(f"[build_forecast] MELT unexpected {url}: {e!r}")
            continue

    if melt_grid is None:
        raise HTTPException(
            status_code=502,
            detail=f"No MELT field for {ts_key}",
        )

    # Open the melt field and clip to >= 0
    da_melt = _open_any_grid_as_da(melt_grid)
    melt = da_melt.clip(min=0).astype("float32")

    return _da_to_cog_3857(melt, out)


# ------------ latest helpers (dir listing + tile probe) ------------
def _find_last_available_ts_key(ttl_sec: int = 600) -> str | None:
    url = COLLAB_BASE + "/"
    try:
        r = requests.get(
            url, timeout=15, headers={"User-Agent": "snodas-forecast/1.0"}
        )
    except Exception as e:
        log.info(f"[dirlist] exception {e!r}")
        return None
    if r.status_code != 200:
        log.info(f"[dirlist] HTTP {r.status_code}")
        return None
    txt = r.text or ""
    if not txt.strip():
        log.info("[dirlist] empty body")
        return None
    try:
        href_pat = re.compile(
            r'href\s*=\s*(?P<q>[\'"])(?P<u>.*?)(?P=q)', re.IGNORECASE
        )
        names = [m.group("u") for m in href_pat.finditer(txt)]
        if not names:
            bare = re.compile(r"href\s*=\s*([^ >]+)", re.IGNORECASE)
            names = [m.group(1) for m in bare.finditer(txt)]
    except Exception as e:
        log.info(f"[dirlist] href parse error: {e!r}")
        return None
    if not names:
        log.info("[dirlist] no hrefs")
        return None
    allowed_exts = (".grz", ".tar", ".tar.gz", ".tgz")
    ts_keys: list[str] = []
    ts_re = re.compile(r"TS(\d{10})")
    for n in names:
        base = n.strip().split("?")[0]
        if not any(base.lower().endswith(ext) for ext in allowed_exts):
            continue
        m = ts_re.search(base)
        if m:
            ts_keys.append(m.group(1))
    if not ts_keys:
        log.info("[dirlist] no TS tokens")
        return None
    ts_keys.sort()
    latest = ts_keys[-1]
    log.info(f"[dirlist] latest TS: {latest} ({len(ts_keys)} candidates)")
    return latest


_SAMPLE_TILES = [(5, 8, 10), (5, 5, 11), (5, 10, 8)]


def _has_any_tile(ts_key: str, samples=_SAMPLE_TILES) -> bool:
    try:
        cog = _build_forecast_cog_ts(ts_key)
    except Exception as e:
        log.info(f"[check] COG build failed {ts_key}: {e}")
        return False
    try:
        with COGReader(cog.as_posix()) as cr:
            for z, x, y in samples:
                try:
                    data, mask = cr.tile(
                        x, y, z, tilesize=256, resampling_method="bilinear"
                    )
                    band = data[0].astype("float32")
                    band = np.where(np.isfinite(band) & (band > 0), band, 0.0)
                    if (band > 0).any():
                        return True
                except Exception as te:
                    log.info(
                        f"[check] tile probe failed ts={ts_key} "
                        f"zxy=({z},{x},{y}): {te}"
                    )
                    continue
    except Exception as e:
        log.info(f"[check] open/tile loop failed {ts_key}: {e}")
        return False
    return False


def _pick_ts_key_with_fallback(
    initial_ts_key: str, days_back: int = 10
) -> str | None:
    if _has_any_tile(initial_ts_key):
        return initial_ts_key
    try:
        base_dt = datetime.strptime(initial_ts_key, "%Y%m%d%H").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        la = _find_last_available_ts_key()
        if la and _has_any_tile(la):
            log.info(f"[fallback] using dir latest {la}")
            return la
        return None
    for d in range(1, days_back + 1):
        day = (base_dt - timedelta(days=d)).date().strftime("%Y%m%d")
        for hh in _VALID_HOURS:
            cand = f"{day}{hh}"
            log.info(f"[fallback] probing {cand} (d-{d})")
            if _has_any_tile(cand):
                log.info(f"[fallback] selected {cand}")
                return cand
    la = _find_last_available_ts_key()
    if la and _has_any_tile(la):
        log.info(f"[fallback] selected dir latest {la}")
        return la
    return None


def _pick_ts_with_fallback(preferred_days: int) -> str | None:
    tried: set[int] = set()
    for d in (preferred_days, preferred_days - 1, preferred_days + 1):
        if d >= 1 and d not in tried:
            tried.add(d)
            ts = _find_latest_ts_for_horizon(d)
            if ts:
                return ts
    # last-ditch
    return _find_last_available_ts_key()


# ------------ rendering helpers ------------
def _da_to_png(data, mask, max_val: float | None):
    band = data[0].astype("float32")
    band = np.where(np.isfinite(band) & (band > 0), band, 0.0)
    if max_val is None or max_val <= 0:
        # auto-scale by 99th percentile, with a floor
        p99 = float(np.nanpercentile(band, 99)) if np.isfinite(band).any() else 0.05
        max_val = p99 or 0.05
    band = np.clip(band / float(max_val), 0, 1) * 255
    band = band.astype("uint8")[np.newaxis, :, :]
    return render(band, mask=mask, img_format="PNG")


def _resolve_latest_date(days_back: int = 60) -> Optional[str]:
    utc_today = dt.date.today()
    for i in range(days_back + 1):
        d = utc_today - dt.timedelta(days=i)
        ok, _ = nsidc_exists(d)
        if ok:
            return yyyymmdd(d)
    return None


def _hdr_to_cog_3857(hdr_path: Path, out_path: Path) -> Path:
    dat_path = hdr_path.with_suffix(".dat")
    if not dat_path.exists():
        candidates = list(hdr_path.parent.glob("*.dat"))
        if candidates:
            dat_path = candidates[0]
        else:
            raise HTTPException(status_code=500, detail="Matching .dat not found")
    da = _open_bil_as_da(hdr_path, dat_path)
    return _da_to_cog_3857(da, out_path)


def build_24h_cog(date: dt.date) -> Path:
    dstr = yyyymmdd(date)
    out = CACHE / f"melt24h_{dstr}_3857_cog.tif"
    if out.exists() and out.stat().st_size > 0:
        return out
    tarp = _download_day_tar(date)
    hdr, _ = _extract_var1044(tarp)
    return _hdr_to_cog_3857(hdr, out)


def build_72h_cog(end_date: dt.date) -> Path:
    dstr = yyyymmdd(end_date)
    out = CACHE / f"melt72h_end_{dstr}_3857_cog.tif"
    if out.exists() and out.stat().st_size > 0:
        return out
    cogs = [
        build_24h_cog(end_date - dt.timedelta(days=2)),
        build_24h_cog(end_date - dt.timedelta(days=1)),
        build_24h_cog(end_date),
    ]
    with rasterio.open(cogs[0]) as base:
        acc = base.read(1).astype("float32")
        for c in cogs[1:]:
            with rasterio.open(c) as src:
                arr = src.read(1)
                if (
                    src.transform != base.transform
                    or src.width != base.width
                    or src.height != base.height
                    or src.crs != base.crs
                ):
                    tmp = np.full_like(acc, np.nan, dtype="float32")
                    reproject(
                        arr,
                        tmp,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=base.transform,
                        dst_crs=base.crs,
                        resampling=Resampling.bilinear,
                    )
                    arr = tmp
                mask = np.isfinite(arr)
                acc[mask] = np.nan_to_num(acc[mask], nan=0.0) + arr[mask]
        tmp = out.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **base.profile) as ds:
            ds.write(acc, 1)
    cog_translate(
        tmp.as_posix(),
        out.as_posix(),
        img_profiles.get("deflate"),
        in_memory=False,
        quiet=True,
    )
    Path(tmp).unlink(missing_ok=True)
    return out


def _tile_from_cog_png(cog: Path, z: int, x: int, y: int, max_m: float | None) -> bytes:
    if not cog.exists():
        raise HTTPException(status_code=500, detail=f"COG missing: {cog.name}")
    with COGReader(cog.as_posix()) as reader:
        data, mask = reader.tile(x, y, z, tilesize=256, resampling_method="bilinear")
    band = data[0].astype("float32")
    band = np.where(np.isfinite(band) & (band > 0), band, 0.0)
    if max_m:
        band = np.clip(band / float(max_m), 0.0, 1.0) * 255.0
    else:
        band = np.clip(band / 0.10, 0.0, 1.0) * 255.0
    band_u8 = band.astype("uint8")[np.newaxis, :, :]
    return render(band_u8, mask=mask, img_format="PNG")


def _serve_tile_for_ts(ts_key: str, z: int, x: int, y: int, max_val: float | None) -> Response:
    ts_key = _normalize_ts_key(ts_key)
    chosen = _pick_ts_key_with_fallback(ts_key, days_back=10) or ts_key
    log.info(f"[tiles] serving forecast tile ts={chosen} zxy=({z},{x},{y})")
    cog = _build_forecast_cog_ts(chosen)
    with COGReader(cog.as_posix()) as cr:
        data, mask = cr.tile(x, y, z, tilesize=256, resampling_method="bilinear")
    png = _da_to_png(data, mask, max_val)
    resp = Response(png, media_type="image/png")
    resp.headers["X-Forecast-TS"] = chosen
    resp.headers["X-Route"] = "forecast-ts"
    return resp


# ------------ endpoints ------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "SNODAS Snowmelt Tiles. See /web for the demo and /docs for API."


@app.get("/healthz", response_class=PlainTextResponse)
def health():
    return "ok"


@app.get("/config.js", response_class=PlainTextResponse)
def config_js():
    token = os.environ.get("MAPBOX_TOKEN", "")
    return f"window.__SNODAS_CONFIG__ = {{ MAPBOX_TOKEN: '{token}' }};"


@app.get("/meta/latest")
def meta_latest():
    d = _resolve_latest_date()
    if d is None:
        return JSONResponse({"available": False, "date": None, "url": None})
    date_obj = dt.datetime.strptime(d, "%Y%m%d").date()
    return JSONResponse(
        {"available": True, "date": d, "url": nsidc_daily_tar_url(date_obj)}
    )


@app.get("/tiles/24h/latest/{z}/{x}/{y}.png")
def tiles_24h_latest(z: int, x: int, y: int, max: float | None = Query(None)):
    d = _resolve_latest_date()
    if d is None:
        raise HTTPException(status_code=503, detail="No recent NSIDC day available")
    date_obj = dt.datetime.strptime(d, "%Y%m%d").date()
    cog = build_24h_cog(date_obj)
    img = _tile_from_cog_png(cog, z, x, y, max_m=max)
    return Response(img, media_type="image/png")


@app.get("/tiles/24h/{date_yyyymmdd:regex(\\d{8})}/{z}/{x}/{y}.png")
def tiles_24h(
    date_yyyymmdd: str,
    z: int,
    x: int,
    y: int,
    max: float | None = Query(None),
):
    try:
        d = dt.datetime.strptime(date_yyyymmdd, "%Y%m%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be YYYYMMDD")
    cog = build_24h_cog(d)
    img = _tile_from_cog_png(cog, z, x, y, max_m=max)
    return Response(img, media_type="image/png")


@app.get("/tiles/72h/{end_yyyymmdd:regex(\\d{8})}/{z}/{x}/{y}.png")
def tiles_72h(
    end_yyyymmdd: str,
    z: int,
    x: int,
    y: int,
    max: float | None = Query(None),
):
    try:
        d = dt.datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be YYYYMMDD")
    cog = build_72h_cog(d)
    img = _tile_from_cog_png(cog, z, x, y, max_m=max)
    return Response(img, media_type="image/png")


@app.get("/tiles/forecast/latest/{z}/{x}/{y}.png")
def tiles_forecast_latest(
    z: int,
    x: int,
    y: int,
    max: float | None = Query(None),
):
    """
    Serve forecast tiles using the best-available TS key around D+2,
    with fallbacks handled inside _pick_ts_with_fallback.
    """
    ts_key = _pick_ts_with_fallback(2)  # prefer D+2, fallback to D+1/D+3/dir-latest
    if not ts_key:
        raise HTTPException(status_code=503, detail="No usable forecast TS key")

    log.info(f"[tiles] /tiles/forecast/latest using ts={ts_key}")
    try:
        return _serve_tile_for_ts(ts_key, z, x, y, max_val=max)
    except HTTPException as e:
        if e.status_code in (502, 503, 504):
            log.error("tile render failed (forecast/upstream): %s", e.detail)
            raise
        log.error("tile render HTTPException: %s", e.detail)
        raise HTTPException(status_code=502, detail="Forecast tile render failed")
    except Exception as e:
        log.error("tile render failed: %r", e)
        raise HTTPException(status_code=502, detail="Forecast tile render failed")


@app.get("/tiles/forecast/{ts_key:regex(\\d{8}(?:[_-]?\\d{2}))}/{z}/{x}/{y}.png")
def tiles_forecast(
    ts_key: str,
    z: int,
    x: int,
    y: int,
    max: float | None = Query(None),
):
    """
    Direct forecast tile access for a specific TS key.
    """
    log.info(f"[tiles] /tiles/forecast using ts={ts_key}")
    return _serve_tile_for_ts(ts_key, z, x, y, max_val=max)


@app.get("/forecast/latest_ts")
def forecast_latest_ts():
    """
    Horizon discovery status endpoint, used by frontend to show which TS
    we consider "latest" for D+2/D+3 and what the directory listing latest is.
    """
    d2 = _find_latest_ts_for_horizon(2)
    d3 = _find_latest_ts_for_horizon(3)
    latest_any = _find_last_available_ts_key()

    if not d2 and not d3:
        return {
            "d2": None,
            "d3": None,
            "last_available_ts_key": latest_any,
            "reason": "no_horizon_match_via_probe",
        }
    return {
        "d2": d2,
        "d3": d3,
        "last_available_ts_key": d3 or d2 or latest_any,
    }


# ------------ debug ------------
@app.get("/__probe_collab")
def __probe_collab():
    try:
        r = requests.get(
            COLLAB_BASE + "/",
            timeout=15,
            headers={"User-Agent": "snodas-forecast/1.0"},
        )
        return {
            "ok": r.ok,
            "status": r.status_code,
            "length": len(r.text or ""),
        }
    except Exception as e:
        return {"ok": False, "error": repr(e)}


@app.get("/__diag")
def __diag():
    info = {
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "cache_dir": str(CACHE),
        "cache_exists": CACHE.exists(),
        "cache_writable": os.access(CACHE, os.W_OK),
        "env_MAPBOX_TOKEN": bool(os.environ.get("MAPBOX_TOKEN")),
    }
    try:
        import rasterio as _r

        info["rasterio"] = _r.__version__
    except Exception as e:
        info["rasterio_err"] = repr(e)
    try:
        import rioxarray as _rx

        info["rioxarray"] = _rx.__version__
    except Exception as e:
        info["rioxarray_err"] = repr(e)
    try:
        import rio_cogeo as _c

        info["rio_cogeo"] = getattr(_c, "__version__", "unknown")
    except Exception as e:
        info["rio_cogeo_err"] = repr(e)
    try:
        import cfgrib as _cg

        info["cfgrib"] = _cg.__version__
    except Exception as e:
        info["cfgrib_err"] = repr(e)
    return info


@app.get("/__logs")
def __logs(since_ms: int | None = None, limit: int = 200):
    try:
        limit = max(1, min(int(limit), 1000))
    except Exception:
        limit = 200
    rows = list(_LOG_DEQUE)
    if since_ms is not None:
        try:
            s = int(since_ms)
            rows = [r for r in rows if r["ts"] > s]
        except Exception:
            pass
    rows = rows[-limit:]
    return {"count": len(rows), "items": rows}
@app.get("/__debug_melt/{ts_key}")
def __debug_melt(ts_key: str):
    """
    Debug endpoint: show all MELT discovery candidates for a TS key,
    including:
      - URLs found
      - Whether they are accepted or skipped
      - Parsed header descriptions (if available)
      - Skip reason (if applicable)

    Example:
      /__debug_melt/2025120406
    """
    ts_key_norm = _normalize_ts_key(ts_key)
    out = {
        "ts": ts_key_norm,
        "melt_urls_raw": [],
        "accepted": [],
        "skipped": [],
    }

    # 1. Get melt URLs from discovery function
    melt_urls = _candidate_melt_urls_for_ts(ts_key_norm)
    out["melt_urls_raw"] = melt_urls

    # 2. Now try each URL to see if it is accepted/rejected by downloader
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)

        for url in melt_urls:
            entry = {"url": url}

            try:
                grid_desc, desc = _download_collab_to_grid_or_skip(
                    url,
                    work,
                    want=(
                        "melt",
                        "snowmelt",
                        "melt rate",
                        "liquid water equivalent",
                        "melted water equivalent",
                    ),
                )
                entry["accepted"] = True
                entry["type"] = grid_desc.get("type")
                entry["desc_text"] = desc
                entry["grid_desc"] = {k: str(v) for k, v in grid_desc.items()}
                out["accepted"].append(entry)

            except HTTPException as e:
                entry["accepted"] = False
                entry["reason"] = f"HTTP {e.status_code}: {e.detail}"
                out["skipped"].append(entry)

            except Exception as e:
                entry["accepted"] = False
                entry["reason"] = f"Unexpected error: {repr(e)}"
                out["skipped"].append(entry)

    return out
