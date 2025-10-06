import os
import io
import tarfile
import calendar
import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rio_tiler.io import COGReader
from rio_tiler.profiles import img_profiles
from starlette.responses import JSONResponse, PlainTextResponse

CACHE = Path(os.environ.get("SNODAS_CACHE", "./cache")).resolve()
CACHE.mkdir(parents=True, exist_ok=True)

APP_TZ = os.environ.get("APP_TZ", "America/Chicago")

NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02158/unmasked"

app = FastAPI(title="SNODAS Snowmelt Tiles (NOHRSC/NSIDC G02158)",
              version="0.1.0",
              description="24h and 72h snowmelt tiles for Mapbox/Leaflet")
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

def yyyymmdd(date: dt.date) -> str:
    return date.strftime("%Y%m%d")

def nsidc_month_folder(date: dt.date) -> str:
    m = date.month
    return f"{m:02d}_{calendar.month_abbr[m].title()}"

def nsidc_daily_tar_url(date: dt.date) -> str:
    dstr = yyyymmdd(date)
    return f"{NSIDC_BASE}/{date.year}/{nsidc_month_folder(date)}/{dstr}/SNODAS_unmasked_{dstr}.tar"

def download_day(date: dt.date) -> Path:
    dstr = yyyymmdd(date)
    tar_path = CACHE / f"SNODAS_unmasked_{dstr}.tar"
    if tar_path.exists() and tar_path.stat().st_size > 0:
        return tar_path

    url = nsidc_daily_tar_url(date)
    r = requests.get(url, timeout=180)
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail=f"NSIDC file not found for {dstr}")
    tar_path.write_bytes(r.content)
    return tar_path

def extract_variable_1044(tar_path: Path) -> Tuple[Path, Path]:
    tempdir = CACHE / (tar_path.stem + "_extracted")
    tempdir.mkdir(exist_ok=True)
    hdr_path = None
    dat_path = None
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if "1044" in m.name and (m.name.endswith(".Hdr") or m.name.endswith(".dat"))]
        if not members:
            raise HTTPException(status_code=500, detail="Variable 1044 not found in TAR.")
        for m in members:
            outp = tempdir / Path(m.name).name
            if not outp.exists():
                tf.extract(m, path=tempdir)
            if outp.suffix == ".Hdr":
                hdr_path = outp
            elif outp.suffix == ".dat":
                dat_path = outp
    if not hdr_path or not dat_path:
        raise HTTPException(status_code=500, detail="Missing .Hdr or .dat for variable 1044.")
    return hdr_path, dat_path

def hdr_to_cog(hdr_path: Path, cog_path: Path) -> Path:
    if cog_path.exists():
        return cog_path

    with rasterio.open(hdr_path) as src:
        data = src.read(1).astype("float32")
        data[data <= -9999] = np.nan
        data = data / 100000.0

        dst_crs = "EPSG:3857"
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        dst = np.full((height, width), np.nan, dtype="float32")

        reproject(
            data, dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        temp_tif = cog_path.with_suffix(".tmp.tif")
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=np.nan,
        )
        with rasterio.open(temp_tif, "w", **profile) as ds:
            ds.write(dst, 1)

    cog_translate(
        temp_tif.as_posix(),
        cog_path.as_posix(),
        cog_profiles.get("deflate"),
        in_memory=False,
        quiet=True,
    )
    temp_tif.unlink(missing_ok=True)
    return cog_path

def build_24h_cog(date: dt.date) -> Path:
    dstr = yyyymmdd(date)
    cog_path = CACHE / f"melt24h_{dstr}_3857_cog.tif"
    if cog_path.exists() and cog_path.stat().st_size > 0:
        return cog_path

    tar_path = download_day(date)
    hdr, _ = extract_variable_1044(tar_path)
    cog = hdr_to_cog(hdr, cog_path)

    # cleanup to save space
    import shutil
    try:
        tar_path.unlink(missing_ok=True)
        shutil.rmtree(CACHE / (tar_path.stem + "_extracted"), ignore_errors=True)
    except Exception:
        pass

    return cog

def build_72h_cog(ending_date: dt.date) -> Path:
    dstr = yyyymmdd(ending_date)
    out_cog = CACHE / f"melt72h_end_{dstr}_3857_cog.tif"
    if out_cog.exists() and out_cog.stat().st_size > 0:
        return out_cog

    cogs = [build_24h_cog(ending_date - dt.timedelta(days=2)),
            build_24h_cog(ending_date - dt.timedelta(days=1)),
            build_24h_cog(ending_date)]

    with rasterio.open(cogs[0]) as base:
        sum_arr = base.read(1).astype("float32")
        for cog in cogs[1:]:
            with rasterio.open(cog) as src:
                arr = src.read(1)
                if (src.transform != base.transform) or (src.width != base.width) or (src.height != base.height) or (src.crs != base.crs):
                    tmp = np.full_like(sum_arr, np.nan)
                    reproject(
                        arr, tmp,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=base.transform, dst_crs=base.crs,
                        resampling=Resampling.bilinear,
                    )
                    arr = tmp
                mask = ~np.isnan(arr)
                sum_arr[mask] = np.nan_to_num(sum_arr[mask], nan=0.0) + arr[mask]

        temp_tif = out_cog.with_suffix(".tmp.tif")
        profile = base.profile.copy()
        with rasterio.open(temp_tif, "w", **profile) as ds:
            ds.write(sum_arr, 1)

    cog_translate(
        temp_tif.as_posix(),
        out_cog.as_posix(),
        cog_profiles.get("deflate"),
        in_memory=False,
        quiet=True,
    )
    Path(temp_tif).unlink(missing_ok=True)
    return out_cog

def tile_from_cog(cog_path: Path, z: int, x: int, y: int, img_format: str = "png") -> bytes:
    if not cog_path.exists():
        raise HTTPException(status_code=500, detail=f"COG missing: {cog_path.name}")
    with COGReader(cog_path.as_posix()) as cog:
        tile, _ = cog.tile(x, y, z, tilesize=256, resampling_method="bilinear")
    profile = img_profiles.get(img_format)
    buffer = io.BytesIO()
    tile.save(buffer, **profile)
    return buffer.getvalue()

def parse_date(ds: str) -> dt.date:
    try:
        return dt.datetime.strptime(ds, "%Y%m%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be YYYYMMDD")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "SNODAS Snowmelt Tiles. See /web for the demo and /docs for API."

@app.get("/healthz", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/meta/availability/{date_yyyymmdd}")
def check_availability(date_yyyymmdd: str):
    d = parse_date(date_yyyymmdd)
    url = nsidc_daily_tar_url(d)
    r = requests.head(url, timeout=30)
    return JSONResponse({"date": date_yyyymmdd, "available": r.status_code == 200, "url": url})

@app.get("/tiles/24h/{date_yyyymmdd}/{z}/{x}/{y}.png")
def tiles_24h(date_yyyymmdd: str, z: int, x: int, y: int):
    d = parse_date(date_yyyymmdd)
    cog = build_24h_cog(d)
    img = tile_from_cog(cog, z, x, y, img_format="png")
    return Response(img, media_type="image/png")

@app.get("/tiles/72h/{end_yyyymmdd}/{z}/{x}/{y}.png")
def tiles_72h(end_yyyymmdd: str, z: int, x: int, y: int):
    d = parse_date(end_yyyymmdd)
    cog = build_72h_cog(d)
    img = tile_from_cog(cog, z, x, y, img_format="png")
    return Response(img, media_type="image/png")

@app.get("/legend", response_class=PlainTextResponse)
def legend():
    return (
        "Snowmelt (m) per period. Suggested breaks:\n"
        "0, 0.005, 0.01, 0.02, 0.05, 0.1\n"
        "Style ramp client-side; tiles are single-band continuous."
    )
