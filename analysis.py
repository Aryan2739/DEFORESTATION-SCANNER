"""Geospatial and multi-spectral canopy loss analysis pipeline."""

import calendar
from collections import defaultdict
import datetime
import gc
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from odc.stac import load
import planetary_computer
import pystac_client
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from scipy import ndimage
import streamlit as st
import xarray as xr

logger = logging.getLogger(__name__)

PRESETS = {
    "Nagpur: Pench / Ramtek Corridor": (79.18, 21.42, 79.36, 21.60),
    "Chandrapur: Ballarpur / Tadoba Buffer": (79.32, 19.82, 79.50, 20.00),
    "Pune: Mulshi / Tamhini Ghats": (73.32, 18.42, 73.50, 18.60),
    "Select on Map / Custom AOI": None,
}

BAD_SCL = [0, 1, 3, 6, 8, 9, 10, 11]
HANSEN_VERSIONS = ["GFC-2024-v1.12", "GFC-2023-v1.11"]

MAX_AREA_BY_RES = {
    10: 150.0,
    20: 500.0,
    40: 900.0,
}
MAX_TIMELINE_AREA_10M = 75.0


def utm_crs(bbox):
    """Return the EPSG code of the UTM zone for the bounding box center."""
    lon, lat = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def calc_area_km2(bbox):
    """Approximate area of a (west, south, east, north) bounding box in km2."""
    west, south, east, north = bbox
    if east <= west or north <= south:
        return 0.0
    lat_rad = np.radians((south + north) / 2.0)
    w_km = (east - west) * 111.32 * np.cos(lat_rad)
    h_km = (north - south) * 110.57
    return float(max(0.0, w_km * h_km))


def parse_coords_or_url(url_or_text):
    """Extract (lat, lon) from a coordinate string or map URL."""
    text = url_or_text.strip()
    patterns = [
        r"@(-?\d+\.\d+),(-?\d+\.\d+)",
        r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)",
        r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
    return None


@st.cache_data(show_spinner=False)
def geocode_place(query):
    """Geocode a place name using OpenStreetMap Nominatim."""
    params = urllib.parse.urlencode({"q": query, "format": "json", "limit": 1})
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "CanopyObservatory/2.1"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data:
                return (
                    float(data[0]["lat"]),
                    float(data[0]["lon"]),
                    data[0]["display_name"],
                ), None
            return None, f"No matching location found for '{query}'."
    except urllib.error.HTTPError as exc:
        logger.warning("Geocoding HTTP error: %s", exc)
        return None, f"Geocoding service returned HTTP {exc.code}."
    except urllib.error.URLError:
        logger.exception("Place geocoding request failed")
        return None, "Location lookup failed. Please retry or enter coordinates."
    except Exception:
        logger.exception("Unexpected place geocoding failure")
        return None, "Location lookup failed. Please retry or enter coordinates."


def validate_run_params(bbox, y1, y2, m1, m2, res, timeline, today=None):
    """Validate spatial, temporal, and memory constraints before running STAC queries."""
    if today is None:
        today = datetime.date.today()

    errors = []
    km2 = calc_area_km2(bbox)
    max_km2 = MAX_AREA_BY_RES.get(res, 500.0)

    if km2 <= 0.1:
        errors.append("Selected bounding box is invalid or smaller than 0.1 km².")
    elif km2 > max_km2:
        errors.append(
            f"Selected area is {km2:.1f} km², which exceeds the {max_km2:.0f} km² memory "
            f"limit for {res} m resolution. Switch to 20 m / 40 m resolution or draw a smaller box."
        )

    if res == 10 and timeline and km2 > MAX_TIMELINE_AREA_10M:
        errors.append(
            f"Annual Timeline at 10 m resolution is limited to {MAX_TIMELINE_AREA_10M:.0f} km² "
            f"(current: {km2:.1f} km²) to prevent out-of-memory crashes. Use 20 m resolution."
        )

    if y1 >= y2:
        errors.append("Baseline year must be earlier than Target year.")

    for yr_label, yr in [("Baseline year", y1), ("Target year", y2)]:
        if yr > today.year:
            errors.append(f"{yr_label} ({yr}) is in the future.")
        elif yr == today.year:
            last_day_of_m2 = calendar.monthrange(yr, m2)[1]
            window_end = datetime.date(yr, m2, last_day_of_m2)
            if window_end > today:
                errors.append(
                    f"Acquisition window {calendar.month_abbr[m1]} to {calendar.month_abbr[m2]} {yr} "
                    f"is not complete yet (today is {today.isoformat()}). "
                    f"Select an earlier target year (e.g., {today.year - 1}) or earlier completed months."
                )

    return errors


@st.cache_data(show_spinner=False, max_entries=6)
def build_composite(bbox, year, m1, m2, max_cloud, res):
    """Query Planetary Computer Sentinel-2 L2A and build a cloud-masked median composite."""
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        last_day = calendar.monthrange(year, m2)[1]
        all_items = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-{m1:02d}-01/{year}-{m2:02d}-{last_day:02d}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        ).item_collection()
    except Exception:
        logger.exception("STAC catalog search failed for %s", year)
        return None, f"Satellite catalog search failed for {year}. Please retry later."

    if not all_items:
        return None, (
            f"No Sentinel-2 scenes with <{max_cloud}% cloud cover found for "
            f"{year} (months {m1:02d} to {m2:02d}). Try increasing Max Cloud (%) or widening the month range."
        )

    by_tile = defaultdict(list)
    for item in all_items:
        by_tile[item.properties.get("s2:mgrs_tile", "default")].append(item)

    selected = []
    for tile_items in by_tile.values():
        clearest = sorted(tile_items, key=lambda i: i.properties.get("eo:cloud_cover", 100))[:6]
        selected.extend(clearest)

    crs = utm_crs(bbox)
    spec_bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    try:
        ds = load(
            selected,
            bands=spec_bands + ["SCL"],
            bbox=bbox,
            resolution=res,
            crs=crs,
            groupby="solar_day",
        )
        raw = ds[spec_bands].astype("float32")
        valid_dn = (raw["B02"] > 0) & (raw["B04"] > 0) & (raw["B08"] > 0)
        offset = xr.where(ds.time >= np.datetime64("2022-01-25"), 1000.0, 0.0)
        spec = (
            ((raw - offset) / 10000.0)
            .where(valid_dn & (~ds["SCL"].isin(BAD_SCL)))
            .median("time", skipna=True)
        )

        blue, green, red, nir, swir1, swir2 = (spec[k].values for k in spec_bands)
        valid_mask = ~np.isnan(nir) & ~np.isnan(red)
        ndvi = np.nan_to_num((nir - red) / (nir + red + 1e-6), nan=0.0).astype("float32")
        nbr = np.nan_to_num((nir - swir2) / (nir + swir2 + 1e-6), nan=0.0).astype("float32")
        ndwi = np.nan_to_num((green - nir) / (green + nir + 1e-6), nan=0.0).astype("float32")
        rgb = np.nan_to_num(
            np.clip(np.stack([red, green, blue], axis=-1) / 0.3, 0.0, 1.0), nan=0.0
        ).astype("float32")
        transform = from_origin(
            float(ds.x.values[0]) - res / 2.0,
            float(ds.y.values[0]) + res / 2.0,
            res,
            res,
        )
        del ds, raw, spec
        gc.collect()
    except Exception:
        logger.exception("Raster composite load failed for %s", year)
        return None, f"Satellite processing failed for {year}. Retry later or choose a smaller area."

    return {
        "rgb": rgb,
        "ndvi": ndvi,
        "nbr": nbr,
        "ndwi": ndwi,
        "valid_mask": valid_mask,
        "scene_count": len(selected),
        "crs": crs,
        "transform": transform,
    }, None


def detect_loss(
    ndvi_b,
    nbr_b,
    ndwi_b,
    valid_b,
    ndvi_a,
    nbr_a,
    ndwi_a,
    valid_a,
    min_drop,
    forest_thresh=0.50,
):
    """Detect canopy loss purely from Sentinel-2 spectral indices (NDVI, NBR, NDWI)."""
    both_valid = valid_b & valid_a & (ndwi_b < 0.05) & (ndwi_a < 0.08)
    baseline_canopy = both_valid & (ndvi_b > forest_thresh) & (nbr_b > 0.18)
    loss = (
        baseline_canopy
        & ((ndvi_b - ndvi_a) > min_drop)
        & ((nbr_b - nbr_a) > (min_drop * 0.65))
        & (ndvi_a < 0.40)
        & (nbr_a < 0.18)
    )
    loss = ndimage.binary_opening(loss, structure=np.ones((3, 3), dtype=bool))
    loss = ndimage.binary_closing(loss, structure=np.ones((2, 2), dtype=bool))
    return loss, baseline_canopy


def read_hansen_layer(layer, ver, tiles, bbox, shape, transform, crs):
    """Read and reproject a Hansen Global Forest Change layer."""
    west, south, east, north = bbox
    pad = (west - 0.01, south - 0.01, east + 0.01, north + 0.01)
    base = "https://storage.googleapis.com/earthenginepartners-hansen"
    datasets = []
    try:
        for t in tiles:
            url = f"{base}/{ver}/Hansen_{ver}_{layer}_{t}.tif"
            datasets.append(rasterio.open(url))
        if len(datasets) == 1:
            win = from_bounds(*pad, datasets[0].transform)
            data = datasets[0].read(1, window=win)
            win_tf = datasets[0].window_transform(win)
        else:
            mosaic, win_tf = merge(datasets, bounds=pad)
            data = mosaic[0]
        for ds in datasets:
            ds.close()
        dest = np.zeros(shape, dtype="uint8")
        reproject(
            data,
            dest,
            src_transform=win_tf,
            src_crs="EPSG:4326",
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
        return dest, None
    except Exception:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        logger.exception("Hansen %s (%s) read failed", layer, ver)
        return None, "Reference data tile download failed."


@st.cache_data(show_spinner=False, max_entries=6)
def hansen_truth(bbox, crs, _transform, shape, y1, y2):
    """Load Hansen GFC lossyear layer and return both Strict and Buffered truth masks."""
    west, south, east, north = bbox
    tops = sorted(set(range(int(np.ceil(south / 10) * 10), int(np.ceil(north / 10) * 10) + 1, 10)))
    lefts = sorted(set(range(int(np.floor(west / 10) * 10), int(np.floor(east / 10) * 10) + 1, 10)))
    tiles = [
        f"{abs(t):02d}{'N' if t >= 0 else 'S'}_{abs(l):03d}{'E' if l >= 0 else 'W'}"
        for t in tops
        for l in lefts
    ]

    errors = []
    for ver in HANSEN_VERSIONS:
        loss_dest, err = read_hansen_layer("lossyear", ver, tiles, bbox, shape, _transform, crs)
        if loss_dest is not None:
            last_yr = int(ver.split("-")[1])
            if y1 >= last_yr:
                return None, (
                    f"Hansen {ver} reference data ends in {last_yr}, so it cannot validate "
                    f"changes between {y1} and {y2}. Set Baseline Year < {last_yr} and Target Year <= {last_yr} "
                    "to compute Hansen validation scores."
                )

            warn_msg = None
            if y2 > last_yr:
                warn_msg = (
                    f"Note: Hansen {ver} only covers through {last_yr}. Validation metrics below "
                    f"compare against Hansen loss through {last_yr} (post-{last_yr} losses are not in Hansen yet)."
                )

            y1_strict = max(1, (y1 - 2000) + 1)
            y2_strict = min(last_yr - 2000, y2 - 2000)
            truth_strict = (loss_dest >= y1_strict) & (loss_dest <= y2_strict)

            y1_buf = max(1, (y1 - 2000) - 1)
            y2_buf = min(last_yr - 2000, (y2 - 2000) + 1)
            truth_buffered = (loss_dest >= y1_buf) & (loss_dest <= y2_buf)

            return {
                "truth_strict": truth_strict,
                "truth_buffered": truth_buffered,
                "last_yr": last_yr,
                "version": ver,
            }, warn_msg
        if err:
            errors.append(err)

    return None, "Could not load the Hansen reference data. Validation was skipped; please retry later."


def _compute_metrics(tp_count, fp_count, fn_count):
    prec = tp_count / max(tp_count + fp_count, 1)
    rec = tp_count / max(tp_count + fn_count, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = tp_count / max(tp_count + fp_count + fn_count, 1)
    return {
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "f1": round(float(f1), 3),
        "iou": round(float(iou), 3),
    }


def accuracy(pred, truth_strict, rgb, truth_buffered=None):
    """Calculate both Strict and Buffered accuracy scores against Hansen GFC."""
    if truth_buffered is None:
        truth_buffered = truth_strict

    strict_tp = int((pred & truth_strict).sum())
    strict_fp = int((pred & (~truth_strict)).sum())
    strict_fn = int(((~pred) & truth_strict).sum())
    strict_scores = _compute_metrics(strict_tp, strict_fp, strict_fn)

    t_buf = ndimage.binary_dilation(truth_buffered, structure=np.ones((3, 3), dtype=bool))
    p_buf = ndimage.binary_dilation(pred, structure=np.ones((3, 3), dtype=bool))
    tp_p = pred & t_buf
    tp_t = truth_buffered & p_buf
    fp_buf = pred & (~t_buf)
    fn_buf = truth_buffered & (~p_buf)

    n_tp = int(max(tp_p.sum(), tp_t.sum()))
    n_fp = int(fp_buf.sum())
    n_fn = int(fn_buf.sum())
    buffered_scores = _compute_metrics(n_tp, n_fp, n_fn)

    img = (rgb * 0.45).copy()
    img[tp_p | (pred & truth_buffered)] = [0.15, 0.85, 0.35]
    img[fp_buf] = [0.90, 0.20, 0.15]
    img[fn_buf] = [0.20, 0.55, 0.95]

    combined_scores = {
        "strict": strict_scores,
        "buffered": buffered_scores,
        "precision": strict_scores["precision"],
        "recall": strict_scores["recall"],
        "f1": strict_scores["f1"],
        "iou": strict_scores["iou"],
    }
    return combined_scores, (np.clip(img, 0.0, 1.0) * 255).astype("uint8")


def generate_optional_ai_briefing(facts, acc):
    """Generate an optional AI briefing if ANTHROPIC_API_KEY is configured on the server."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        return None, None

    prompt = (
        "Write a concise 3-sentence technical remote-sensing briefing summarizing these "
        f"Sentinel-2 canopy loss metrics: {json.dumps(facts)}. "
        f"Validation metrics vs Hansen GFC: {json.dumps(acc) if acc else 'Not run'}. "
        "Remind the reader that spectral canopy loss can reflect fire, phenology, or harvesting."
    )
    req_body = json.dumps(
        {
            "model": "claude-3-5-haiku-latest",
            "max_tokens": 220,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=req_body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            content = payload.get("content", [])
            if content and "text" in content[0]:
                return content[0]["text"].strip(), None
    except Exception:
        logger.exception("Server-side AI summary generation failed")
        return None, "Optional AI briefing could not be generated; the core analysis completed."
    return None, None