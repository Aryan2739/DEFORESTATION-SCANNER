"""Export utilities, image compression, report formatting, and session-scoped persistence."""

import base64
import io
import json
import logging
from pathlib import Path
import pickle
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import shapes
from rasterio.warp import transform_geom

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")


def _session_cache_file(session_id=None):
    """Return a cache file path isolated to the visitor's session ID."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(session_id or ""))
    if not safe_id:
        return None
    return CACHE_DIR / f"result_{safe_id}.pkl"


def make_overlay(rgb, loss):
    """Blend detected canopy loss mask (red-orange) over true-color RGB composite."""
    out = rgb.copy()
    out[loss] = 0.30 * out[loss] + 0.70 * np.array([0.92, 0.15, 0.10], dtype="float32")
    return (np.clip(out, 0.0, 1.0) * 255).astype("uint8")


def to_png(arr):
    """Encode an uint8 RGB numpy array as PNG bytes for download."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_web_jpeg_b64(arr, max_dim=960, quality=82):
    """
    Downscale and compress an RGB array to JPEG base64 so the interactive
    before/after slider stays fast and lightweight in the browser.
    """
    im = Image.fromarray(arr).convert("RGB")
    w, h = im.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def to_geojson(loss_mask, transform, crs, ha_px):
    """Vectorize binary canopy loss mask to an EPSG:4326 GeoJSON FeatureCollection string."""
    feats = []
    for geom, val in shapes(loss_mask.astype("uint8"), mask=loss_mask, transform=transform):
        if val == 1:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": transform_geom(crs, "EPSG:4326", geom),
                    "properties": {
                        "class": "canopy_loss_detected",
                        "pixel_ha": round(float(ha_px), 4),
                    },
                }
            )
    return json.dumps({"type": "FeatureCollection", "features": feats}, indent=2)


def make_gif(frames, years, vals, y1):
    """Create an animated GIF showing cumulative canopy loss progression across years."""
    try:
        font = ImageFont.load_default(size=max(14, frames[0].shape[1] // 30))
    except TypeError:
        font = ImageFont.load_default()
    pics = []
    for arr, yr, ha in zip(frames, years, vals):
        im = Image.fromarray(arr).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 28], fill=(16, 35, 26))
        d.text(
            (10, 5),
            f"{yr}  |  {ha:,.1f} ha cumulative canopy loss detected since {y1}",
            fill=(248, 250, 252),
            font=font,
        )
        pics.append(im)
    buf = io.BytesIO()
    pics[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pics[1:],
        duration=900,
        loop=0,
    )
    return buf.getvalue()


def make_markdown(r, story):
    """Generate a Markdown report including telemetry, strict/buffered validation, and caveats."""
    f = r["facts"]
    lines = [
        f"# Spectral Canopy Loss Report: {f['region']}",
        "",
        f"Comparison ({r['years'][0]} vs {r['years'][1]}) via Sentinel-2 L2A multi-spectral composites.",
        "",
        "## Key Telemetry",
        "",
    ]
    lines += [f"- **{k}**: {v}" for k, v in f.items()]

    if r.get("acc"):
        a = r["acc"]
        s = a["strict"]
        b = a["buffered"]
        lines += [
            "",
            "## Hansen GFC Validation (Independent Baseline)",
            "",
            "| Metric | Strict (Exact Pixel & Year) | Buffered (+/-1 Pixel, +/-1 Year) |",
            "| :--- | :--- | :--- |",
            f"| **Precision** | {s['precision']} | {b['precision']} |",
            f"| **Recall** | {s['recall']} | {b['recall']} |",
            f"| **F1 Score** | {s['f1']} | {b['f1']} |",
            f"| **IoU** | {s['iou']} | {b['iou']} |",
        ]

    lines += [
        "",
        "## Interpretation & Limitations (What This Cannot Tell You)",
        "",
        "- **Canopy loss is not always permanent deforestation.** Satellite spectral drops (dNDVI / dNBR) detect loss of green canopy cover and moisture.",
        "- **Common look-alikes:** Wildfires, prescribed burns, rotational harvesting of crops or plantations, and seasonal leaf-fall in dry deciduous forests can trigger similar spectral drops.",
        "- Always verify major disturbance patches with high-resolution imagery or field inspection.",
        "",
        "## Summary",
        "",
        r["summary"],
    ]
    if r.get("ai_briefing"):
        lines += ["", "## Server AI Briefing", "", r["ai_briefing"]]

    lines += [
        "",
        "## Field Notes",
        "",
        story or "_No field notes recorded._",
    ]
    return "\n".join(lines)


def save_last_result(result_dict, session_id=None):
    """Persist the analysis result scoped to the current visitor's session_id."""
    cache_file = _session_cache_file(session_id)
    if cache_file is None:
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for old_file in CACHE_DIR.glob("result_*.pkl"):
            try:
                if (old_file.stat().st_mtime + 86400) < Path().stat().st_mtime:
                    old_file.unlink()
            except OSError:
                pass
        with open(cache_file, "wb") as fh:
            pickle.dump(result_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    except Exception as exc:
        logger.warning("Failed to persist result to %s: %s", cache_file, exc)
        return str(exc)


def load_last_result(session_id=None):
    """Load the visitor's own cached analysis result if available."""
    cache_file = _session_cache_file(session_id)
    if cache_file is None or not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        logger.warning("Could not load cached result from %s: %s", cache_file, exc)
        return None


def clear_last_result(session_id=None):
    """Remove the visitor's persisted analysis result from disk."""
    cache_file = _session_cache_file(session_id)
    if cache_file is None:
        return
    try:
        if cache_file.exists():
            cache_file.unlink()
    except OSError as exc:
        logger.warning("Could not remove cached result file: %s", exc)