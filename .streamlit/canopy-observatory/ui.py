"""Streamlit user interface for Deforestation Scanner."""

import base64
import datetime
import gc
import logging
from pathlib import Path

import folium
from folium.plugins import Draw
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from analysis import (
    MAX_AREA_BY_RES,
    PRESETS,
    accuracy,
    build_composite,
    calc_area_km2,
    detect_loss,
    generate_optional_ai_briefing,
    geocode_place,
    hansen_truth,
    parse_coords_or_url,
    utm_crs,
    validate_run_params,
)
from exports import (
    clear_last_result,
    load_last_result,
    make_gif,
    make_markdown,
    make_overlay,
    save_last_result,
    to_geojson,
    to_png,
    to_web_jpeg_b64,
)

logger = logging.getLogger(__name__)

APP_NAME = "Deforestation Scanner"
CONTACT_EMAIL = "konoha.labs0@gmail.com"

HERO_BEFORE = "https://storage.googleapis.com/planet-t2/hasdeo-arand-adani-coal-mines-YlY_kqCNg/frame-000.png"
HERO_AFTER = "https://storage.googleapis.com/planet-t2/hasdeo-arand-adani-coal-mines-YlY_kqCNg/frame-001.png"
CASE_STUDY_URL = "https://sgveeravalli.com/blog/2025/hasdeoarand/"
HASDEO_BBOX = (82.775, 22.795, 82.845, 22.854)
REGIONS = {"Hasdeo Arand PEKB Mine, India": HASDEO_BBOX, **PRESETS}



def _make_favicon():
    img = Image.new("RGB", (32, 32), color=(16, 35, 26))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, 29, 29], outline=(47, 122, 77), width=2)
    draw.rectangle([7, 7, 15, 15], fill=(47, 122, 77))
    draw.rectangle([17, 7, 25, 15], fill=(31, 92, 63))
    draw.rectangle([7, 17, 15, 25], fill=(31, 92, 63))
    draw.rectangle([17, 17, 25, 25], fill=(226, 85, 31))
    return img


def _search_region_from_widget():
    """Resolve the visible AOI search field and switch the workbench to custom bounds."""
    query = st.session_state.get("place_search", "").strip()
    if not query:
        st.session_state["place_search_message"] = "Enter a place name or latitude, longitude."
        return

    parsed = parse_coords_or_url(query)
    if parsed:
        lat, lon = parsed
        place_name = f"{lat:.4f}, {lon:.4f}"
    else:
        geo, geo_error = geocode_place(query)
        if not geo:
            st.session_state["place_search_message"] = geo_error or "Location not found. Try coordinates in lat, lon format."
            return
        lat, lon, place_name = geo

    if abs(lat) > 90 or abs(lon) > 180:
        st.session_state["place_search_message"] = "Coordinates are outside valid latitude or longitude ranges."
        return

    st.session_state["custom_bbox"] = (
        round(lon - 0.06, 4),
        round(max(-90.0, lat - 0.06), 4),
        round(lon + 0.06, 4),
        round(min(90.0, lat + 0.06), 4),
    )
    st.session_state["region_preset"] = "Select on Map / Custom AOI"
    for key, value in zip(("aoi_west", "aoi_south", "aoi_east", "aoi_north"), st.session_state["custom_bbox"]):
        st.session_state[key] = float(value)
    st.session_state["place_search_message"] = f"Centered a custom area on {place_name}."


def _sync_aoi_fields_to_region():
    region = st.session_state.get("region_preset")
    bounds = REGIONS.get(region)
    if bounds is None:
        bounds = st.session_state.get("custom_bbox", HASDEO_BBOX)
    else:
        st.session_state["custom_bbox"] = bounds
        st.session_state["place_search_message"] = ""
    for key, value in zip(("aoi_west", "aoi_south", "aoi_east", "aoi_north"), bounds):
        st.session_state[key] = float(value)


def _apply_aoi_field_edits():
    bounds = tuple(
        float(st.session_state.get(key, default))
        for key, default in zip(
            ("aoi_west", "aoi_south", "aoi_east", "aoi_north"), HASDEO_BBOX
        )
    )
    west, south, east, north = bounds
    if west >= east or south >= north or south < -90 or north > 90 or west < -180 or east > 180:
        st.session_state["place_search_message"] = "Bounds are invalid. Check west/east and south/north values."
        return
    st.session_state["custom_bbox"] = bounds
    st.session_state["region_preset"] = "Select on Map / Custom AOI"
    st.session_state["place_search_message"] = "Custom area bounds updated."



def render_swipe_slider(rgb_b, rgb_a, label_b, label_a):
    b64_b = to_web_jpeg_b64(rgb_b, max_dim=960, quality=82)
    b64_a = to_web_jpeg_b64(rgb_a, max_dim=960, quality=82)
    h, w = rgb_b.shape[:2]
    aspect = round((h / max(w, 1)) * 100, 2)
    iframe_h = int(min(max(860 * (h / max(w, 1)), 420), 820)) + 20
    html_parts = [
        "<style>*{box-sizing:border-box;margin:0;padding:0}body{background:transparent;font-family:system-ui,Arial,sans-serif;overflow:hidden}",
        f".wrap{{position:relative;width:100%;padding-bottom:{aspect}%;border:1px solid #c5d1c8;border-radius:2px;overflow:hidden;user-select:none}}",
        ".wrap img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:fill;pointer-events:none}",
        "#ib{clip-path:inset(0 50% 0 0)}",
        ".ln{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;transform:translateX(-50%);pointer-events:none;z-index:4}",
        ".hd{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);padding:4px 8px;border-radius:2px;background:#10231a;border:1px solid #fff;color:#fff;font-size:10px;font-weight:600;pointer-events:none;z-index:5}",
        ".lb{position:absolute;top:10px;padding:4px 10px;background:rgba(16,35,26,.9);color:#f8fafc;font-size:11px;font-weight:600;border-radius:2px;pointer-events:none;z-index:6}",
        ".ll{left:10px}.lr{right:10px}",
        ".rng{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;z-index:10}</style>",
        f"<div class='wrap'><img src='data:image/jpeg;base64,{b64_a}'><img id='ib' src='data:image/jpeg;base64,{b64_b}'>",
        "<div id='ln' class='ln'></div><div id='hd' class='hd'>SLIDE</div>",
        f"<div class='lb ll'>{label_b}</div><div class='lb lr'>{label_a}</div>",
        "<input type='range' min='0' max='100' value='50' class='rng' oninput=\"var v=this.value;document.getElementById('ib').style.clipPath='inset(0 '+(100-v)+'% 0 0)';document.getElementById('ln').style.left=v+'%';document.getElementById('hd').style.left=v+'%';\"></div>",
    ]
    components.html("".join(html_parts), height=iframe_h)


def _inject_theme_and_hero():
    # The preview's hero markup and styling live in this Python file. The
    # interactive controls below use Streamlit/Python so they stay connected
    # to the real analysis, map, validation, and export functions.
    try:
        tf = Path(".streamlit/config.toml")
        tf.parent.mkdir(exist_ok=True)
        if not tf.exists():
            tf.write_text(
                '[theme]\nbase = "light"\nprimaryColor = "#1f5c3f"\n'
                'backgroundColor = "#f4f7f3"\nsecondaryBackgroundColor = "#e9efe9"\n'
                'textColor = "#10231a"\n'
            )
    except OSError as exc:
        logger.warning("Could not write .streamlit/config.toml: %s", exc)

    page_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    :root {--ink:#10231a; --canopy:#267349; --ember:#c94a1a; --paper:#eaf2e9; --sheet:#fdfefb; --line:#bcd0c0; --muted:#496451; --leaf:#77b96a; --mist:#e5f0e6; --celadon:#d7e9d9; --sunlight:#f1d28b;}
    #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], .viewerBadge_container__1QSob, .viewerBadge_link__1S137 {display: none !important; visibility: hidden !important;}
    html, body, .stApp, [data-testid='stAppViewContainer'] {background-color: var(--paper); color: var(--ink); font-family: 'IBM Plex Sans', system-ui, sans-serif;}
    .stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb='select'] div {font-family: 'IBM Plex Sans', system-ui, sans-serif; border-radius: 2px !important;}
    [data-testid='stMain'], section.main {overflow-x: hidden;}
    .block-container, [data-testid='stMainBlockContainer'] {padding-top: 0 !important; padding-bottom: 0 !important; max-width: 1260px;}
    .cm-bleed {width: 100vw; position: relative; left: 50%; margin-left: -50vw; margin-bottom: 20px; border-bottom: 1px solid var(--line);}
    .cm-hero {position: relative; min-height: 360px; overflow: hidden; background: #12301f; isolation: isolate;}
    .cm-image-pair {position: absolute; inset: 0; z-index: 0; display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); overflow: hidden;}
    .cm-image {position: relative; min-width: 0; height: 100%; margin: 0; overflow: hidden;}
    .cm-image img {display: block; width: 100%; height: 100%; object-fit: cover; filter: saturate(1.18) contrast(1.05) brightness(1.02); transform: scale(1); transition: transform 0.65s cubic-bezier(.2,.7,.2,1), filter 0.4s ease;}
    .cm-image:hover img, .cm-image:focus-visible img {transform: scale(1.045); filter: saturate(1.25) contrast(1.07) brightness(1.06);}
    .cm-image:focus-visible {outline: 2px solid #ffffff; outline-offset: -4px; z-index: 2;}
    .cm-image-pair::after {content: ''; position: absolute; inset: 0; z-index: 1; pointer-events: none; background: rgba(35,90,43,0.12); mix-blend-mode: color;}
    .cm-hero:focus-visible {outline: 3px solid rgba(255,255,255,0.9); outline-offset: -3px;}
    .cm-scrim {position: absolute; inset: 0; z-index: 1; pointer-events: none; background: linear-gradient(180deg, rgba(10, 22, 16, 0.62) 0%, rgba(10, 22, 16, 0.18) 42%, rgba(10, 22, 16, 0.72) 100%);}
    .cm-top {position: relative; z-index: 3; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 6%; border-bottom: 1px solid rgba(255,255,255,0.15); background: rgba(10, 22, 16, 0.65);}
    .cm-brand {display: inline-flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; color: #ffffff; letter-spacing: 0.02em; text-transform: uppercase;}
    .cm-mark {width: 28px; height: 28px; display: block; flex: 0 0 28px;}
    .cm-spec-tag {font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: rgba(255,255,255,0.9); text-align: right;}
    .cm-year {position: absolute; z-index: 3; top: 68px; display: flex; flex-direction: column; gap: 2px; padding: 6px 10px; border-radius: 2px; background: rgba(10, 22, 16, 0.88); border: 1px solid rgba(255,255,255,0.35); color: #ffffff; line-height: 1.2;}
    .cm-year b {font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.88rem;}
    .cm-year span {font-size: 0.74rem; opacity: 0.9;}
    .cm-y1 {left: 6%;}
    .cm-y2 {right: 6%; border-left: 3px solid var(--ember);}
    .cm-split {position: absolute; z-index: 3; top: 53px; bottom: 0; left: 50%; width: 1px; background: #ffffff;}
    .cm-split {transition: width 0.2s ease, box-shadow 0.2s ease; pointer-events: none;}
    .cm-hero:has(.cm-image:hover) .cm-split, .cm-hero:has(.cm-image:focus-visible) .cm-split {width: 2px; box-shadow: 0 0 12px rgba(255,255,255,0.75);}
    .cm-year {transition: transform 0.2s ease, border-color 0.2s ease, background-color 0.2s ease;}
    .cm-year:hover {transform: translateY(-2px); border-color: #ffffff; background: rgba(10,22,16,0.96);}
    .cm-copy {position: absolute; z-index: 3; left: 6%; bottom: 28px; max-width: 46%;}
    .cm-h {font-weight: 700; font-size: clamp(1.5rem, 2.8vw, 2.25rem); line-height: 1.15; letter-spacing: -0.01em; color: #ffffff; margin: 0 0 8px 0;}
    .cm-sub {font-size: 0.92rem; line-height: 1.45; color: rgba(255, 255, 255, 0.94); max-width: 580px;}
    .cm-facts {background: var(--ink); color: #ffffff; display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid rgba(255,255,255,0.15);}
    .cm-fact {padding: 12px 24px; border-left: 1px solid rgba(255, 255, 255, 0.14);}
    .cm-fact {transition: background-color 0.2s ease;}
    .cm-fact:hover {background: rgba(255,255,255,0.06);}
    .cm-fact:first-child {border-left: none; padding-left: 6vw;}
    .cm-fact b {display: block; font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.82rem; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.03em;}
    .cm-fact span {font-size: 0.8rem; opacity: 0.8;}
    div[data-testid='stVerticalBlockBorderWrapper'] {background-color: var(--sheet); border: 1px solid var(--line) !important; border-radius: 2px !important; padding: 10px 8px;}
    [data-testid='stExpander'] {border: 1px solid var(--line); border-radius: 2px !important;}
    .t-strip {background: #e9efe9; color: var(--ink); border: 1px solid var(--line); border-left: 3px solid var(--canopy); border-radius: 2px; padding: 10px 14px; margin: 6px 0 14px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;}
    .t-strip a {color: var(--canopy); font-weight: 600; text-decoration: underline;}
    .caveat-box {display:flex; align-items:flex-start; gap:10px; background: #fff8f1; border: 1px solid #e6c5a3; border-left: 3px solid var(--ember); border-radius: 2px; padding: 10px 12px; margin: 8px 0 16px 0; font-size: 0.78rem; line-height: 1.45; color: var(--ink);}
    .caveat-box b {font-weight: 600;}
    .caveat-mark {display:grid; place-items:center; width:18px; height:18px; flex:0 0 18px; border:1px solid var(--ember); color:var(--ember); font:0.76rem 'IBM Plex Mono',monospace;}
    .caveat-copy {flex:1; min-width:0;}
    .caveat-copy p {margin:3px 0 0; color:#654a3e;}
    .caveat-meta {margin-left:auto; padding-top:2px; white-space:nowrap; color:#a95330; font:0.62rem 'IBM Plex Mono',monospace;}
    .p-label {display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 0.95rem; line-height: 1.25; color: var(--ink); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.02em;}
    .p-num {flex: none; width: 20px; height: 20px; border-radius: 2px; background: var(--ink); color: #ffffff; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; justify-content: center;}
    [data-testid='stMetric'] {background: #ffffff; border: 1px solid var(--line); border-top: 3px solid var(--ink); border-radius: 2px; padding: 10px 12px;}
    [data-testid='stMetricLabel'] p {font-size: 0.78rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase;}
    [data-testid='stMetricValue'] {font-family: 'IBM Plex Mono', monospace !important; font-size: 1.55rem !important; font-weight: 600 !important; color: var(--ink) !important;}
    .stButton > button, .stDownloadButton > button {border-radius: 2px !important; font-weight: 600; font-size: 0.88rem; padding: 0.55rem 1rem;}
    .stApp .stButton > button, .stApp .stDownloadButton > button {transition: transform 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;}
    .stApp .stButton > button:hover:not(:disabled), .stApp .stDownloadButton > button:hover:not(:disabled) {transform: translateY(-1px); box-shadow: 0 4px 10px rgba(16,35,26,0.16);}
    .stApp .stButton > button:active, .stApp .stDownloadButton > button:active {transform: scale(0.99);}
    .stApp input:not([type='checkbox']):not([type='radio']):not([type='range']), .stApp textarea, .stApp [data-baseweb='select'] > div {transition: border-color 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;}
    .stButton > button[kind='primary'], [data-testid='stBaseButton-primary'] {background-color: var(--canopy); color: #ffffff; border: 1px solid var(--ink); border-radius: 2px !important;}
    .stButton > button[kind='primary']:hover, [data-testid='stBaseButton-primary']:hover {background-color: var(--ink); color: #ffffff;}
    .stDownloadButton > button, [data-testid='stBaseButton-secondary'] {background-color: var(--sheet); color: var(--ink); border: 1px solid var(--ink); border-radius: 2px !important;}
    .cm-foot {width: 100vw; position: relative; left: 50%; margin-left: -50vw; margin-top: 48px; background: var(--ink); color: #ffffff; border-top: 2px solid var(--canopy);}
    .cm-foot-grid {display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 32px; padding: 32px 6vw;}
    .cm-foot-copy {font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 10px;}
    .cm-foot-note {font-size: 0.82rem; line-height: 1.5; color: rgba(255, 255, 255, 0.82); max-width: 380px;}
    .cm-foot-head {font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.85rem; color: #ffffff; margin-bottom: 8px; text-transform: uppercase;}
    .stApp .cm-foot a.cm-link {display: block; padding: 2px 0; font-size: 0.82rem; color: rgba(255, 255, 255, 0.85); text-decoration: none;}
    .stApp .cm-foot a.cm-link:hover {text-decoration: underline;}
    .cm-facts {background: var(--ink);}
    div[data-testid='stVerticalBlockBorderWrapper'] {background: var(--sheet); border-color: var(--line) !important; box-shadow: none;}
    div[data-testid='stVerticalBlockBorderWrapper']:has(.p-state) {padding: 0 !important; background: var(--sheet) !important; border: 1px solid var(--line) !important;}
    div[data-testid='stVerticalBlockBorderWrapper']:has(.p-state) > div[data-testid='stVerticalBlock'] {gap: 0 !important;}
    div[data-testid='stHorizontalBlock']:has(.p-state) {gap: 0 !important;}
    div[data-testid='stHorizontalBlock']:has(.p-state) > div[data-testid='stColumn'] {min-width: 0; padding: 14px 16px 13px; border-right: 1px solid var(--line);}
    div[data-testid='stHorizontalBlock']:has(.p-state) > div[data-testid='stColumn']:last-child {border-right: 0;}
    [data-testid='stExpander'] {background: var(--sheet); border-color: var(--line);}
    .p-label {color: var(--ink); text-transform: none; letter-spacing: 0; font-size: 0.9rem;}
    .p-num {width: auto; height: auto; padding: 0; color: var(--canopy); background: transparent; font-size: 0.78rem;}
    .p-state {margin-left: auto; padding: 3px 6px; border: 1px solid var(--line); color: var(--muted); font: 0.65rem 'IBM Plex Mono', monospace; letter-spacing: 0.06em;}
    .cm-months-head {display:flex; justify-content:space-between; align-items:center; margin:12px 0 1px; color:var(--muted); font-size:0.72rem;}
    .cm-months-head span:last-child {color:var(--canopy); font:0.65rem 'IBM Plex Mono',monospace;}
    .stApp [data-testid='stWidgetLabel'] p, .stApp [data-testid='stWidgetLabel'] label, .stApp label {color:var(--muted) !important; font-size:0.72rem !important; line-height:1.25 !important; font-weight:500 !important;}
    .stApp input:not([type='checkbox']):not([type='radio']):not([type='range']), .stApp textarea {min-height:31px !important; padding:6px 8px !important; color:var(--ink) !important; background:#fcfdfc !important; border:1px solid var(--line) !important; border-radius:2px !important; font-size:0.78rem !important;}
    .stApp [data-baseweb='select'] > div {min-height:31px !important; color:var(--ink) !important; background:#fcfdfc !important; border:1px solid var(--line) !important; border-radius:2px !important; box-shadow:none !important;}
    .stApp [data-testid='stNumberInput'] button, .stApp [data-testid='stNumberInputStepDown'], .stApp [data-testid='stNumberInputStepUp'] {display:none !important;}
    .stApp [data-testid='stNumberInput'] input {font-variant-numeric:tabular-nums;}
    .stApp [data-testid='stCaptionContainer'] p {margin:3px 0 0 !important; color:#1c3d2b !important; font:0.78rem/1.45 'IBM Plex Sans',system-ui,sans-serif !important;}
    .stApp [data-testid='stThumbValue'] {display:none !important;}
    .stApp input[type='checkbox'] {accent-color:var(--canopy);}
    .cm-workbench-head {display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin: 0 0 13px; color:var(--ink);}
    .cm-workbench-kicker {color:var(--muted); font:0.68rem 'IBM Plex Mono',monospace; letter-spacing:0.11em; text-transform:uppercase;}
    .cm-workbench-kicker span {margin-left:8px; color:#718976;}
    .cm-workbench-title {margin:4px 0 0; font-size:1.45rem; line-height:1.15; font-weight:600; letter-spacing:-0.025em;}
    .cm-workbench-status {display:flex; align-items:center; gap:7px; color:var(--canopy); font:0.66rem 'IBM Plex Mono',monospace; letter-spacing:0.04em;}
    .cm-workbench-status i {width:7px; height:7px; display:inline-block; background:var(--leaf); box-shadow:0 0 0 3px rgba(119,185,106,0.16);}
    [data-testid='stMetric'] {background: #e8f2e8; border-top-color: #367d53;}
    [data-testid='stMetricValue'] {color: #206342 !important;}
    .caveat-box {background: #fff7e8; border-color: #e8d09c; border-left-color: #c18a34;}
    .cm-foot {background: var(--ink); border-top-color: var(--canopy);}
    .stButton > button[kind='primary']:hover, [data-testid='stBaseButton-primary']:hover {background-color: #20583a; color: #ffffff;}
    @media (max-width: 900px) {
      .cm-foot-grid {grid-template-columns: 1fr; gap: 24px; padding: 24px 6vw;}
      .cm-hero {min-height: 360px;}
      .cm-image {min-height: 360px;}
      .cm-copy {max-width: 84%; bottom: 58px;}
      .cm-facts {grid-template-columns: 1fr;}
      .cm-top {align-items: flex-start; flex-direction: column;}
      .cm-spec-tag {text-align: left;}
      .cm-workbench-head {align-items:flex-start; flex-direction:column;}
      div[data-testid='stHorizontalBlock']:has(.p-state) > div[data-testid='stColumn'] {padding:12px 11px;}
    }
    @media (max-width: 700px) {
      div[data-testid='stHorizontalBlock']:has(.p-state) {flex-direction:column; gap:0 !important;}
      div[data-testid='stHorizontalBlock']:has(.p-state) > div[data-testid='stColumn'] {width:100% !important; border-right:0; border-bottom:1px solid var(--line);}
      div[data-testid='stHorizontalBlock']:has(.p-state) > div[data-testid='stColumn']:last-child {border-bottom:0;}
    }
    @media (prefers-reduced-motion: reduce) {
      html {scroll-behavior: auto;}
      *, *::before, *::after {animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; scroll-behavior: auto !important;}
    }
    """

    hero_html = "".join([
        "<div class='cm-bleed'><div class='cm-hero' role='group' aria-label='Satellite comparison for the same Hasdeo Arand site'>",
        "<div class='cm-image-pair'>",
        f"<figure class='cm-image' tabindex='0'><img src='{HERO_BEFORE}' alt='Hasdeo Arand PEKB coal mine in 2016, before expansion'></figure>",
        f"<figure class='cm-image' tabindex='0'><img src='{HERO_AFTER}' alt='The same Hasdeo Arand site in 2025, after mine expansion'></figure>",
        "</div><div class='cm-scrim'></div>",
        f"<div class='cm-top'><span class='cm-brand'><svg class='cm-mark' viewBox='0 0 32 32' aria-hidden='true' focusable='false'><path d='M5.5 23.8C7.2 15.2 14.6 7.8 25.4 6.2c1.4 8.5-2 17.4-10 19.3-3.8.9-7.2.3-9.9-1.7Z' fill='#a8d98c'/><path d='M6.5 26c4.8-4.3 10.2-9.2 16.7-15.2' fill='none' stroke='#214c31' stroke-linecap='round' stroke-width='1.5'/><path d='M13 20.1c-.4-3-2.3-5.3-5.2-6.4M16.7 16.8c2.8.2 4.9 1.6 6 4' fill='none' stroke='#356644' stroke-linecap='round' stroke-width='1.25'/></svg>{APP_NAME}</span>",
        "<span class='cm-spec-tag'>REFERENCE CASE STUDY</span></div>",
        "<div class='cm-year cm-y1'><b>2016 | BEFORE</b><span>PEKB coal mine, Chhattisgarh</span></div>",
        "<div class='cm-year cm-y2'><b>2025 | AFTER</b><span>Same mine area after expansion</span></div>",
        "<div class='cm-split'></div>",
        "<div class='cm-copy'><div class='cm-h'>Hasdeo Arand, India</div>",
        "<div class='cm-sub'>Published PlanetScope imagery of the PEKB coal mine area, before and after major expansion. The source study estimates about 1,013 ha of forest cleared by 2025.</div></div>",
        "</div>",
        "<div class='cm-facts'>",
        "<div class='cm-fact'><b>HIGH-RESOLUTION IMAGERY</b><span>Cloud-masked Sentinel-2 satellite composites sharp down to 10 metres</span></div>",
        "<div class='cm-fact'><b>VERIFIED RESULTS</b><span>Strict and buffered accuracy checks against UMD Hansen reference data</span></div>",
        "<div class='cm-fact'><b>OPEN GIS EXPORTS</b><span>Analyze any custom map area and download GeoJSON polygons, CSV metrics, and PNGs</span></div>",
        "</div></div>",
    ])
    st.markdown("<style>" + page_css + "</style>" + hero_html, unsafe_allow_html=True)


def _render_privacy_policy():
    with st.container(border=True):
        st.markdown("### Terms and Conditions")
        st.caption("Last updated: October 2026")
        st.markdown(
            """
            **1. Please read before using the results**

            This tool finds changes in tree cover using Sentinel-2 satellite images. A drop in the satellite signal does not prove deforestation. Dry-season leaf fall, crop harvests and wildfires can look the same. Please check the results with field visits or high-resolution images before using them for legal, government or business work.

            **2. Area limits per search**

            To keep the server running well, each search has a size limit:
            - 10 m detail: up to 150 km² (75 km² for multi-year timelines)
            - 20 m detail: up to 500 km²
            - 40 m detail: up to 900 km²

            Please do not use bots or scripts to flood the data servers.

            **3. Data credits**
            - Satellite images: Copernicus Sentinel-2 (ESA) via Microsoft Planetary Computer
            - Forest loss reference map: Hansen/UMD/Google/USGS/NASA (Global Forest Change v1.12)
            - Base maps: OpenStreetMap contributors, Esri, USGS, NOAA

            **4. No warranty**

            This tool is provided "as is." We do not promise it is accurate or right for any particular purpose.
            """
        )


def _render_terms_and_conditions():
    with st.container(border=True):
        st.markdown("### Terms and Conditions")
        st.caption("Last updated: October 2026")
        st.markdown(
            """
            **1. Using this tool**

            This tool is free and made for learning and research. You are responsible for how you use the results.

            **2. Results are not proof**

            The tool shows possible tree cover loss, not confirmed deforestation. Please verify results before using them for important decisions.

            **3. What you must not do**
            - Don't use the results to blame a person or company without checking the facts first.
            - Don't try to break, copy or misuse the app.

            **4. Changes**

            We may update these terms from time to time. The date at the top shows the latest version.

            **5. Questions?**

            Use the contact details in the footer.
            """
        )


def _scroll_to_run_area():
    """Smoothly scroll the page down to the spot where the analysis runs."""
    st.markdown(
        "<div id='run-status-anchor' style='scroll-margin-top:90px'></div>",
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const calm = window.parent.matchMedia("(prefers-reduced-motion: reduce)").matches;
        let tries = 0;
        const timer = setInterval(() => {
            const target =
                doc.getElementById("run-status-anchor") ||
                doc.querySelector('[data-testid="stStatus"]');
            tries += 1;
            if (target) {
                target.scrollIntoView({behavior: calm ? "auto" : "smooth", block: "start"});
                clearInterval(timer);
            } else if (tries > 20) {
                clearInterval(timer);
            }
        }, 100);
        </script>
        """,
        height=0,
    )


def main():
    st.set_page_config(
        page_title=f"{APP_NAME} | Geospatial Workbench",
        page_icon=_make_favicon(),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_theme_and_hero()

    query_params = st.query_params
    view_page = query_params.get("view", "workbench")
    today = datetime.date.today()  # needed by the footer on every page

    pending_toast = st.session_state.pop("pending_toast", None)
    if pending_toast:
        st.toast(pending_toast[0], icon=pending_toast[1])

    if view_page == "privacy":
        _render_privacy_policy()
        if st.button("Back to Workbench"):
            st.query_params.clear()
            st.rerun()
    elif view_page == "terms":
        _render_terms_and_conditions()
        if st.button("Back to Workbench"):
            st.query_params.clear()
            st.rerun()
    else:
        if "result" not in st.session_state:
            cached = load_last_result()
            if cached is not None:
                st.session_state["result"] = cached
                st.session_state["restored_from_disk"] = True

        if "custom_bbox" not in st.session_state:
            st.session_state["custom_bbox"] = HASDEO_BBOX
        if "region_preset" not in st.session_state:
            st.session_state["region_preset"] = next(iter(REGIONS))
        if st.session_state.pop("_sync_aoi_fields_after_map", False):
            for key, value in zip(("aoi_west", "aoi_south", "aoi_east", "aoi_north"), st.session_state["custom_bbox"]):
                st.session_state[key] = float(value)
        initial_region_bounds = REGIONS.get(st.session_state["region_preset"])
        if initial_region_bounds is None:
            initial_region_bounds = st.session_state["custom_bbox"]
        for key, value in zip(("aoi_west", "aoi_south", "aoi_east", "aoi_north"), initial_region_bounds):
            st.session_state.setdefault(key, float(value))

        st.markdown(
            "<div class='cm-workbench-head'>"
            "<div><div class='cm-workbench-kicker'>ANALYSIS WORKBENCH</div>"
            "<h2 class='cm-workbench-title'>Configure observation</h2></div>"
            "<div class='cm-workbench-status'><i></i> ANALYSIS ENGINE READY</div></div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            col_loc, col_time, col_cal = st.columns([1.1, 1.0, 1.0], gap="small")
            with col_loc:
                st.markdown("<div class='p-label'><span class='p-num'>01</span>Study area<span class='p-state'>AOI</span></div>", unsafe_allow_html=True)
                place = st.selectbox("Region preset", list(REGIONS), key="region_preset", on_change=_sync_aoi_fields_to_region)
                st.text_input(
                    "Place or coordinate search",
                    key="place_search",
                    placeholder="Place name or lat, lon",
                    on_change=_search_region_from_widget,
                )
                st.caption(st.session_state.get("place_search_message") or "Enter coordinates or a place name to locate an area.")
                c1, c2, c3, c4 = st.columns(4, gap="small")
                west = c1.number_input("West", format="%.3f", key="aoi_west", on_change=_apply_aoi_field_edits)
                south = c2.number_input("South", format="%.3f", key="aoi_south", on_change=_apply_aoi_field_edits)
                east = c3.number_input("East", format="%.3f", key="aoi_east", on_change=_apply_aoi_field_edits)
                north = c4.number_input("North", format="%.3f", key="aoi_north", on_change=_apply_aoi_field_edits)
                if REGIONS[place] is None:
                    bbox = (west, south, east, north)
                    if west < east and south < north:
                        st.session_state["custom_bbox"] = bbox
                else:
                    bbox = REGIONS[place]

            with col_time:
                st.markdown("<div class='p-label'><span class='p-num'>02</span>Time window<span class='p-state'>DATES</span></div>", unsafe_allow_html=True)
                t1, t2 = st.columns(2)
                y1 = int(t1.number_input("Baseline year", 2017, today.year, 2017, help="Live analysis default is 2017–2024. The poster above is a separate PlanetScope case study from 2016 and 2025."))
                y2 = int(t2.number_input("Target year", 2017, today.year, 2024, help="UMD Hansen Global Forest Change reference data currently covers through 2024."))
                month_names = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
                month_values = st.session_state.get("acquisition_months", (10, 12))
                st.markdown(
                    f"<div class='cm-months-head'><span>Acquisition months</span><span>{month_names[month_values[0] - 1]} - {month_names[month_values[1] - 1]}</span></div>",
                    unsafe_allow_html=True,
                )
                m1, m2 = st.slider("Acquisition months", 1, 12, (10, 12), key="acquisition_months", label_visibility="collapsed")
                st.caption("Cloud-masked seasonal window applied to both years.")
                if y2 > 2024:
                    st.caption("Note: Hansen GFC reference data covers through 2024. Use Target Year <= 2024 for non-zero Hansen validation metrics.")

            with col_cal:
                st.markdown("<div class='p-label'><span class='p-num'>03</span>Analysis setup<span class='p-state'>EXECUTE</span></div>", unsafe_allow_html=True)
                p1, p2 = st.columns(2)
                res = p1.selectbox("Resolution", [10, 20, 40], index=1, format_func=lambda value: f"{value} m / px", help="20 m is recommended. 10 m is capped at 150 km² to prevent memory crashes.")
                max_cloud = p2.number_input("Max cloud %", 5, 60, 20)
                k1, k2 = st.columns(2)
                timeline = k1.checkbox("Annual timeline", value=False)
                check_acc = k2.checkbox("Verify vs. Hansen", value=True)

                # Dynamic button text state showing "Running analysis..." when clicked
                run_label = "Running analysis..." if st.session_state.get("is_running", False) else "Run analysis"
                go = st.button(run_label, type="primary", use_container_width=True)

        with st.expander("Advanced spectral thresholds"):
            a1, a2, a3 = st.columns(3, gap="large")
            forest_thresh = a1.slider("Baseline NDVI floor (greenness threshold for initial canopy)", 0.35, 0.70, 0.50)
            min_drop = a2.slider("Minimum dNDVI drop (minimum spectral drop to flag canopy loss)", 0.15, 0.50, 0.22)
            biomass = a3.slider("Aboveground biomass (t/ha, for rough carbon equivalent)", 100, 400, 200)

        st.markdown(
            "<aside class='caveat-box'><span class='caveat-mark'>!</span><div class='caveat-copy'>"
            "<b>What this tool can and cannot tell you</b><p>"
            "Spectral canopy loss (dNDVI / dNBR) detects drops in greenness or moisture. It does not prove permanent deforestation. "
            "Wildfires, crop harvests, and dry-season deciduous leaf-fall can look similar in satellite observations.</p></div>"
            "<span class='caveat-meta'>INTERPRET WITH CARE</span></aside>",
            unsafe_allow_html=True,
        )

        c_lat, c_lon = (bbox[1] + bbox[3]) / 2.0, (bbox[0] + bbox[2]) / 2.0
        area_km2 = calc_area_km2(bbox)
        max_allowed_km2 = MAX_AREA_BY_RES.get(res, 500.0)
        osm_url = f"https://www.openstreetmap.org/#map=12/{c_lat:.4f}/{c_lon:.4f}"
        st.markdown(
            f"<div class='t-strip'><div><b>Area bounds</b> [{bbox[0]:.4f}E, {bbox[1]:.4f}N, {bbox[2]:.4f}E, {bbox[3]:.4f}N] &nbsp;|&nbsp; "
            f"<b>Projection</b> {utm_crs(bbox)} &nbsp;|&nbsp; <b>Footprint</b> {area_km2:.1f} km2 (Limit at {res}m: {max_allowed_km2:.0f} km2)</div>"
            f"<div><a href='{osm_url}' target='_blank' rel='noopener'>View on OpenStreetMap</a></div></div>",
            unsafe_allow_html=True,
        )

        if REGIONS[place] is None:
            with st.container(border=True):
                st.markdown("<div class='p-label'>Select area on the map: click to recenter the green box, or draw a rectangle</div>", unsafe_allow_html=True)
                w_c, s_c, e_c, n_c = st.session_state["custom_bbox"]
                m = folium.Map(location=[(s_c + n_c) / 2.0, (w_c + e_c) / 2.0], zoom_start=11, tiles=None)

                compact_layer_css = """
                <style>
                .leaflet-control-layers {
                    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
                    font-size: 11px !important;
                    line-height: 1.25 !important;
                    padding: 4px 8px !important;
                    border-radius: 2px !important;
                    border: 1px solid #99a89e !important;
                    box-shadow: none !important;
                }
                .leaflet-control-layers label {
                    margin-bottom: 2px !important;
                    display: flex !important;
                    align-items: center !important;
                    gap: 4px !important;
                }
                .leaflet-control-layers-selector {
                    margin: 0 !important;
                    transform: scale(0.85);
                }
                .leaflet-control-layers-separator {
                    margin: 4px 0 !important;
                }
                </style>
                """
                m.get_root().header.add_child(folium.Element(compact_layer_css))

                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Tiles &copy; Esri",
                    name="Esri Satellite",
                    overlay=False,
                    control=True,
                ).add_to(m)
                folium.TileLayer(
                    tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    attr="&copy; OpenStreetMap contributors",
                    name="OpenStreetMap Streets",
                    overlay=False,
                    control=True,
                ).add_to(m)
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                    attr="Labels &copy; Esri",
                    name="Place Names & Labels",
                    overlay=True,
                    control=True,
                    show=True,
                ).add_to(m)
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
                    attr="Roads &copy; Esri",
                    name="Roads & Highways",
                    overlay=True,
                    control=True,
                    show=True,
                ).add_to(m)
                folium.LayerControl(collapsed=False).add_to(m)
                folium.Rectangle(bounds=[[s_c, w_c], [n_c, e_c]], color="#267349", weight=2, fill=True, fill_color="#267349", fill_opacity=0.10).add_to(m)
                Draw(
                    export=False,
                    draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False, "circlemarker": False, "rectangle": {"shapeOptions": {"color": "#1f5c3f", "weight": 2}}},
                ).add_to(m)
                map_data = st_folium(m, height=440, use_container_width=True, key="aoi_map_picker", returned_objects=["last_active_drawing", "last_clicked"])
                if map_data:
                    drw = map_data.get("last_active_drawing")
                    if drw and drw.get("geometry", {}).get("type") == "Polygon":
                        pts = drw["geometry"]["coordinates"][0]
                        lons, lats = [p[0] for p in pts], [p[1] for p in pts]
                        nb = (round(min(lons), 4), round(min(lats), 4), round(max(lons), 4), round(max(lats), 4))
                        if nb != st.session_state["custom_bbox"]:
                            st.session_state["custom_bbox"] = nb
                            st.session_state["_sync_aoi_fields_after_map"] = True
                            st.rerun()
                    elif map_data.get("last_clicked"):
                        clat, clon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
                        nb = (round(clon - 0.06, 4), round(clat - 0.06, 4), round(clon + 0.06, 4), round(clat + 0.06, 4))
                        if nb != st.session_state["custom_bbox"]:
                            st.session_state["custom_bbox"] = nb
                            st.session_state["_sync_aoi_fields_after_map"] = True
                            st.rerun()

        if go:
            st.session_state["is_running"] = True
            st.rerun()

        if st.session_state.get("is_running", False):
            st.session_state["is_running"] = False
            _scroll_to_run_area()
            validation_errors = validate_run_params(bbox, y1, y2, m1, m2, res, timeline, today=today)
            if validation_errors:
                for err_msg in validation_errors:
                    st.error(err_msg)
                st.stop()

            west, south, east, north = bbox
            years = list(range(y1 + 1, y2 + 1)) if timeline else [y2]

            with st.status("Executing Sentinel-2 STAC multi-spectral pipeline...", expanded=True) as status:
                st.write(f"Compositing baseline Sentinel-2 L2A tiles for {y1}...")
                base_comp, base_err = build_composite(tuple(bbox), y1, m1, m2, max_cloud, res)
                if base_comp is None:
                    status.update(label="Pipeline stopped", state="error")
                    st.error(base_err)
                    st.stop()

                rgb_b = base_comp["rgb"]
                ndvi_b, nbr_b, ndwi_b = base_comp["ndvi"], base_comp["nbr"], base_comp["ndwi"]
                valid_b, n_b = base_comp["valid_mask"], base_comp["scene_count"]
                crs, transform = base_comp["crs"], base_comp["transform"]
                H_b, W_b = rgb_b.shape[:2]

                ha_px = (res * res) / 10000.0
                frames, tl_ha = [], []
                final_loss = final_forest = final_rgb_a = None
                n_a = 0
                H, W = H_b, W_b

                for yr in years:
                    st.write(f"Compositing target Sentinel-2 L2A tiles for {yr} & detecting canopy change...")
                    yr_comp, yr_err = build_composite(tuple(bbox), yr, m1, m2, max_cloud, res)
                    if yr_comp is None:
                        status.update(label="Pipeline stopped", state="error")
                        st.error(yr_err)
                        st.stop()

                    H = min(H_b, yr_comp["rgb"].shape[0])
                    W = min(W_b, yr_comp["rgb"].shape[1])

                    loss, forest = detect_loss(
                        ndvi_b[:H, :W], nbr_b[:H, :W], ndwi_b[:H, :W], valid_b[:H, :W],
                        yr_comp["ndvi"][:H, :W], yr_comp["nbr"][:H, :W], yr_comp["ndwi"][:H, :W], yr_comp["valid_mask"][:H, :W],
                        min_drop=min_drop, forest_thresh=forest_thresh,
                    )
                    frames.append(make_overlay(yr_comp["rgb"][:H, :W], loss))
                    tl_ha.append(float(loss.sum() * ha_px))

                    if yr == y2:
                        final_loss, final_forest = loss, forest
                        final_rgb_a = yr_comp["rgb"][:H, :W]
                        n_a = yr_comp["scene_count"]

                    del yr_comp
                    gc.collect()

                hansen_warnings = []
                acc = acc_img = last_year = hansen_ver = None
                if check_acc:
                    st.write("Loading Hansen GFC reference layers for independent validation...")
                    h_res, h_err = hansen_truth(tuple(bbox), crs, transform, (H, W), y1, y2)
                    if h_err:
                        hansen_warnings.append(h_err)
                    if h_res is not None:
                        last_year, hansen_ver = h_res["last_yr"], h_res["version"]
                        acc, acc_img = accuracy(final_loss, h_res["truth_strict"], final_rgb_a, truth_buffered=h_res["truth_buffered"])

                labeled, num_patches = ndimage.label(final_loss)
                patch_sizes = (np.bincount(labeled.ravel())[1:] * ha_px) if num_patches > 0 else np.array([])
                major_clearings = int((patch_sizes >= 5.0).sum())
                loss_ha, forest_ha = tl_ha[-1], float(final_forest.sum() * ha_px)
                co2 = loss_ha * biomass * 0.47 * 3.67

                facts = {
                    "region": place if REGIONS[place] else f"AOI ({west:.2f}, {south:.2f}, {east:.2f}, {north:.2f})",
                    "compared_years": f"{y1} vs {y2}",
                    "baseline_canopy_ha": round(forest_ha),
                    "canopy_loss_detected_ha": round(loss_ha, 1),
                    "percent_canopy_loss_detected": round(100.0 * loss_ha / max(forest_ha, 1.0), 2),
                    "disturbance_patches_count": int(num_patches),
                    "major_patches_over_5ha": major_clearings,
                    "potential_co2_equiv_tonnes": round(co2),
                }
                summary = (
                    f"Multi-spectral comparison between {y1} and {y2} detected {loss_ha:,.1f} ha "
                    f"({facts['percent_canopy_loss_detected']}%) of spectral canopy loss across {num_patches} "
                    f"disturbance patches ({major_clearings} patches >= 5 ha). If caused by permanent biomass "
                    f"removal rather than seasonal phenology or fire recovery, this corresponds to approximately {co2:,.0f} tCO2."
                )

                ai_briefing, ai_err = generate_optional_ai_briefing(facts, acc)
                if ai_err:
                    hansen_warnings.append(ai_err)

                geojson_str = to_geojson(final_loss, transform, crs, ha_px)
                gif = make_gif(frames, years, tl_ha, y1) if len(frames) > 1 else None
                status.update(label="Analysis complete", state="complete", expanded=False)

            result_bundle = dict(
                facts=facts, summary=summary, ai_briefing=ai_briefing, warnings=hansen_warnings,
                overlay=frames[-1], years=(y1, y2), scenes=(n_b, n_a),
                rgb_b=(rgb_b[:H, :W] * 255).astype("uint8"), rgb_a=(final_rgb_a * 255).astype("uint8"),
                tl=(years, tl_ha, frames), gif=gif, acc=acc, acc_img=acc_img,
                last_year=last_year, hansen_ver=hansen_ver, geojson=geojson_str,
            )
            st.session_state["result"] = result_bundle
            st.session_state["restored_from_disk"] = False
            save_err = save_last_result(result_bundle)
            if save_err:
                st.session_state["pending_toast"] = ("Analysis complete, but saving it to disk failed.", "⚠️")
            else:
                st.session_state["pending_toast"] = ("Analysis complete. Results are ready.", "✅")
            st.rerun()

        r = st.session_state.get("result")
        if r:
            if st.session_state.get("restored_from_disk"):
                rc1, rc2 = st.columns([5, 1])
                rc1.info("Restored your most recent analysis from local cache.")
                if rc2.button("Clear cached run", use_container_width=True):
                    clear_last_result()
                    del st.session_state["result"]
                    st.session_state["restored_from_disk"] = False
                    st.session_state["pending_toast"] = ("Saved run cleared.", "🗑️")
                    st.rerun()

            for w_msg in r.get("warnings", []):
                st.warning(w_msg)

            f, (y_before, y_after) = r["facts"], r["years"]
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Baseline canopy (ha)", f"{f['baseline_canopy_ha']:,}")
            k2.metric("Canopy loss detected (ha)", f"{f['canopy_loss_detected_ha']:,}")
            k3.metric("Canopy change (%)", f"{f['percent_canopy_loss_detected']}%")
            k4.metric("Disturbance patches", f"{f['disturbance_patches_count']:,}")
            k5.metric("Potential biomass CO2 (t)", f"{f['potential_co2_equiv_tonnes']:,}")

            with st.container(border=True):
                st.markdown("<div class='p-label'>Technical Summary</div>", unsafe_allow_html=True)
                st.write(r["summary"])
                if r.get("ai_briefing"):
                    st.caption("Server Briefing: " + r["ai_briefing"])

            names = ["Canopy Loss Overlay", "Split Comparison"]
            tl_years, tl_ha, tl_frames = r["tl"]
            if len(tl_frames) > 1:
                names.append("Annual Progression")
            if r.get("acc"):
                names.append("Hansen GFC Validation")
            names.append("Field Notes & Export")
            tabs = dict(zip(names, st.tabs(names)))

            with tabs["Canopy Loss Overlay"]:
                st.image(r["overlay"], caption=f"Red overlay indicates spectral canopy loss detected between {y_before} and {y_after}.", use_container_width=True)
            with tabs["Split Comparison"]:
                render_swipe_slider(r["rgb_b"], r["rgb_a"], f"{y_before} ({r['scenes'][0]} SCENES)", f"{y_after} ({r['scenes'][1]} SCENES)")
                l_col, r_col = st.columns(2)
                l_col.image(r["rgb_b"], caption=f"Baseline Composite: {y_before}", use_container_width=True)
                r_col.image(r["rgb_a"], caption=f"Target Composite: {y_after}", use_container_width=True)

            if "Annual Progression" in tabs:
                with tabs["Annual Progression"]:
                    yr = st.select_slider("Select Year", options=tl_years, value=tl_years[-1])
                    idx = tl_years.index(yr)
                    st.image(tl_frames[idx], caption=f"Cumulative canopy loss detected from {y_before} through {yr}: {tl_ha[idx]:,.1f} ha", use_container_width=True)
                    st.line_chart(pd.DataFrame({f"Cumulative Canopy Loss since {y_before} (ha)": tl_ha}, index=tl_years))
                    if r.get("gif"):
                        st.download_button("Download Time-Series GIF", r["gif"], file_name="canopy_progression.gif", mime="image/gif")

            if "Hansen GFC Validation" in tabs:
                with tabs["Hansen GFC Validation"]:
                    acc = r["acc"]
                    s_sc, b_sc = acc["strict"], acc["buffered"]
                    h_last = r.get("last_year") or 2024

                    st.markdown(
                        f"**Strict Score (Exact Pixel & Exact Year Window):** Direct pixel-by-pixel comparison "
                        f"between independent Sentinel-2 detection and Hansen GFC (`lossyear` strictly between {y_before + 1} and {min(y_after, h_last)})."
                    )
                    m_s = st.columns(4)
                    m_s[0].metric("STRICT PRECISION", s_sc["precision"])
                    m_s[1].metric("STRICT RECALL", s_sc["recall"])
                    m_s[2].metric("STRICT F1 SCORE", s_sc["f1"])
                    m_s[3].metric("STRICT IoU", s_sc["iou"])

                    st.markdown(
                        "**Buffered Score (+/-1 Pixel, +/-1 Year Tolerance):** Accounts for 10/20 m Sentinel-2 vs. "
                        "30 m Landsat pixel registration offsets and late-year acquisition timing differences."
                    )
                    m_b = st.columns(4)
                    m_b[0].metric("BUFFERED PRECISION", b_sc["precision"])
                    m_b[1].metric("BUFFERED RECALL", b_sc["recall"])
                    m_b[2].metric("BUFFERED F1 SCORE", b_sc["f1"])
                    m_b[3].metric("BUFFERED IoU", b_sc["iou"])

                    st.info(
                        "**What these scores mean:**\n"
                        "- **Precision:** Fraction of flagged Sentinel-2 canopy loss pixels that also appear in Hansen GFC.\n"
                        "- **Recall:** Fraction of Hansen GFC loss pixels captured by this detector.\n"
                        "- **F1 Score & IoU:** Combined overlap scores (0 = no overlap, 1 = identical)."
                    )
                    st.image(r["acc_img"], caption="Green = Agreement with Hansen GFC  |  Red = Flagged only by Sentinel-2  |  Blue = Flagged only by Hansen GFC", use_container_width=True)

            with tabs["Field Notes & Export"]:
                st.text_area("Regional Investigation Notes", key="story", height=140, placeholder="Record local context (e.g., dry deciduous phenology, fire history, crop harvest, or verified clearing)...")

            d1, d2, d3, d4 = st.columns(4)
            d1.download_button("Download Report (.MD)", make_markdown(r, st.session_state.get("story", "")), file_name="canopy_loss_report.md", use_container_width=True)
            d2.download_button("Download Telemetry (.CSV)", "\n".join(f"{k},{v}" for k, v in f.items()), file_name="canopy_loss_metrics.csv", use_container_width=True)
            d3.download_button("Download Loss Raster (.PNG)", to_png(r["overlay"]), file_name="canopy_loss_detected.png", mime="image/png", use_container_width=True)
            d4.download_button("Download Polygons (.GEOJSON)", r["geojson"], file_name="canopy_loss_detected.geojson", mime="application/geo+json", use_container_width=True)

        else:
            with st.container(border=True):
                st.markdown("<div class='p-label'>No results yet</div>", unsafe_allow_html=True)
                st.write(
                    "1. Pick an area on the map or from the list.\n"
                    "2. Choose the two years you want to compare.\n"
                    "3. Press **Run analysis**. Your results will show up here."
                )

    links = [
        ("Sentinel-2 satellite images (ESA)", "https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2"),
        ("Microsoft Planetary Computer", "https://planetarycomputer.microsoft.com"),
        ("Hansen Global Forest Change", "https://glad.earthengine.app/view/global-forest-change"),
        ("Hasdeo Arand PlanetScope comparison and study", CASE_STUDY_URL),
    ]
    footer_html = "".join([
        "<div class='cm-foot'><div class='cm-foot-grid'>",
        f"<div><div class='cm-foot-copy'>&copy; {today.year} {APP_NAME}</div>",
        "<div class='cm-foot-note'>Results are automated spectral estimates from Sentinel-2 imagery. Fire, seasonal leaf-fall, and crop rotation can resemble canopy loss. Always verify on the ground.</div></div>",
        "<div><div class='cm-foot-head'>Data and credits</div>",
        "".join(f"<a class='cm-link' href='{u}' target='_blank' rel='noopener'>{n}</a>" for n, u in links),
        "</div>",
        "<div><div class='cm-foot-head'>Legal &amp; Documentation</div>",
        "<div class='cm-foot-note'>",
        "<a class='cm-link' href='/?view=privacy' target='_self'>Privacy Policy</a>",
        "<a class='cm-link' href='/?view=terms' target='_self'>Terms &amp; Conditions</a>",
        f"<a class='cm-link' href='mailto:{CONTACT_EMAIL}'>Contact: {CONTACT_EMAIL}</a>",
        "</div></div>",
        "</div></div>",
    ])
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
