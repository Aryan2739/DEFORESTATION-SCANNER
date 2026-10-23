"""Generate 100% real Sentinel-2 banner prototypes (no AI) and save hero_before.jpg / hero_after.jpg."""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis import build_composite, detect_loss
from exports import make_overlay


def _crop_banner(arr, target_w=1600, target_h=460):
    """Center-crop and resize a satellite array to a clean edge-to-edge banner aspect ratio."""
    h, w = arr.shape[:2]
    target_ratio = target_w / float(target_h)
    current_ratio = w / float(h)

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        cropped = arr[:, x0 : x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        cropped = arr[y0 : y0 + new_h, :]

    img = Image.fromarray(cropped).convert("RGB")
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def _render_ui_mockup(img_left, img_right, left_tag, right_tag, out_filename):
    """Assemble the complete website header poster mockup with real satellite halves."""
    W, H_hero, H_bar = 1600, 420, 76
    canvas = Image.new("RGB", (W, H_hero + H_bar), (16, 35, 26))

    # Split at 56% width to match ui.py layout
    split_x = int(W * 0.56)
    left_crop = img_left.resize((W, H_hero)).crop((0, 0, split_x, H_hero))
    right_crop = img_right.resize((W, H_hero)).crop((split_x, 0, W, H_hero))
    canvas.paste(left_crop, (0, 0))
    canvas.paste(right_crop, (split_x, 0))

    # Apply subtle dark gradient scrim for text legibility
    overlay = Image.new("RGBA", (W, H_hero + H_bar), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Top bar scrim & bottom text scrim
    draw.rectangle([0, 0, W, 52], fill=(10, 22, 16, 195))
    draw.rectangle([0, H_hero - 150, W, H_hero], fill=(10, 22, 16, 175))
    draw.line([(split_x, 52), (split_x, H_hero)], fill=(255, 255, 255, 230), width=2)

    try:
        f_title = ImageFont.truetype("arialbd.ttf", 34)
        f_sub = ImageFont.truetype("arial.ttf", 16)
        f_mono = ImageFont.truetype("consolab.ttf", 15)
        f_small = ImageFont.truetype("consola.ttf", 13)
    except OSError:
        f_title = ImageFont.load_default()
        f_sub = f_mono = f_small = f_title

    # Top navigation brand bar
    draw.rectangle([80, 18, 96, 34], fill=(201, 74, 26, 255), outline=(255, 255, 255, 255), width=2)
    draw.text((108, 17), "CANOPY LOSS OBSERVATORY", fill=(255, 255, 255, 255), font=f_mono)
    draw.text((W - 390, 18), "SENTINEL-2 L2A / HANSEN GFC v1.12", fill=(215, 225, 218, 255), font=f_small)

    # Left badge
    draw.rectangle([80, 72, 470, 124], fill=(10, 22, 16, 225), outline=(255, 255, 255, 80), width=1)
    draw.text((92, 80), left_tag[0], fill=(255, 255, 255, 255), font=f_mono)
    draw.text((92, 101), left_tag[1], fill=(200, 215, 205, 255), font=f_small)

    # Right badge
    draw.rectangle([W - 490, 72, W - 80, 124], fill=(10, 22, 16, 225), outline=(255, 255, 255, 80), width=1)
    draw.rectangle([W - 490, 72, W - 486, 124], fill=(201, 74, 26, 255))
    draw.text((W - 474, 80), right_tag[0], fill=(255, 255, 255, 255), font=f_mono)
    draw.text((W - 474, 101), right_tag[1], fill=(200, 215, 205, 255), font=f_small)

    # Main technical heading
    draw.text((80, H_hero - 115), "Sentinel-2 L2A Multi-Spectral Change Detection", fill=(255, 255, 255, 255), font=f_title)
    draw.text(
        (80, H_hero - 68),
        "Compute cloud-masked median surface reflectance composites across custom bounding boxes to quantify NDVI and SWIR NBR disturbances.",
        fill=(230, 238, 232, 255),
        font=f_sub,
    )

    # Credit tag
    draw.rectangle([W - 650, H_hero - 36, W - 80, H_hero - 10], fill=(10, 22, 16, 220))
    draw.text(
        (W - 640, H_hero - 30),
        "Real Telemetry: Tadoba / Chandrapur Corridor (19.91N, 79.41E) | ESA Sentinel-2 L2A",
        fill=(220, 230, 224, 255),
        font=f_small,
    )

    # Bottom 3-column specification bar
    draw.rectangle([0, H_hero, W, H_hero + H_bar], fill=(16, 35, 26, 255))
    draw.line([(0, H_hero), (W, H_hero)], fill=(255, 255, 255, 45), width=1)

    specs = [
        (80, "SENSOR INPUT", "ESA Sentinel-2 L2A (B02, B03, B04, B08, B11, B12 + SCL)"),
        (560, "REFERENCE DATASET", "Hansen Global Forest Change (GFC-2024-v1.12, 30 m)"),
        (1080, "SPATIAL GRID", "Auto-projected local UTM zone (10 m, 20 m, or 40 m)"),
    ]
    for x_pos, title, desc in specs:
        if x_pos > 80:
            draw.line([(x_pos - 24, H_hero + 12), (x_pos - 24, H_hero + H_bar - 12)], fill=(255, 255, 255, 40), width=1)
        draw.text((x_pos, H_hero + 16), title, fill=(255, 255, 255, 255), font=f_mono)
        draw.text((x_pos, H_hero + 40), desc, fill=(195, 210, 200, 255), font=f_small)

    final_img = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    final_img.save(out_filename, format="PNG", optimize=True)
    print(f"Saved prototype image -> {out_filename}")


def main():
    # Real forest-edge coordinates in Chandrapur / Tadoba Buffer
    bbox = (79.32, 19.84, 79.48, 19.96)
    print("Fetching real 2020 & 2024 ESA Sentinel-2 L2A composites from Planetary Computer...")
    comp_2020, err1 = build_composite(bbox, 2020, 10, 12, max_cloud=25, res=20)
    comp_2024, err2 = build_composite(bbox, 2024, 10, 12, max_cloud=25, res=20)

    if comp_2020 is None or comp_2024 is None:
        raise RuntimeError(f"Could not fetch Sentinel-2 scenes: {err1 or err2}")

    H = min(comp_2020["rgb"].shape[0], comp_2024["rgb"].shape[0])
    W = min(comp_2020["rgb"].shape[1], comp_2024["rgb"].shape[1])

    rgb_b = comp_2020["rgb"][:H, :W]
    rgb_a = comp_2024["rgb"][:H, :W]
    ndvi_b, nbr_b, ndwi_b, valid_b = (comp_2020[k][:H, :W] for k in ("ndvi", "nbr", "ndwi", "valid_mask"))
    ndvi_a, nbr_a, ndwi_a, valid_a = (comp_2024[k][:H, :W] for k in ("ndvi", "nbr", "ndwi", "valid_mask"))

    loss, _ = detect_loss(
        ndvi_b, nbr_b, ndwi_b, valid_b,
        ndvi_a, nbr_a, ndwi_a, valid_a,
        min_drop=0.18, forest_thresh=0.45,
    )

    # Enhance true-color contrast slightly so forest canopy texture is crisp
    rgb_b_uint8 = (np.clip(rgb_b * 1.25, 0.0, 1.0) * 255).astype("uint8")
    rgb_a_uint8 = (np.clip(rgb_a * 1.25, 0.0, 1.0) * 255).astype("uint8")

    # Option 1 Right Half: True-color 2024 with red canopy loss overlay
    overlay_uint8 = make_overlay(np.clip(rgb_a * 1.20, 0.0, 1.0), loss)

    # Option 2 Right Half: Spectral NDVI / SWIR visualization + red loss patches
    ndvi_norm = np.clip((ndvi_a - 0.1) / 0.65, 0.0, 1.0)
    spec_viz = np.zeros_like(rgb_a)
    spec_viz[..., 0] = (1.0 - ndvi_norm) * 0.45
    spec_viz[..., 1] = ndvi_norm * 0.78 + 0.10
    spec_viz[..., 2] = ndvi_norm * 0.35 + 0.08
    spec_viz[loss] = [0.95, 0.18, 0.10]
    spec_uint8 = (np.clip(spec_viz, 0.0, 1.0) * 255).astype("uint8")

    img_before = _crop_banner(rgb_b_uint8)
    img_after_overlay = _crop_banner(overlay_uint8)
    img_after_spectral = _crop_banner(spec_uint8)

    # Save raw before/after pair for ui.py to use directly
    img_before.save("hero_before.jpg", format="JPEG", quality=90)
    img_after_overlay.save("hero_after.jpg", format="JPEG", quality=90)
    print("Saved hero_before.jpg and hero_after.jpg for live website banner.")

    # Build Prototype 1 (True Color vs. Canopy Loss Overlay)
    _render_ui_mockup(
        img_before,
        img_after_overlay,
        left_tag=("2020 | Baseline True-Color (RGB)", "Sentinel-2 L2A surface reflectance"),
        right_tag=("2024 | Detected Canopy Loss", "dNDVI + SWIR dNBR disturbance mask"),
        out_filename="prototype_1_loss_overlay.png",
    )

    # Build Prototype 2 (True Color vs. Spectral NDVI + Loss)
    _render_ui_mockup(
        img_before,
        img_after_spectral,
        left_tag=("2020 | Optical True-Color (B04/B03/B02)", "Sentinel-2 L2A 10-20m composite"),
        right_tag=("2024 | Spectral NDVI + Loss Mask", "Multi-spectral index response"),
        out_filename="prototype_2_spectral_split.png",
    )

    # Automatically open the generated prototype images on Windows
    if hasattr(os, "startfile"):
        os.startfile("prototype_1_loss_overlay.png")
        os.startfile("prototype_2_spectral_split.png")


if __name__ == "__main__":
    main()