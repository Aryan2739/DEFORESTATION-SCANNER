"""Unit tests for analysis.py and exports.py."""

import csv
import datetime
import io
import json
import unittest
import numpy as np
from PIL import Image
from rasterio.transform import from_origin

from analysis import (
    accuracy,
    calc_area_km2,
    detect_loss,
    parse_coords_or_url,
    utm_crs,
    validate_run_params,
)
from exports import make_metrics_csv, to_geojson, to_png, to_web_jpeg_b64


class TestAnalysis(unittest.TestCase):
    def test_utm_crs_northern_and_southern(self):
        # Nagpur, India (~79.27E, 21.51N) -> UTM Zone 44N -> EPSG:32644
        self.assertEqual(utm_crs((79.18, 21.42, 79.36, 21.60)), "EPSG:32644")
        # Rondonia, Brazil (~-62.0W, -10.0S) -> UTM Zone 20S -> EPSG:32720
        self.assertEqual(utm_crs((-62.5, -10.5, -61.5, -9.5)), "EPSG:32720")

    def test_calc_area_km2(self):
        area = calc_area_km2((79.18, 21.42, 79.36, 21.60))
        self.assertGreater(area, 300.0)
        self.assertLess(area, 420.0)
        self.assertEqual(calc_area_km2((79.36, 21.60, 79.18, 21.42)), 0.0)

    def test_parse_coords_or_url(self):
        self.assertEqual(
            parse_coords_or_url("https://www.google.com/maps/@21.5123,79.2745,12z"),
            (21.5123, 79.2745),
        )
        self.assertEqual(parse_coords_or_url("19.9100, 79.4100"), (19.91, 79.41))
        self.assertIsNone(parse_coords_or_url("not a coordinate"))

    def test_validate_run_params_incomplete_2026_months(self):
        today = datetime.date(2026, 10, 15)
        bbox = (79.20, 21.45, 79.30, 21.55)
        # Oct-Dec 2026 is not finished on Oct 15, 2026
        errs = validate_run_params(bbox, 2020, 2026, 10, 12, res=20, timeline=False, today=today)
        self.assertTrue(any("not complete yet" in e for e in errs))

        # Oct-Dec 2025 is valid on Oct 15, 2026
        errs_ok = validate_run_params(bbox, 2020, 2025, 10, 12, res=20, timeline=False, today=today)
        self.assertEqual(errs_ok, [])

    def test_validate_run_params_10m_memory_limit(self):
        today = datetime.date(2026, 10, 15)
        # ~370 km2 box exceeds 150 km2 limit at 10m resolution
        large_bbox = (79.18, 21.42, 79.36, 21.60)
        errs = validate_run_params(large_bbox, 2020, 2024, 10, 12, res=10, timeline=False, today=today)
        self.assertTrue(any("exceeds the 150 km²" in e for e in errs))

    def test_detect_loss_and_strict_vs_buffered_accuracy(self):
        shape = (20, 20)
        valid = np.ones(shape, dtype=bool)
        ndwi = np.zeros(shape, dtype="float32")
        ndvi_b = np.full(shape, 0.65, dtype="float32")
        nbr_b = np.full(shape, 0.35, dtype="float32")

        ndvi_a = ndvi_b.copy()
        nbr_a = nbr_b.copy()
        # Create a 5x5 canopy loss patch in rows 5..10, cols 5..10
        ndvi_a[5:10, 5:10] = 0.25
        nbr_a[5:10, 5:10] = 0.05

        loss, baseline_canopy = detect_loss(
            ndvi_b, nbr_b, ndwi, valid, ndvi_a, nbr_a, ndwi, valid, min_drop=0.22, forest_thresh=0.50
        )
        self.assertTrue(baseline_canopy.all())
        self.assertGreater(loss[5:10, 5:10].sum(), 0)
        self.assertEqual(loss[0:4, 0:4].sum(), 0)

        # Shift truth by 1 pixel to verify buffered score > strict score
        truth_shifted = np.zeros(shape, dtype=bool)
        truth_shifted[6:11, 6:11] = True
        rgb = np.zeros((20, 20, 3), dtype="float32")

        scores, _ = accuracy(loss, truth_shifted, rgb, truth_buffered=truth_shifted)
        self.assertGreater(scores["buffered"]["f1"], scores["strict"]["f1"])

    def test_jpeg_slider_compression(self):
        arr = np.random.randint(0, 255, size=(1200, 1600, 3), dtype="uint8")
        b64 = to_web_jpeg_b64(arr, max_dim=960, quality=80)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 100)


class TestExports(unittest.TestCase):
    def test_metrics_csv_quotes_values_and_includes_run_metadata(self):
        result = {
            "facts": {"region": "Hasdeo, Chhattisgarh, India", "loss_ha": 12.5},
            "years": (2016, 2025),
            "scenes": (3, 4),
            "acc": {"strict": {"precision": 0.5}, "buffered": {"precision": 0.75}},
        }
        rows = list(csv.reader(io.StringIO(make_metrics_csv(result))))
        self.assertEqual(rows[0], ["Metric", "Value"])
        self.assertEqual(rows[1], ["region", "Hasdeo, Chhattisgarh, India"])
        self.assertIn(["baseline_year", "2016"], rows)
        self.assertIn(["target_year", "2025"], rows)
        self.assertIn(["strict_precision", "0.5"], rows)
        self.assertIn(["buffered_precision", "0.75"], rows)

    def test_png_export_is_readable_and_keeps_image_dimensions(self):
        image = np.zeros((3, 4, 3), dtype="uint8")
        encoded = to_png(image)
        with Image.open(io.BytesIO(encoded)) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.size, (4, 3))

    def test_geojson_export_contains_loss_polygon(self):
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        payload = json.loads(
            to_geojson(mask, from_origin(82.0, 23.0, 0.001, 0.001), "EPSG:4326", 0.0001)
        )
        self.assertEqual(payload["type"], "FeatureCollection")
        self.assertEqual(len(payload["features"]), 1)
        self.assertEqual(payload["features"][0]["properties"]["class"], "canopy_loss_detected")


if __name__ == "__main__":
    unittest.main()