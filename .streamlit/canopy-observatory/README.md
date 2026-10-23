# Canopy Loss Observatory

A Streamlit geospatial workbench that detects multi-spectral **canopy loss** using cloud-masked **Sentinel-2 L2A** composites from Microsoft Planetary Computer and validates detections against **Hansen Global Forest Change (GFC)**.

## Project Structure

- `ui.py` — Streamlit interface, OpenStreetMap/Esri map selector, caveats, and compressed JPEG before/after slider.
- `analysis.py` — Sentinel-2 STAC queries, spectral index calculation (`NDVI`, `NBR`, `NDWI`), independent canopy loss detection, and Strict + Buffered Hansen GFC validation.
- `exports.py` — Image compression, animated GIF generation, GeoJSON polygonization, Markdown/CSV reporting, and local disk session persistence (`.cache/last_result.pkl`).
- `app.py` — Compatibility wrapper (`streamlit run app.py` or `streamlit run ui.py`).
- `test_analysis.py` — Unit test suite covering CRS selection, area limits, date validation, spectral detection, and strict vs. buffered accuracy metrics.

## Running the App

```bash
pip install streamlit streamlit-folium folium odc-stac pystac-client planetary-computer rasterio scipy numpy pandas pillow xarray
python -m streamlit run ui.py