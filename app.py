"""Entrypoint wrapper so both `streamlit run ui.py` and `streamlit run app.py` work."""

from ui import main

if __name__ == "__main__":
    main()