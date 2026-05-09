"""streamlit_app.py — entry-point shim.

Delegates to regime_trader/ui/streamlit_app.py.
Calls main() explicitly so Streamlit's re-run model re-renders on every interaction.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

from regime_trader.ui.streamlit_app import main

main()
