"""Tests for ODRA / IQM evaluation (shot noise, transpile, readout, optional hardware).

Run locally (no token): ``uv run pytest`` from the repo root.

Run including IQM (token must be visible to the **pytest process**, same shell):

``IQM_TOKEN='your-token' uv run pytest tests/ansatz_odra_evaluation/odra_test -m iqm``

Optional: ``IQM_URL`` (default ``https://odra5.e-science.pl/``). Use ``pytest -rs`` to print skip reasons.
"""
