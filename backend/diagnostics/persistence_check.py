"""backend/diagnostics/persistence_check.py
Validate GJR-GARCH persistence formula and long-run variance.

Acceptance criteria (run with: python -m backend.diagnostics.persistence_check)
─────────────────────────────────────────────────────────────────────────────────
  persistence_standard  ≈ 0.9447   (alpha + beta + gamma/2)
  annual_std_percent    ≈ 14.3%    (long-run unconditional vol, units="percent")

These values match the GJR-GARCH(1,1) fit on SPY 2016-2021 (pre-COVID baseline).

Persistence formula — Engle (2003 Nobel):
  P = alpha + beta + gamma/2

  alpha  : symmetric ARCH effect (response to squared shocks)
  beta   : GARCH decay (persistence of past variance)
  gamma  : leverage asymmetry (extra weight on negative shocks)
  gamma/2: expected asymmetry contribution (50% chance of being below zero)

  Stationarity requires P < 1.
  P > 0.98 triggers CLUSTERING regime (Minsky precondition 1).

Long-run (unconditional) variance:
  sigma2_inf = omega / (1 - P_standard)

Run from repo root:
    python -m backend.diagnostics.persistence_check
"""
from __future__ import annotations

import sys
import os

# ── sys.path wiring so module runs standalone ─────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

# ── GJR-GARCH parameters (SPY 2016-2021 representative fit) ──────────────────
omega = 0.044875
alpha = 0.0
gamma = 0.201447
beta  = 0.843943

# ── Persistence formulas ──────────────────────────────────────────────────────
# Standard GJR-GARCH persistence: assumes 50% probability of negative shocks.
# This is the formula used in volatility_brain.fit_gjr_garch() and the
# persistence gauge on the Volatility Brain dashboard.
persistence_standard = alpha + beta + gamma / 2

# Alternative (full gamma): assumes every shock is negative — conservative bound.
persistence_alt = alpha + beta + gamma

# ── Long-run variance and volatility ─────────────────────────────────────────
# sigma2_inf: unconditional variance (%-pt² since model fitted on r×100)
sigma2_inf = omega / (1.0 - persistence_standard)

# daily_std: unconditional daily std in %-pt
daily_std = np.sqrt(sigma2_inf)

# annual_std_percent: annualised vol in plain percent (e.g. 14.3 = 14.3%/yr)
annual_std_percent = daily_std * np.sqrt(252)

# ── Verify via canonical util ─────────────────────────────────────────────────
from backend.utils.volatility import annualise_vol_from_condvar
annual_from_util = float(annualise_vol_from_condvar(np.array([sigma2_inf]), units="percent").iloc[0])

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  GJR-GARCH PERSISTENCE CHECK")
print(f"{'='*60}")
print(f"  Parameters:")
print(f"    omega  = {omega:.6f}")
print(f"    alpha  = {alpha:.6f}")
print(f"    gamma  = {gamma:.6f}")
print(f"    beta   = {beta:.6f}")
print()
print(f"  Persistence (alpha + beta + gamma/2) = {persistence_standard:.7f}")
print(f"  Persistence alt  (alpha + beta + gamma) = {persistence_alt:.7f}")
print(f"  Clustering regime? {persistence_standard > 0.98} (threshold 0.98)")
print()
print(f"  sigma2_inf (long-run variance) = {sigma2_inf:.6f}  %-pt^2/day")
print(f"  daily_std                      = {daily_std:.6f}  %-pt/day")
print(f"  annual_std_percent (manual)    = {annual_std_percent:.4f}  %/yr")
print(f"  annual_std_percent (via util)  = {annual_from_util:.4f}  %/yr")
print()

# ── Acceptance assertion ──────────────────────────────────────────────────────
assert abs(persistence_standard - 0.9446665) < 1e-5, (
    f"persistence_standard mismatch: {persistence_standard:.7f}"
)
assert abs(annual_std_percent - 14.3) < 0.5, (
    f"annual_std_percent out of range: {annual_std_percent:.4f}"
)
assert abs(annual_from_util - annual_std_percent) < 0.001, (
    f"Util output mismatch: {annual_from_util:.4f} vs {annual_std_percent:.4f}"
)

print(f"  [PASS] persistence_standard = {persistence_standard:.7f}  (expected ~0.9446665)")
print(f"  [PASS] annual_std_percent   = {annual_std_percent:.4f}%  (expected ~14.3%)")
print(f"  [PASS] canonical util agrees within 0.001%-pt")
print(f"{'='*60}\n")
