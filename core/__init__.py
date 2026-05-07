"""
core/
─────
Shared domain models for the Regime-Intel reporting system.
These dataclasses are intentionally separate from the execution-layer types
in risk_manager/portfolio_state.py so that the morning-report pipeline can
be run independently of the live broker integration.
"""
