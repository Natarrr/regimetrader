"""src.research — empirical validation tools for factor weights.

Implements Information Coefficient (IC) backtesting per:
  López de Prado (2018), Advances in Financial Machine Learning, ch. 7-8
  Grinold & Kahn (2000), Active Portfolio Management, ch. 6

The IC backtest module is a research tool, NOT a pipeline step.
It is run manually and produces an advisory report only.
It does NOT modify WEIGHTS automatically — weight changes are human decisions.
"""
