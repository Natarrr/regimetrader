# Regime Trader: Institutional Alpha Engine

Regime Trader is an institutional-grade quantitative dashboard designed to bridge academic finance theory with real-time market execution. Rooted in Nobel-prize winning frameworks from Fama, Engle, and Stiglitz, the platform delivers a multi-layered signal engine centered on risk-aware alpha discovery.

### Core Capabilities

* **Regime Detection Engine:** Utilizes Hidden Markov Models (HMM) and Machine Learning to classify market volatility states—from Euphoria to Crash—by integrating VIX dynamics and credit stress signals (HY/IG spreads).
* **Smart Money Discovery:** Extracts high-conviction signals from information asymmetry, specifically tracking CEO/CFO open-market purchases via SEC EDGAR and institutional 13F accumulation.
* **Macro Synthesis:** Real-time monitoring of commodity term structures (Backwardation/Contango) and COT proxies to identify global macro shocks.
* **NLP Intelligence:** Leverages Claude LLM for automated earnings analysis, cross-referencing qualitative sentiment with quantitative data citations.

### Industrial-Grade Architecture

Built for reliability, the system features a robust backend with rate-limited data services, atomic state management, and an exhaustive CI/CD suite ensuring 100% logic integrity. An intuitive Streamlit dashboard provides a unified command center for live monitoring, macro intelligence, and portfolio reconciliation, empowering analysts to identify high-conviction opportunities within a rigorous statistical context.
