# BILAN DE PRÉ-LANCEMENT — regime_trader

*Revue de readiness cadrée selon les pratiques des professionnels (pipeline quant
en 4 étapes : alpha research → backtest → paper trade → live). Système **signal-only
/ advisory** : il génère une liste d'achats classée + un vecteur de poids MVO et
l'émet sur Discord ; **l'exécution est humaine** (aucune intégration broker — `alpaca`
est dans requirements mais jamais câblé). « Lancer » = émettre les signaux en live.*

---

## 1. La stratégie (cadrée comme un pro)

- **Nature** : système **systématique, multi-facteurs, multi-régions, régime-aware**,
  à edge **alt-data** (insider / congress / 13F) — différenciateur rare.
- **Univers** : ~243 tickers US + EU + Asia, écran de liquidité (ADV) sur le sleeve SMID.
- **Familles de facteurs** : value/quality (Piotroski, FCF yield, P/B, ROIC, +E/P/EV-EBITDA/growth/accruals candidats), momentum/consensus (12-1m, révisions, price-target), alt-data (insider, congress, 13F).
- **Construction** : scoring pondéré **cross-sectionnel par groupe-pair géographique**
  (anti-contamination), **gate Piotroski**, **overlay VIX** (NORMAL/BEAR/CRASH),
  **MVO Ledoit-Wolf** pour le sizing, et 3 gates de sélection : extension (already-moved),
  **target-passed** (pas d'upside consensus), **capitulation low-beta**.
- **Cadre théorique** : Grinold-Kahn (IR = IC·√BR), Fama-French, Jegadeesh-Titman,
  Piotroski, Sloan, Amihud, López de Prado (anti-fuite).

## 2. Alignement avec le pipeline pro (4 étapes)

| Étape pro | État | Commentaire |
|---|---|---|
| **Alpha research** | ✅ | Librairie de facteurs ancrée académiquement ; outils IC (`ic_metrics`, de-overlap embargo). |
| **Backtest** | ⚠️ partiel | `backtest_signals` existe et est **désormais net de coûts** (P2.2). Manque : significativité statistique (permutation / walk-forward), IC mesuré sur les nouveaux facteurs (peu d'historique de snapshots). |
| **Paper trade** | ❌ **non fait** | **Étape manquante critique.** Les pros : 30-60 jours / **50+ signaux** forward-testés avant capital réel, et comparaison fills réels vs hypothèses de coût. |
| **Live** | ⚙️ signal-only | « Live » = émettre les signaux. L'exécution est manuelle ⇒ pas de risque d'auto-blow-up, mais dépend de la **discipline humaine** (tenir les signaux à −15/−20 %). |

## 2bis. Résultats du backtest net-de-coûts (archive existante, 2026-06-19)

Backtest sur **23 snapshots** (28 mai → 11 juin 2026), **net** des coûts P2.2,
horizon T+10, vs SPY :

| Tier | n | Win rate | Return moy (net) | Alpha vs SPY | Profit factor |
|---|---|---|---|---|---|
| TACTICAL BUY | 162 | 47.5 % | −0.02 % | **+1.00 %** | 0.99 |
| HIGH BUY | 2 | 0 % | −6.83 % | −5.34 % | 0.00 |
| Large caps | 164 | 47.0 % | −0.11 % | +0.90 % | — |

Pire détracteur récurrent : **9984.T (SoftBank)**, `momentum_long=1.00` → reversion
~ −17 % → **valide le risque momentum-reversion** que les gates extension/target-passed adressent.

**Caveats critiques (à lire impérativement) :**

1. **Échantillon minuscule (~3 semaines, 164 trades pricés) ⇒ NON significatif
   statistiquement.** Pas un edge prouvé : alpha faiblement positif (+1 %) dans le
   bruit, win rate < 50 %, profit factor ≈ 1.
2. **Signaux PRÉ-améliorations.** Ces snapshots ont été générés **avant** les gates
   (target-passed, extension, capitulation low-beta) et les correctifs d'audit (cov
   MVO, beta, devise). Les gates auraient filtré une partie des pertes (déjà
   au-delà du target / already-moved) ⇒ **baseline pré-gate**, pas le système actuel.
3. **Poids hétérogènes** sur la période (config évolutive) — l'ère ancienne
   `v2 (28/23/22/15/12)` ressort à 65.7 % WR / +1.84 % alpha mais sur 35 trades seulement.

**Ce backtest CONFIRME le verdict :** edge historique thin et non significatif ⇒
**NO-GO capital réel**, **GO paper / signaux** pour accumuler de la breadth et
valider les gates en forward.

## 3. Forces (ce qui est solide)

- **Robustesse logicielle** : 1477 tests verts, architecture isolée, `FMPEndpointError`
  (pas de `try/except` silencieux sur les pannes de route).
- **Safety-first** : kill-switch / overlay VIX, **circuit-breaker schema** (refuse d'écrire
  si <40 % de l'univers est complet), **régime CAPITULATION** (bascule vers anchors qualité,
  filtre low-beta **désormais actif** après l'audit).
- **Intégrité quant** : anti-look-ahead (`filingDate`), orthogonalité **surveillée** en
  permanence, IC **de-overlappé** (anti-fuite López de Prado), poids `sum==1.0` assertés.
- **Audit de cohérence fait + corrigé** : covariance MVO assainie (Ledoit-Wolf), beta gate
  activé, gate target-passed (plus de « BUY » sans upside), coûts au backtest, devise pairée.
- **Données absentes gérées** : `None` (unavailable) ≠ `0.0` (dead) — l'absence n'est jamais
  lue comme baissière (sauf design Piotroski assumé).

## 4. Risques & angles morts (à connaître AVANT de lancer du capital)

1. **Pas de paper trading** — aucune validation out-of-sample en conditions réelles. **Gap pro n°1.**
2. **Significativité non établie** — historique de snapshots court ; IC des nouveaux facteurs
   (candidats, beta) pas encore mesuré. Risque de sur-ajustement aux priors académiques.
3. **Edge structurellement retardé** (mémoire `freshness_extension`) — le système **confirme**
   les tendances, il ne **prédit** pas (momentum 12-1m, insider Form-4, 13F ~45j). Les gates
   extension/target-passed atténuent le « chase », mais le **timing reste un risque**.
4. **Sizing** — MVO produit des poids mais (a) **pas d'exécution**, (b) la covariance a peu
   d'historique ⇒ fallback fréquent ; pas de Kelly/vol-target **au niveau book** (seul le
   vol-target VIX existe). Pas de modèle d'**impact marché** pour les small-caps illiquides.
5. **Deux moteurs** (v2.2 LIVE / v3 SHADOW) + **2 flags OFF non validés** (Piotroski-missing,
   composite neutralize) — décision de stratégie en attente du backtest IC.
6. **Dépendance FMP** — source unique du hot-path ⇒ risque de panne mono-source (atténué par
   le circuit-breaker, mais à surveiller).
7. **Discipline psychologique** — exécution humaine ⇒ le risque n°1 devient le **biais de
   l'opérateur** (sauter des signaux, override en drawdown).

## 5. Checklist Go / No-Go (recommandation pro)

**Verdict :**
- **GO** pour lancer le **pipeline de signaux** (Discord / advisory) — faible risque, pas d'exécution auto.
- **NO-GO** pour engager **du capital réel** tant que les étapes ci-dessous ne sont pas faites
  (ne pas sauter l'étape *paper* : c'est la cause n°1 de blow-up selon les pros).

Avant capital réel :
- [ ] **Paper trade 30-60 j / 50+ signaux** ; comparer fills réels vs coûts P2.2 (20/40/60 bps).
- [ ] **Backtest significatif** : permutation test + walk-forward + IC de-overlappé une fois assez de snapshots.
- [ ] **Limites de risque écrites** : max position (10 % ✅ existe), max secteur (30 % ✅), **stop drawdown** au niveau book (à définir), taille de book / capacité ADV.
- [ ] **Monitoring live** : dashboards IC, `factor_orthogonality`, `fmp_health`, alertes de dérive.
- [ ] **Trancher les flags** (Piotroski-missing, composite) **sur évidence IC**, pas a priori.
- [ ] **Plan de bascule v2.2 → v3** (la cible de migration).
- [ ] **Règle de discipline** : suivre les signaux y compris à −15/−20 % (sinon l'edge systématique est annulé).

## 6. Conclusion

Le système est **prêt à émettre des signaux en live aujourd'hui** (advisory, faible risque).
Pour en faire un **outil de trading sur capital réel**, il manque l'étape que **tous les pros
imposent** : un **paper trading** documenté + une **validation statistique** des facteurs.
La bonne nouvelle : l'outillage existe déjà (`backtest_signals` net de coûts, `ic_metrics`
de-overlappé) — il faut surtout **du temps de marché** (accumuler snapshots + paper trades),
pas du code.

---

### Sources (pratiques pro)
- [The Quant's Checklist Before Entering Any Trade — Y. Akbay (Medium)](https://medium.com/@yavuzakbay/the-quants-checklist-before-entering-any-trade-3f4d97172af1)
- [An Engineer's Guide to Building & Validating Quant Strategies — Y.K. Chia (Medium)](https://extremelysunnyyk.medium.com/an-engineers-guide-to-building-and-validating-quantitative-trading-strategies-b4611e5e2ac5)
- [Risk Before Returns: Position Sizing Frameworks (Fixed-Fractional, ATR, Kelly-Lite) — I. Veliu (Medium)](https://medium.com/@ildiveliu/risk-before-returns-position-sizing-frameworks-fixed-fractional-atr-based-kelly-lite-4513f770a82a)
- [Trading Risk Management: Position Sizing, Drawdowns & Capital Protection — QuantVPS](https://www.quantvps.com/blog/trading-risk-management)
- [Harnessing Multi-Factor Strategies Close to the Core — S&P Dow Jones Indices](https://www.spglobal.com/spdji/en/research/article/harnessing-multi-factor-strategies-close-to-the-core/)
