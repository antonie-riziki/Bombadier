# Bombadier – Methodology

This document describes the end‑to‑end methodology used to design, build, and evaluate Bombadier’s agronomy intelligence and the roadmap for thin‑file credit scoring. It complements the Technical Narrative.

## 1) Problem framing

- Goal: Deliver actionable agronomy insights and inclusive lending decisions for smallholder farmers using alt‑data and lightweight channels (USSD/SMS).
- Constraints: low bandwidth, sparse/irregular data, explainability and fairness requirements, fast iteration in hackathon timelines.

## 2) Data strategy

2.1 Sources (current)
- Tabular agronomy datasets in `echofarm/src/`: nutrients (N, P, K), soil pH/quality/type, environment, Crop_Type, Crop_Yield.
- Curated images for illustrative UI feedback (non‑training).

2.2 Planned sources (roadmap)
- Remote sensing (NDVI/EVI, cloud‑free mosaics), gridded weather (ERA5), soil maps (iSDA), market prices, mobile money summaries, IoT sensor telemetry.

2.3 Collection and governance
- Use standard schemas with data dictionaries; track provenance and collection dates.
- Consent and privacy: explicit opt‑in, minimal PII, encryption at rest when persisted.

## 3) Pre‑processing

- Missingness: `SimpleImputer` with mean (numeric) and most‑frequent (categorical) to maintain sample size.
- Encoding: `LabelEncoder` for categorical outputs (e.g., Soil_Type, Crop_Type) and inputs where needed.
- Scaling: `StandardScaler` for models that benefit from normalization (e.g., multinomial logistic regression recommender).
- Train/Validation split: 80/20 random split with fixed seed; future: time/region‑aware splits and k‑fold CV.

## 4) Feature engineering

- Baseline features: N, P, K, Crop_Yield, Soil_Quality, Soil_pH, Soil_Type (+ optional environment).
- Chained features: predicted outputs (Soil_Quality → Soil_pH → Soil_Type) appended to next stage inputs to reflect causal dependencies.
- Future engineered features:
  - Temporal: seasonality, lagged weather indices, rain accumulation windows.
  - Geospatial: soil class tiles, eco‑zones, elevation/slope, distance to markets.
  - Price signals: local crop price trends, dispersion, volatility.
  - Behavioral: mobile money cash‑flow summaries (cash‑in/out ratios, stability, bursts), repayment histories.

## 5) Modeling approach

5.1 Agronomy inference (current)
- Soil Quality (regression): RandomForestRegressor.
- Soil pH (regression): RandomForestRegressor.
- Soil Type (classification): RandomForestClassifier.
- Crop Type (classification): RandomForestClassifier.
- Recommended Crop (ranking): Multinomial Logistic Regression over standardized features → probabilities and top pick.

Rationale: robust to small tabular sets, non‑parametric, fast to train, interpretable via feature importances and SHAP (planned).

5.2 Credit scoring (roadmap)
- Objective: Probability of Default (PD) model with calibrated scores; derive limits and pricing policies.
- Candidate models: LightGBM/Gradient Boosting with monotone constraints where sensible; logistic regression benchmark for transparency.
- Targets & labels: binary default within horizon (e.g., 6–12 months) when repayment data is available; proxy labels during pilot if needed.
- Post‑processing: score calibration (Platt/Isotonic), policy mapping (score → limit, APR, tenor), cut‑offs tuned to portfolio objectives.

## 6) Evaluation

6.1 Agronomy
- Regression: MAE, RMSE, R²; residual diagnostics and error vs. feature ranges.
- Classification: Accuracy, F1, ROC‑AUC, class balance checks; top‑k accuracy for crop recommendation.

6.2 Scoring (future)
- Discrimination: ROC‑AUC, PR‑AUC, KS.
- Calibration: Brier score, calibration curves, expected/observed default rates by decile.
- Stability/shift: PSI/CSI across regions and seasons.

6.3 Robustness
- Region/crop stratified performance; time‑split validation; OOD detection heuristics for unusual feature combinations.

## 7) Explainability and fairness

- Explainability: SHAP values (global and per‑prediction) for tabular models; reason codes surfaced in USSD/SMS (score drivers).
- Gemini prompts provide human‑readable summaries (kept short, localized to Kenya as default) with scope guardrails.
- Fairness diagnostics: compare metrics by sensitive proxies when available (region/coop/gender if collected); mitigation via re‑weighting, thresholding, or constrained optimization.

## 8) Human‑in‑the‑loop

- For edge cases and OOD flags, route to loan officer review with suggested reason codes and data snapshots.
- Feedback loop: officer adjudications stored for continuous model improvement and policy tuning.

## 9) Deployment methodology

- Channel split: Streamlit for exploration/training demonstrations and farmer UX; Flask USSD for production‑like thin‑pipe interactions.
- Secrets via `.env`; call external APIs (Gemini, Africa’s Talking, PayHero) over TLS.
- Future ops: containerize, add CI for lint/tests, model registry (artifact checksums), and blue/green deployments for model swaps.

## 10) Experiment design

- A/B within USSD flows for guidance phrasing and engagement (opt‑in SMS depth vs. short tips).
- Offline/low‑connectivity simulations: fail/timeout injection on SMS/Payments to validate retries and user messaging.
- Sensitivity: perturb key features (±10–20%) to observe stability of recommendations and score monotonicity.

## 11) Limitations and mitigations

- Data sparsity: bootstrap with conservative models, uncertainty flags, and expand data partnerships (sensing/market/behavioral).
- Label quality: introduce human review and weak‑label approaches during cold start; tighten as repayment data accrues.
- Bias risk: proactive diagnostics and governance; publish reason codes and simple policies.
- Generative safety: narrow prompts, temperature control, and rate limits.

## 12) Roadmap checkpoints

1. Add PD model + score API; wire USSD eligibility and score views with reason codes.
2. Portfolio stress engine (weather/price) with policy levers and loss curves.
3. Persistence layer (Postgres) for profiles, telemetry, scores, and outcomes; encryption and access control.
4. SHAP dashboards and fairness reports per release.
5. Offline kiosk mode for batch scoring and deferred sync.

---

This methodology prioritizes farmer value (clear, quantified guidance) while building the scaffolding for transparent, bias‑aware lending. The staged approach lets us deliver utility today and layer rigorous underwriting as data depth grows.


