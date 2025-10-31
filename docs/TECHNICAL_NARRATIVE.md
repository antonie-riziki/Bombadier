# Bombadier – Technical Narrative

This document details the system architecture, data/modeling approach, interfaces, integrations, and the future roadmap for Bombadier: an AI‑native agri‑finance platform designed for thin‑file smallholder farmers.

## 1) System overview

- Channels
  - Web (Streamlit): agronomy analytics, soil/crop ML inferences, expert guidance, onboarding.
  - USSD/SMS (Flask): low‑bandwidth assistant for insights, payments (STK), and loan flows.
- Intelligence
  - Tabular ML models for soil quality, pH, soil type, and crop type.
  - Generative AI (Gemini) for short, localized agronomy guidance and data explanations.
- Integrations
  - Africa’s Talking: SMS delivery.
  - PayHero: M‑Pesa STK push for IoT kits and services.
- Data
  - Curated agronomy datasets in `echofarm/src/`.
  - Trained models in `echofarm/model/`.

## 2) Repository layout (high‑level)

- `echofarm/`
  - `app.py`: Streamlit navigation and shell.
  - `pages/`: feature pages – `soilwise.py`, `main.py`, `chatbot.py`, `registration.py`, optional CV/irrigation stubs.
  - `model/`: serialized scikit‑learn models (`*.pkl`).
  - `src/`: tabular datasets (CSV/Parquet).
- `ussd_app/`
  - `ussd.py`: Flask app exposing `/ussd` menu tree.
  - `send_sms.py`: Africa’s Talking client.
  - `payment.py`: PayHero STK integration.
- `theme_docs/`: hackathon briefs and concept.
- `docs/`: this narrative and method notes.

## 3) Data and features

Primary tabular features come from agronomy datasets:

- Nutrients: Nitrogen (N), Phosphorus (P), Potassium (K).
- Soil and environment: pH, Soil_Type, Soil_Quality, Temperature, Humidity, Wind_Speed.
- Outcomes/context: Crop_Type, Crop_Yield, Date, Country/Region (where available).

Data preparation:

- Missing values: `SimpleImputer` (mode for categorical, mean for numerical) in `pages/main.py`.
- Label encoding for categorical columns (e.g., Soil_Type).
- For recommendations, features are standardized (e.g., `StandardScaler`) before logistic regression.

Future data extensions:

- Remote sensing signals (e.g., NDVI/EVI), gridded weather (ERA5), soil maps (iSDA), market prices, mobile money and IoT sensor streams.

## 4) Modeling stack

Trained scikit‑learn models persisted via `joblib` (see `echofarm/model/`):

- Soil Quality (regression): RandomForestRegressor on [N, P, K, Crop_Yield, …].
- Soil pH (regression): RandomForestRegressor.
- Soil Type (classification): RandomForestClassifier on [N, P, K, Crop_Yield, Soil_Quality, Soil_pH].
- Crop Type (classification): RandomForestClassifier on [N, P, K, Crop_Yield, Soil_Quality, Soil_pH, Soil_Type].

Runtime inference chain (in `soilwise.py`):

1. Predict Soil_Quality from (N, P, K, Crop_Yield); append to feature vector.
2. Predict Soil_pH; append.
3. Predict Soil_Type; append.
4. Predict Crop_Type; render guidance and visuals.

Complementary recommender (in `ct_model.py`):

- Multinomial Logistic Regression over standardized features to produce a “recommended crop” and probabilities.

Explainability and guidance:

- Gemini 2.0 Flash used for: agronomy best practices (ploughing, sowing, irrigation, IPM, harvest, storage), intercropping/crop rotation lists, and human‑readable explanations of grouped analytics (“The Explainer”). System prompts constrain scope and tone to concise, localized advice.

Roadmap – credit risk model:

- A LightGBM/Gradient Boosting model for PD (probability of default) using behavioral + agronomy features, with monotone constraints where sensible, SHAP explainability, fairness diagnostics, and reason codes for USSD/SMS. Outputs: score, limit, pricing, and eligibility flags.

## 5) Interfaces

### 5.1 Streamlit app (EchoFarm)

- `pages/soilwise.py`: self‑service soil test (inputs: N, P, K, Crop_Yield) → sequential predictions → localized recommendations + random visual exemplars (crop/soil image galleries). Optional SMS share via Africa’s Talking.
- `pages/main.py`: dataset exploration, imputation preview, categorical/numerical EDA, grouped views (country/region), and Gemini “Explainer”.
- `pages/chatbot.py`: scoped agri chatbot (Gemini) with short, practical answers.
- `pages/registration.py`: basic onboarding and welcome SMS.

### 5.2 USSD/SMS app

- `ussd_app/ussd.py` exposes `/ussd` with menus:
  - IOT Kit Services: purchase (STK push), dashboard options that trigger SMS advisories.
  - Loan Services: scaffold for apply/eligibility/limit/repayment/insurance/credit score.
  - Report/Support and About stubs.
- SMS sending (`send_sms.py`): Africa’s Talking SDK with sender short code.
- Payments (`payment.py`): PayHero v2 payments endpoint; configurable `Authorization` header. STK push confirmation/receipt is printed/logged.

### 5.3 API keys and configuration

- `.env` variables: `GOOGLE_API_KEY`, `AT_API_KEY`, `PAYHERO_USERNAME`, `PAYHERO_PASSWORD`, `PAYHERO_AUTH`.
- Keys are loaded via `python-dotenv`; never hardcoded.

## 6) Security, privacy, and safety

- Secrets: loaded from environment; exclude `.env` from VCS.
- PII: phone numbers are used for SMS/USSD; future DB persistence must encrypt at rest and restrict access (role‑based). Add consent prompts and clear ToS.
- Model safety: Gemini prompts restrict scope to agriculture; responses are short and factual. For lending, enforce explainability and adverse action notices (reason codes) to meet regulatory expectations.
- Payments: call PayHero over HTTPS; validate callbacks and log transactions.

## 7) Observability and QA

- Logging: add structured logs around USSD requests, SMS sends, and payment posts.
- Monitoring: expose basic health checks for Flask; Streamlit runtime logs for inference errors.
- Evaluation: retain hold‑outs for ML models; track MAE/R2 for regression and F1/ROC‑AUC for classification; add cross‑region robustness tests.

## 8) Deployment and ops

- Local development
  - Streamlit: `cd echofarm && streamlit run app.py`.
  - USSD: `cd ussd_app && python ussd.py` (default port 8000). Use ngrok to expose `/ussd` to aggregator.
- Production suggestions
  - Containerize both services; run behind a reverse proxy (TLS). Configure autoscaling for USSD bursts.
  - Externalize model artifacts via object storage and pin with checksums.
  - Add a managed DB (Postgres) for profiles, sensor readings, scores, and loan states.

## 9) Roadmap

1. Credit Scoring MVP
   - LightGBM PD model; calibrated scores → limits and pricing.
   - Integrate with USSD: `2*2` (eligibility/limit) and `2*7` (credit score + reason codes).
2. Portfolio & Stress
   - Scenario driver (weather/price/FX); aggregate PD/LGD/EAD → loss/ROI curves; simple policy levers (tenor, rate, collateral).
3. Data Backbone
   - Farmer profiles, sensor ingestion, mobile money summaries; consent + privacy controls.
4. Fairness & Explainability
   - Group metrics (by region/crop/gender where available); SHAP reason codes in SMS; audit logs.
5. Offline/Edge
   - Optional on‑device scoring for kiosks; periodic sync to backend.

## 10) Risks and mitigations

- Data sparsity / shift: use robust models, uncertainty flags, and human‑in‑the‑loop overrides.
- Bias: monitor group performance, mitigate via reweighting/constraints, and publish reason codes.
- Connectivity: USSD/SMS default; cache and retry strategies for payments and messaging.
- Safety: scope prompts, rate‑limit generative calls, and sanitize user inputs.

---

Bombadier’s core advantage is the coupling of actionable agronomy with inclusive credit delivery. By starting with farmer value (better yields) and building explainable underwriting on top, we unlock sustainable unit economics for lenders and resilience for smallholders.


