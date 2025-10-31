# Bombadier

AI-native agri-finance for smallholder farmers. Bombadier combines alternative agronomy data (soil N-P-K, pH, crop/yield signals), explainable AI, and last‑mile channels (USSD/SMS) to unlock fair credit, better agronomic decisions, and resilient portfolios.

## Why now
- Emerging alt‑data (remote sensing, low-cost IoT) enables thin‑file underwriting at scale.
- Lenders need explainable, bias-aware scoring; farmers need offline-first access and clear advice.
- Our wedge: actionable agronomy + inclusive credit in one lightweight product.

## What we’ve built
- USSD + SMS assistant: kit purchase (STK push), agronomy tips, and end-to-end loan menu scaffold (eligibility, limits, repayment, insurance).
- Agronomy AI (Streamlit app):
  - Soil quality, pH, soil type, and crop type prediction using trained models (`echofarm/model/*.pkl`).
  - Gemini-powered guidance: intercropping, rotations, and data explanations for farmer trust.
- Payments and messaging: PayHero STK and Africa’s Talking SMS integrated.

## Why it wins
- Inclusive distribution: works on feature phones; no app install, low bandwidth.
- Measurable outcomes: higher yields (better crop fit/inputs) and lower risk (data-driven lending).
- Explainability by design: reasoned guidance and transparent analytics increase adoption and regulator comfort.

## What’s next
- Credit scoring (LightGBM) with reason codes and bias audits; expose score/limit in USSD.
- Portfolio stress engine (weather/price shocks) for loss forecasting and capital efficiency.
- Data backbone: farmer profile, sensor streams, loan/repayment history for longitudinal modeling.

## Repo layout
- `echofarm/`: Streamlit app (agronomy AI, analytics, chatbot, registration)
- `ussd_app/`: Flask USSD service (menus, STK, SMS)
- `echofarm/model/`: trained models (`sqm.pkl`, `sph.pkl`, `styp.pkl`, `ctyp.pkl`)
- `echofarm/src/`: datasets
- `theme_docs/`: hackathon briefs and concept docs

## Quick setup
Prereqs: Python 3.10+, pip, virtualenv; Africa’s Talking, PayHero, and Google Gemini API keys.

1) Environment
```
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # if not present, install streamlit, flask, requests, google-generativeai, africastalking, scikit-learn, joblib, python-dotenv
```

2) Secrets (.env)
```
GOOGLE_API_KEY=...
AT_API_KEY=...
PAYHERO_USERNAME=...
PAYHERO_PASSWORD=...
PAYHERO_AUTH=Basic ...   # or bearer per your account
```

3) Run agronomy app
```
cd echofarm
streamlit run app.py
```

4) Run USSD service (port 8000)
```
cd ussd_app
python ussd.py
```

Notes
- Place model files in `echofarm/model/` (already included) and datasets in `echofarm/src/`.
- Expose `/ussd` publicly via an HTTPS tunnel (e.g., ngrok) to connect your aggregator.
