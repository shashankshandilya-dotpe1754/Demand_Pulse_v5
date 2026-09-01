# DemandPulse

**Pan-India demand forecasting for restaurants** — an ML model that learns how
festivals, events, weather, sports and government regulations move order volume
and ticket size, city by city.

DemandPulse forecasts daily order volume and average order value for six
restaurant formats — **QSR, Fine Dining, PBCL, Casual Dining, Cloud Kitchen and
Cafe** — across **130 Indian cities**.

It combines a curated events database (236 festivals, gazetted holidays, sports
fixtures, commercial events, government orders and emergency crises, geo-scoped
to city, state, zone or pan-India) with live Open-Meteo weather, and feeds both
into a LightGBM model that has learned how each factor moves demand for each
format in each city. A Diwali weekend lifts fine dining and suppresses pubs;
heavy monsoon rain crashes dine-in while cloud kitchens spike; a liquor ban guts
PBCL and leaves QSR untouched — the model learns these relationships from
history instead of relying on hand-written multipliers.

Ships as a **Streamlit dashboard** and a **FastAPI service**, with every forecast
explained: which events drove it, by how much, and how confident the model is.

---

## Run it locally

```bash
pip install -r requirements.txt

streamlit run dashboard.py        # dashboard  → http://localhost:8501
uvicorn main:app --reload         # API + docs → http://localhost:8000/docs
```

The trained models ship in `ml/models/`, so both run out of the box. To retrain:

```bash
python -m ml.simulate             # rebuild the bootstrap training panel (~8s)
python -m ml.train                # train orders + AOV models (~3 min CPU)
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) → **New app**, pick the
   repo and set the main file to **`dashboard.py`**.
3. Deploy. `requirements.txt` and `packages.txt` (which installs `libgomp1` for
   LightGBM) are picked up automatically.

The FastAPI service in `main.py` is not part of the Streamlit deployment — host
it separately (Render, Railway, Fly.io, or any container host) if you need the
JSON API.

---

## What's inside

```
├── dashboard.py            Streamlit dashboard (main entry point)
├── pages/                  Restaurant-impact page
├── main.py                 FastAPI service: events, weather, forecasts
├── data/
│   ├── events_db.py        236 Pan-India events, 2023–2026
│   ├── india_geo.py        130 cities with zone / state / coordinates
│   └── training/           training CSVs (generated, git-ignored)
├── utils/
│   ├── restaurant_impact.py   engine router: ML first, rules as fallback
│   ├── weather_service.py     Open-Meteo historical + forecast
│   └── auto_updater.py        daily 6 AM IST events refresh
└── ml/                     the forecasting model — see ml/README.md
    ├── features.py         93 features (events, weather, calendar, geo)
    ├── simulate.py         bootstrap training-data generator
    ├── train.py            time-split LightGBM training
    ├── predict.py          inference + counterfactual explanations
    └── models/             trained boosters + validation metrics
```

## The model

Two LightGBM models — one on `log(orders_index)`, one on `log(AOV)` — trained on
93 features covering festivals by religion and region, geo-scoped local events,
sports fixtures, weather (temperature, rainfall, WMO codes, storm/heat/cold
flags), government orders (liquor bans, traffic restrictions, operating hours,
environmental and digital mandates), emergency crises, the Indian salary cycle,
festival proximity and long weekends, plus city, state, zone and tier.

Demand multipliers are no longer constants. Each is a **counterfactual run of
the model** — the same day scored with and without its events, with and without
its actual weather — so every number traces back to a learned relationship.
Individual events get their share of the day's lift by Shapley attribution,
which stays fair when a holiday, a festival and a cricket season all land
together.

| model | MAPE | R² |
|---|---|---|
| orders | **8.86%** | 0.933 |
| naive day-of-week × type baseline | 27.78% | 0.149 |
| AOV | 4.68% | 0.986 |

Validated on a strict time split (trained before 2026-04-13, scored after), so
the score measures forecasting rather than interpolation.

> **Note on training data.** The shipped models are trained on a synthetic
> bootstrap panel, not real sales history — they demonstrate a working pipeline
> and encode a reasonable prior, not evidence about real demand. Drop real
> history at `data/training/real_history.csv` and rerun `python -m ml.train`;
> the schema is in [`ml/README.md`](ml/README.md).

## Engine selection

| env var | effect |
|---|---|
| *(unset)* | ML if a trained model exists, else the legacy rule engine |
| `DEMANDPULSE_ENGINE=ml` | force ML, raise if unavailable |
| `DEMANDPULSE_ENGINE=rules` | force the legacy multiplier engine |

The rule-based multipliers are kept deliberately: they are the fallback when no
model file is present, and the A/B baseline behind `/forecast/compare`.

## API endpoints

| endpoint | what it does |
|---|---|
| `GET /forecast/day?city=Bengaluru` | one day, all six formats, live weather attached |
| `GET /forecast/range?city=Mumbai&start_date=…&end_date=…` | up to 120 days |
| `GET /forecast/compare?city=New Delhi&forecast_date=…&restaurant_type=PBCL` | ML vs rules, side by side |
| `GET /ml/model-info` | active engine, training date, holdout scores, top features |
| `GET /events/today`, `/events/range`, `/events/category/{c}` | events database |
| `GET /weather/city/{city}` | historical + 7-day forecast |
| `GET /calendar/{date}?city=…` | full day context card |
