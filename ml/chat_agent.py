"""
ml/chat_agent.py
================
"Ask DemandPulse" — a merchant-facing chat layer over the forecasting model.

THE ONE RULE
------------
The LLM does not forecast. It cannot do arithmetic on demand, cannot estimate a
percentage, cannot guess what Diwali does to a pub. It has exactly four tools —
all of which call the trained LightGBM model or the events database — and its
job is to pick the right one, then explain what came back in plain language.

Everything numeric in an answer traces to a `predict_impact()` call. If the
model has no answer, the correct response is to say so, not to improvise one.

Usage
-----
    from ml.chat_agent import answer
    result = answer("Should I add staff in Bengaluru this weekend?")
    print(result["text"])          # narrated answer
    print(result["tool_trace"])    # exactly which forecasts backed it

The Streamlit page (pages/02_Ask_DemandPulse.py) wraps this with chat history.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from data.india_geo import CITIES, CITY_LIST
from ml.config import RESTAURANT_TYPES
from ml.llm import LLMUnavailable, available, chat

MAX_TOOL_ROUNDS = 5

SYSTEM = """You are DemandPulse, a demand-forecasting assistant for Indian restaurant operators.

CRITICAL RULE: you never produce a demand number, percentage or forecast yourself.
Every figure you state must come from a tool result in this conversation. If a tool
did not give you a number, say you do not have it — do not estimate, interpolate or
reason your way to one. You may compute trivial arithmetic on returned values (a
sum, a difference between two returned figures) but nothing more.

You have the trained model behind these tools. Use them freely — call several if the
question spans days, cities or restaurant formats.

How to answer:
- Lead with the decision the operator has to make, not the data. "Yes, add staff
  Saturday evening" beats "the index is 184.2".
- Quote the forecast index as a percentage change versus a normal day, which is what
  it means, and give the 80% range when a decision depends on the downside.
- Name the drivers the tool returned (which festival, which weather) rather than
  speaking generally about seasonality.
- Keep it to a short paragraph or a few bullets. These are busy people.
- If a question is outside what the tools cover — competitor data, individual outlet
  history, menu pricing — say so plainly and suggest what you can answer instead.

Context you can rely on:
- The forecast index is scaled so 100 = a typical quiet weekday for that city and
  restaurant format. 150 means about 50% more orders than such a day.
- Restaurant formats: QSR, Fine Dining, PBCL (pub/bar/cafe/lounge), Casual Dining,
  Cloud Kitchen, Cafe.
- The models are currently trained on a synthetic bootstrap panel, not real sales
  history. If someone asks how accurate this is, say so honestly and mention the
  holdout MAPE from get_model_info.
- Today's date is {today}.
"""

TOOLS = [
    {
        "name": "get_forecast",
        "description": ("Forecast for ONE city on ONE date. Returns predicted index "
                        "(100 = typical quiet weekday), percent change, 80% prediction "
                        "interval, active events with each one's contribution, weather "
                        "effect, predicted average order value, and confidence. Call it "
                        "once per restaurant format you need."),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": f"One of the supported cities, e.g. {CITY_LIST[:5]}"},
                "forecast_date": {"type": "string", "description": "YYYY-MM-DD"},
                "restaurant_type": {"type": "string", "enum": RESTAURANT_TYPES,
                                    "description": "Omit to get all six formats"},
            },
            "required": ["city", "forecast_date"],
        },
    },
    {
        "name": "get_forecast_range",
        "description": ("Forecast every day between two dates for one city. Use this for "
                        "'next week', 'this month', 'when is my busiest day' questions. "
                        "Max 60 days."),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "restaurant_type": {"type": "string", "enum": RESTAURANT_TYPES},
            },
            "required": ["city", "start_date", "end_date"],
        },
    },
    {
        "name": "list_events",
        "description": ("Events active in a date range for a city — festivals, holidays, "
                        "sports, government orders, crises. Use when asked WHAT is "
                        "happening rather than how much demand changes."),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "city": {"type": "string"},
                "category": {"type": "string",
                             "description": "Optional filter, e.g. Festival, Government Order"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_model_info",
        "description": ("How the live model was trained and how accurate it is on held-out "
                        "data. Use for questions about trust, accuracy or methodology."),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ── Tool implementations (all call the real model) ────────────────────────────
def _slim(p: dict) -> dict:
    """Trim a prediction to what the LLM needs — keeps context small and focused."""
    return {
        "date": p["date"], "weekday": p["weekday"], "city": p["city"],
        "restaurant_type": p["restaurant_type"],
        "predicted_index": p["predicted_index"],
        "pct_change_vs_normal_day": p["pct_change"],
        "interval_80": p.get("prediction_interval_80"),
        "signal": p["signal"], "confidence": p["confidence"],
        "predicted_aov_rupees": p.get("predicted_aov"),
        "event_effect_pct": round((p["event_multiplier"] - 1) * 100, 1),
        "weather_effect_pct": round((p["weather_multiplier"] - 1) * 100, 1),
        "weather_note": p.get("weather_note"),
        "events": p.get("factors", [])[:6],
        "engine": p.get("engine"),
    }


def _resolve_city(city: str) -> str:
    if city in CITIES:
        return city
    lower = {c.lower(): c for c in CITY_LIST}
    if city.lower() in lower:
        return lower[city.lower()]
    hits = [c for c in CITY_LIST if city.lower() in c.lower()]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"Unknown city {city!r}. Supported cities include: {CITY_LIST[:12]} …")


def tool_get_forecast(city: str, forecast_date: str, restaurant_type: str = None) -> dict:
    from utils.restaurant_impact import predict_impact
    from utils.weather_service import get_forecast_weather, get_historical_weather

    city = _resolve_city(city)
    d = date.fromisoformat(forecast_date)
    geo = CITIES[city]
    wx = {}
    try:
        rows = (get_forecast_weather(geo["lat"], geo["lon"], days=8)
                if d >= date.today() else
                get_historical_weather(geo["lat"], geo["lon"], forecast_date, forecast_date))
        for r in rows:
            if r.get("date") == forecast_date and "error" not in r:
                wx = r
    except Exception:
        pass
    types = [restaurant_type] if restaurant_type else RESTAURANT_TYPES
    out = [
        _slim(predict_impact(city, d, t,
                             temp_max=wx.get("temp_max_c"),
                             precip_mm=wx.get("precipitation_mm"),
                             weather_code=wx.get("weather_code"),
                             temp_min=wx.get("temp_min_c")))
        for t in types
    ]
    return {"city": city, "date": forecast_date,
            "weather_used": bool(wx), "forecasts": out}


def tool_get_forecast_range(city: str, start_date: str, end_date: str,
                            restaurant_type: str = None) -> dict:
    from utils.restaurant_impact import predict_date_range

    city = _resolve_city(city)
    s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if (e - s).days > 60:
        e = s + timedelta(days=60)
    res = predict_date_range(city, s, e)
    types = [restaurant_type] if restaurant_type else RESTAURANT_TYPES
    trimmed = {
        t: [{"date": p["date"], "weekday": p["weekday"][:3],
             "predicted_index": p["predicted_index"],
             "pct_change_vs_normal_day": p["pct_change"],
             "signal": p["signal"],
             "top_event": (p.get("factors") or [None])[0]}
            for p in res.get(t, [])]
        for t in types if t in res
    }
    return {"city": city, "start_date": start_date, "end_date": e.isoformat(),
            "series": trimmed}


def tool_list_events(start_date: str, end_date: str, city: str = None,
                     category: str = None) -> dict:
    from data.events_db import get_events_by_range

    city = _resolve_city(city) if city else None
    evs = get_events_by_range(start_date, end_date, category=category, city=city)
    return {"count": len(evs), "events": [
        {"name": e["name"], "category": e["category"],
         "subcategory": e.get("subcategory"), "start_date": e["start_date"],
         "end_date": e["end_date"], "scope": e.get("scope"),
         "expected_direction": e.get("impact_on_demand"),
         "why": e.get("description")}
        for e in evs[:40]
    ]}


def tool_get_model_info() -> dict:
    from utils.restaurant_impact import engine_status

    return engine_status()


DISPATCH = {
    "get_forecast": tool_get_forecast,
    "get_forecast_range": tool_get_forecast_range,
    "list_events": tool_list_events,
    "get_model_info": lambda **kw: tool_get_model_info(),
}


# ── Agent loop ────────────────────────────────────────────────────────────────
def answer(question: str, history: list = None) -> dict:
    """Answer one question, running tool calls until the model is done."""
    if not available():
        raise LLMUnavailable(
            "Chat needs an LLM key — set ANTHROPIC_API_KEY or OPENAI_API_KEY in the "
            "environment, or in Streamlit secrets when deployed."
        )

    messages = list(history or []) + [{"role": "user", "content": question}]
    system = SYSTEM.format(today=date.today().isoformat())
    trace = []

    for _ in range(MAX_TOOL_ROUNDS):
        resp = chat(messages, system=system, tools=TOOLS)
        if not resp["tool_calls"]:
            return {"text": resp["text"], "tool_trace": trace, "messages": messages}

        # Record the assistant turn, then execute every requested tool.
        messages.append({"role": "assistant", "content": _assistant_block(resp)})
        results = []
        for call in resp["tool_calls"]:
            fn = DISPATCH.get(call["name"])
            try:
                out = fn(**call["input"]) if fn else {"error": f"no such tool {call['name']}"}
                err = None
            except Exception as exc:                                   # noqa: BLE001
                out, err = {"error": str(exc)}, str(exc)
            trace.append({"tool": call["name"], "input": call["input"], "error": err})
            results.append((call, out))
        messages.append({"role": "user", "content": _result_block(results)})

    return {"text": "I could not finish that lookup — try narrowing the question to "
                    "one city and a shorter date range.",
            "tool_trace": trace, "messages": messages}


def _assistant_block(resp):
    """Provider-shaped assistant turn carrying the tool calls."""
    from ml.llm import provider

    if provider() == "anthropic":
        blocks = ([{"type": "text", "text": resp["text"]}] if resp["text"] else []) + [
            {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
            for c in resp["tool_calls"]
        ]
        return blocks
    return resp["text"] or ""


def _result_block(results):
    from ml.llm import provider

    if provider() == "anthropic":
        return [{"type": "tool_result", "tool_use_id": c["id"],
                 "content": json.dumps(out, default=str)[:12000]}
                for c, out in results]
    return "TOOL RESULTS:\n" + "\n".join(
        f"{c['name']}({json.dumps(c['input'])}) -> {json.dumps(out, default=str)[:6000]}"
        for c, out in results
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What does next weekend look like for a cafe in Pune?"
    try:
        r = answer(q)
        print(r["text"])
        print("\n--- tools used ---")
        for t in r["tool_trace"]:
            print(" ", t["tool"], t["input"], "ERROR: " + t["error"] if t["error"] else "")
    except LLMUnavailable as exc:
        print(exc)
