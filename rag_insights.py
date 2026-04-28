import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROUTE_BENCHMARKS = {
    "KUL-SIN": {"min": 140, "typical": 220, "max": 420},
    "KUL-BKK": {"min": 230, "typical": 380, "max": 700},
    "KUL-TPE": {"min": 520, "typical": 800, "max": 1400},
    "KUL-KCH": {"min": 150, "typical": 260, "max": 520},
    "KUL-MYY": {"min": 170, "typical": 300, "max": 580},
}


@dataclass
class RetrievedDoc:
    text: str
    score: float
    metadata: Dict[str, str]


class SimpleVectorDB:
    """
    Lightweight vector retrieval using TF-IDF embeddings (equivalent vector model).
    """

    def __init__(self, docs: List[Dict[str, str]]):
        self.docs = docs
        corpus = [d["text"] for d in docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def similarity_search(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_vectors).flatten()
        idxs = np.argsort(sims)[::-1][:k]
        out = []
        for i in idxs:
            out.append(
                RetrievedDoc(
                    text=self.docs[i]["text"],
                    score=float(sims[i]),
                    metadata=self.docs[i].get("metadata", {}),
                )
            )
        return out


def _build_kb_docs() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []

    for route, b in ROUTE_BENCHMARKS.items():
        docs.append(
            {
                "text": (
                    f"Route benchmark {route}: market fare usually ranges RM {b['min']} to RM {b['max']}, "
                    f"typical selling zone RM {b['typical']}. "
                    f"Below minimum often indicates aggressive underpricing. "
                    f"Above maximum often needs premium demand conditions."
                ),
                "metadata": {"type": "route_benchmark", "route": route},
            }
        )

    docs.extend(
        [
            {
                "text": (
                    "Elasticity heuristic: leisure-dominant routes are price sensitive. "
                    "If price rises far above market baseline, conversion usually drops quickly."
                ),
                "metadata": {"type": "demand"},
            },
            {
                "text": (
                    "Business-heavy or urgency windows can tolerate higher pricing, "
                    "especially for short booking horizons and low seats remaining."
                ),
                "metadata": {"type": "demand"},
            },
            {
                "text": (
                    "Airline pricing heuristic: expected load factor above 90% with strong revenue "
                    "can indicate underpricing risk if seats are likely to sell out."
                ),
                "metadata": {"type": "heuristic"},
            },
            {
                "text": (
                    "Airline pricing heuristic: expected load factor below 40% near departure "
                    "can indicate overpricing risk or weak demand; tactical discounting may help."
                ),
                "metadata": {"type": "heuristic"},
            },
            {
                "text": (
                    "Market validation rule: optimal fare should usually sit within 70%-150% of market base fare, "
                    "unless special events, disruptions, or extreme demand shocks apply."
                ),
                "metadata": {"type": "heuristic"},
            },
            {
                "text": (
                    "Weather and holiday interaction: strong weather score and holiday periods "
                    "often support stronger willingness-to-pay."
                ),
                "metadata": {"type": "demand"},
            },
        ]
    )
    return docs


def get_vector_db() -> SimpleVectorDB:
    return SimpleVectorDB(_build_kb_docs())


def build_runtime_query(
    route: str,
    days: int,
    weather: float,
    holiday: bool,
    price: float,
    demand: float,
    load_factor: float,
) -> str:
    return f"""
Route: {route}
Days to departure: {days}
Weather score: {weather}
Holiday: {holiday}
Optimal price: {price}
Expected demand: {demand}
Load factor: {load_factor}
"""


def _market_validation(
    route_key: str, optimal_price: float, market_price: float
) -> Tuple[str, List[str], List[str]]:
    warnings: List[str] = []
    recs: List[str] = []

    bench = ROUTE_BENCHMARKS.get(route_key)
    if bench is None:
        status = (
            f"No specific route benchmark for {route_key}; compared against market anchor RM {market_price:.2f}."
        )
    else:
        status = (
            f"Benchmark for {route_key}: RM {bench['min']} to RM {bench['max']} "
            f"(typical RM {bench['typical']})."
        )
        if optimal_price < bench["min"]:
            warnings.append("Optimal price is below historical minimum benchmark.")
            recs.append("Consider raising fare gradually and monitor conversion impact.")
        if optimal_price > bench["max"]:
            warnings.append("Optimal price is above historical maximum benchmark.")
            recs.append("Validate premium assumptions (event demand, urgency, capacity tightness).")

    lower = market_price * 0.7
    upper = market_price * 1.5
    if optimal_price < lower or optimal_price > upper:
        warnings.append("Optimal price sits outside market-anchor band (70%-150%).")
        recs.append("Review constraints or scenario assumptions for realism.")

    return status, warnings, recs


def _llm_explanation(prompt: str) -> str:
    """
    Optional LLM synthesis. Falls back to deterministic explanation if unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""

    try:
        from openai import OpenAI  # optional dependency

        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=280,
        )
        return (resp.output_text or "").strip()
    except Exception:
        return ""


def generate_pricing_insight(
    route_key: str,
    scenario: Dict[str, float],
    optimizer_out: Dict[str, float],
    retrieved_docs: List[RetrievedDoc],
) -> Dict[str, object]:
    optimal_price = float(optimizer_out["optimal_price"])
    expected_demand = float(optimizer_out["expected_demand"])
    expected_revenue = float(optimizer_out["expected_revenue"])
    market_price = float(optimizer_out.get("market_price", scenario.get("market_price", 300.0)))
    load_factor = float(scenario.get("load_factor", 0.0))

    market_text, warnings, recs = _market_validation(route_key, optimal_price, market_price)

    if load_factor > 90:
        warnings.append("Very high load factor: potential underpricing risk.")
        recs.append("Test a modest fare increase and compare revenue uplift.")
    elif load_factor < 35:
        warnings.append("Low load factor: potential overpricing risk.")
        recs.append("Test tactical discount or ancillary bundles to stimulate bookings.")

    boundary_hit = bool(optimizer_out.get("iterations_used", 1) > 1)
    if boundary_hit:
        recs.append("Adaptive sweep expanded range; monitor if optimal remains near upper boundary.")

    confidence = "High"
    if len(warnings) >= 2:
        confidence = "Medium"
    if len(warnings) >= 4:
        confidence = "Low"

    retrieved_summary = "\n".join(
        [f"- ({d.score:.2f}) {d.text}" for d in retrieved_docs[:3]]
    )

    plain_explanation = (
        f"The optimizer selected RM {optimal_price:.2f} with expected demand {expected_demand:.1f} "
        f"and expected revenue RM {expected_revenue:.2f}. "
        f"This result is anchored to market base fare RM {market_price:.2f} and scenario demand signals "
        f"(days-to-departure, weather, holiday, and seat pressure)."
    )

    prompt = f"""
You are an airline pricing strategy assistant.
Explain the pricing recommendation in plain English.
Include:
1) Why this price is optimal for the scenario
2) Comparison against market range
3) Any anomalies or risks
4) Practical next-step recommendation

Scenario:
{scenario}

Optimizer Output:
{optimizer_out}

Retrieved Knowledge:
{retrieved_summary}
"""
    llm_text = _llm_explanation(prompt)
    explanation = llm_text if llm_text else plain_explanation

    return {
        "explanation": explanation,
        "market_validation": market_text,
        "confidence": confidence,
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": list(dict.fromkeys(recs)),
        "retrieved_docs": retrieved_docs,
    }
