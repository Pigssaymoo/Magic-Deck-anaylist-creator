"""edh.tools recommendation source.

Endpoints (live-verified):
  GET  https://edh.tools/api/cards?q={name}            -> card search (oracle ids)
  POST https://edh.tools/api/recommendations           -> {target_oracle_ids, synergy, page}
  Response: {cards: [{oracle_id, name, score (0-1), ...}], more, page, targets}

edh.tools scores are rank-percentile-normalized within the response (top
card -> 1.0), matching the convention the other sources use, so the
cross-source blend is scale-comparable regardless of the native scale.
The raw score is kept on the signal (``score``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .httpclient import Http, SourceBlocked

SEARCH_URL = "https://edh.tools/api/cards"
REC_URL = "https://edh.tools/api/recommendations"
MAX_PAGES = 2


@dataclass
class EdhToolsSignal:
    name: str
    oracle_id: str
    score: float
    norm: float
    rank: int
    n: int


@dataclass
class EdhToolsData:
    signals: dict[str, EdhToolsSignal] = field(default_factory=dict)  # keyed by raw name


def _percentile(rank: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return (n - rank + 1) / n


def normalize_cards(cards: list[dict[str, Any]]) -> dict[str, EdhToolsSignal]:
    """Normalize edh.tools recommendation cards (pure, unit-testable).

    Scores are rank-percentile-normalized within the response (top card ->
    1.0), matching the convention the other sources use, so the cross-source
    blend is scale-comparable regardless of the native scale. The raw score
    is kept on the signal (``score``).
    """
    cleaned = [
        c
        for c in cards
        if c.get("name") and isinstance(c.get("score"), (int, float))
    ]
    if not cleaned:
        return {}
    cleaned.sort(key=lambda c: (-float(c["score"]), (c.get("name") or "").casefold()))
    n = len(cleaned)
    out: dict[str, EdhToolsSignal] = {}
    for i, c in enumerate(cleaned):
        name = (c.get("name") or "").strip()
        rank = i + 1
        raw = float(c["score"])
        sig = EdhToolsSignal(
            name=name,
            oracle_id=c.get("oracle_id") or c.get("id") or "",
            score=raw,
            norm=_percentile(rank, n),
            rank=rank,
            n=n,
        )
        existing = out.get(name)
        if existing is None or sig.norm > existing.norm:
            out[name] = sig
    return out


def fetch(http: Http, cfg: Config, commander_oracle_ids: list[str]) -> EdhToolsData:
    if not commander_oracle_ids:
        raise SourceBlocked("edh.tools: no target oracle ids available")
    all_cards: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        body = {
            "target_oracle_ids": commander_oracle_ids,
            "synergy": True,
            "page": page,
        }
        data = http.post_json(REC_URL, body, ttl=cfg.http.rec_cache_days * 24 * 3600)
        batch = data.get("cards") or []
        all_cards.extend(batch)
        if not data.get("more"):
            break
    return EdhToolsData(signals=normalize_cards(all_cards))


def search_card(http: Http, cfg: Config, name: str) -> dict[str, Any] | None:
    """Search edh.tools for a card (used to resolve oracle ids)."""
    data = http.get_json(SEARCH_URL, ttl=cfg.http.rec_cache_days * 24 * 3600, params={"q": name})
    results = data if isinstance(data, list) else (data.get("cards") or data.get("results") or [])
    for r in results:
        if (r.get("name") or "").casefold() == name.casefold():
            return r
    return results[0] if results else None
