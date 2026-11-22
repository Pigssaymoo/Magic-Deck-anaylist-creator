"""Recommander public API source.

Endpoint (live-verified): POST https://api.recommander.cards/public-release/api/decks/recommend/top
Body: {card_format: "name", commander, partner?, candidate_step?: ...}
Response: {result_code: "success"|"error_*", data: {recommendations: [{oracle_id, name, score}]}}

IMPORTANT: Recommander scores have NO fixed range (can be > 1 or negative) and
are only comparable within a single response -> we rank-normalize within the
response before blending (percentile, top card -> 1.0).

Attribution (required by their terms): recommendations served by
Recommander (https://recommander.cards).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .httpclient import Http, SourceBlocked

URL = "https://api.recommander.cards/public-release/api/decks/recommend/top"


@dataclass
class RecommanderSignal:
    name: str
    oracle_id: str
    raw_score: float
    norm: float          # percentile within the response (top -> 1.0)
    rank: int
    n: int


@dataclass
class RecommanderData:
    signals: dict[str, RecommanderSignal] = field(default_factory=dict)  # keyed by raw name


def _percentile(rank: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return (n - rank + 1) / n


def normalize_response(data: dict[str, Any]) -> dict[str, RecommanderSignal]:
    """Rank-normalize a raw Recommander response (pure, unit-testable)."""
    if (data.get("result_code") or "").lower() != "success":
        raise SourceBlocked(f"Recommander returned result_code={data.get('result_code')!r}")
    recs = ((data.get("data") or {}).get("recommendations")) or []
    # Sort by score desc (stable tie-break on name) -> rank 1..n
    recs = sorted(recs, key=lambda r: (-(r.get("score") or 0.0), (r.get("name") or "").casefold()))
    n = len(recs)
    out: dict[str, RecommanderSignal] = {}
    for i, r in enumerate(recs):
        name = (r.get("name") or "").strip()
        if not name:
            continue
        rank = i + 1
        sig = RecommanderSignal(
            name=name,
            oracle_id=r.get("oracle_id") or "",
            raw_score=float(r.get("score") or 0.0),
            norm=_percentile(rank, n),
            rank=rank,
            n=n,
        )
        existing = out.get(name)
        if existing is None or sig.norm > existing.norm:
            out[name] = sig
    return out


def fetch(http: Http, cfg: Config, commander: str, partner: str | None = None) -> RecommanderData:
    body: dict[str, Any] = {"card_format": "name", "commander": commander}
    if partner:
        body["partner"] = partner
    data = http.post_json(URL, body, ttl=cfg.http.rec_cache_days * 24 * 3600)
    signals = normalize_response(data)
    return RecommanderData(signals=signals)
