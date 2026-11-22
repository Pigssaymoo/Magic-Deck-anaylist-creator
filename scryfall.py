"""Scryfall bulk-data canonicalization.

One entry per unique card definition. The feed is the Scryfall "Oracle Cards"
bulk dataset: a gzipped JSONL with exactly one card object per Oracle ID,
which is precisely the canonical-card semantics we need.

The descriptor endpoint (``/bulk-data/oracle_cards``) returns a
``jsonl_download_uri`` whose filename embeds a timestamp. We download it once
into ``<cache>/scryfall/`` and reuse the file while it exists, so repeated
runs are offline-friendly.
"""
from __future__ import annotations

import difflib
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from .cards import Card, ordered_identity
from .config import Config
from .httpclient import USER_AGENT, Http, SourceBlocked

ORACLE_CARDS_URL = "https://api.scryfall.com/bulk-data/oracle_cards"
SEARCH_URL = "https://api.scryfall.com/cards/search"


def _parse_card(raw: dict[str, Any]) -> Card | None:
    scryfall_id = raw.get("id")
    if not scryfall_id:
        return None
    prices = raw.get("prices") or {}
    mv: float | None = None
    for key in ("usd", "usd_face", "usd_tix"):
        val = prices.get(key)
        if val:
            try:
                mv = round(float(val), 2)
                break
            except (TypeError, ValueError):
                continue
    type_line = raw.get("type_line") or ""
    card_types = [t.strip().lower() for t in type_line.split("\u2014")[0].split() if t.strip()]
    legalities = raw.get("legalities") or {}
    return Card(
        name=raw.get("name") or "Unknown",
        scryfall_id=scryfall_id,
        oracle_id=raw.get("oracle_id") or scryfall_id,
        mana_cost=raw.get("mana_cost") or "",
        cmc=int(raw.get("cmc") or 0),
        type_line=type_line,
        colors=list(raw.get("colors") or []),
        color_identity=ordered_identity(raw.get("color_identity") or []),
        card_types=card_types,
        oracle_text=raw.get("oracle_text") or "",
        mv_usd=mv,
        printed_in=list(raw.get("printed_in") or []),
        keywords=[k.lower() for k in (raw.get("keywords") or [])],
        commander_legal=legalities.get("commander") == "legal",
    )


class ScryfallDB:
    """In-memory canonical card database."""

    def __init__(self, all_cards: list[Card]) -> None:
        self.all: list[Card] = all_cards
        self.by_name: dict[str, Card] = {}
        self.by_oracle: dict[str, Card] = {}
        for c in all_cards:
            key = c.name.casefold()
            if key not in self.by_name:
                self.by_name[key] = c
            if c.oracle_id not in self.by_oracle:
                self.by_oracle[c.oracle_id] = c
        self.commanders: list[Card] = [
            c for c in all_cards if c.commander_legal and c.is_creature
        ]

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, http: Http, cfg: Config) -> "ScryfallDB":
        ttl = max(3600, cfg.http.rec_cache_days * 24 * 3600)
        desc = http.get_json(ORACLE_CARDS_URL, ttl=ttl)
        uri = str(desc.get("jsonl_download_uri") or "")
        if not uri.startswith("https://"):
            raise SourceBlocked(
                f"Scryfall bulk descriptor has no valid jsonl_download_uri: {desc!r}"
            )
        cache_dir = Path(cfg.http.cache_dir) / "scryfall"
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = uri.rsplit("/", 1)[-1] or "oracle-cards.jsonl.gz"
        dest = cache_dir / filename
        if not dest.exists() or dest.stat().st_size == 0:
            _download(uri, dest)
        _prune_old(cache_dir)
        cards = _parse_feed(dest)
        if not cards:
            raise SourceBlocked(
                "Scryfall oracle-cards feed parsed to zero cards; "
                "refusing to continue with an empty card database"
            )
        return cls(cards)


def _download(uri: str, dest: Path, attempts: int = 4) -> None:
    """Stream the gzipped JSONL to ``dest`` (atomic), retrying with backoff."""
    tmp = dest.with_name(dest.name + ".part")
    backoff = 2.0
    last: Exception | None = None
    for _ in range(attempts):
        try:
            with requests.get(
                uri, stream=True, timeout=(30, 900), headers={"User-Agent": USER_AGENT}
            ) as r:
                if r.status_code != 200:
                    raise IOError(f"HTTP {r.status_code} for {uri}")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            with open(tmp, "rb") as fh:
                if fh.read(2) != b"\x1f\x8b":
                    raise IOError("downloaded file does not look like gzip")
            os.replace(tmp, dest)
            return
        except Exception as exc:  # noqa: BLE001 - retry any failure, then raise
            last = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    raise SourceBlocked(f"failed to download Scryfall bulk data from {uri}: {last}")


def _parse_feed(path: Path) -> list[Card]:
    """Parse the gzipped JSONL; one Card per oracle_id (lowest Scryfall id)."""
    best: dict[str, Card] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            card = _parse_card(raw)
            if card is None:
                continue
            existing = best.get(card.oracle_id)
            if existing is None or card.scryfall_id < existing.scryfall_id:
                best[card.oracle_id] = card
    return sorted(best.values(), key=lambda c: c.name.casefold())


def _prune_old(cache_dir: Path, keep: int = 3) -> None:
    """Keep only the newest few bulk feeds; the filename embeds a timestamp."""
    try:
        files = sorted(
            (p for p in cache_dir.glob("*.jsonl.gz") if not p.name.endswith(".part")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in files[keep:]:
            p.unlink(missing_ok=True)
    except OSError:
        pass


# ----------------------------------------------------------------- lookups
def exact_card(db: ScryfallDB, name: str) -> Card | None:
    return db.by_name.get(name.casefold())


def find_card(db: ScryfallDB, name: str, threshold: float = 0.85) -> Card | None:
    """Fuzzy card lookup: exact (casefold) first, then best difflib match."""
    hit = exact_card(db, name)
    if hit is not None:
        return hit
    candidates = list(db.by_name.values())
    best_name, best_ratio = "", 0.0
    for card in candidates:
        ratio = difflib.SequenceMatcher(None, name.casefold(), card.name.casefold()).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, card.name
    if best_ratio >= threshold:
        return db.by_name[best_name.casefold()]
    return None


def find_commander(db: ScryfallDB, name: str, threshold: float = 0.85) -> Card | None:
    """Resolve a user-supplied commander name (fuzzy) among legal commanders."""
    hit = exact_card(db, name)
    if hit is not None and hit.commander_legal:
        return hit
    names = [c.name for c in db.commanders]
    close = difflib.get_close_matches(name, names, n=3, cutoff=threshold)
    if close:
        return db.by_name[close[0].casefold()]
    # last resort: any legendary creature with a close name
    return find_card(db, name, threshold)


def closest_matches(db: ScryfallDB, name: str, n: int = 3) -> list[tuple[str, float]]:
    """Best-effort suggestions for error messages (never raises)."""
    names = [c.name for c in db.commanders]
    scored = []
    for cand in names:
        ratio = difflib.SequenceMatcher(None, name.casefold(), cand.casefold()).ratio()
        scored.append((cand, round(ratio, 3)))
    scored.sort(key=lambda t: (-t[1], t[0].casefold()))
    return scored[:n]


def search_cards(http: Http, query: str, *, cap: int = 500, ttl_hours: float = 24.0) -> list[Card]:
    """Paged Scryfall card search (e.g. ``otag:draw``, ``atag:tokens``).

    Results are CMC-ascending (``order=cmc``), one request per page via the
    rate-limited Http client, capped at ``cap`` cards. Each result shares the
    canonical card JSON shape, so ``_parse_card`` converts it directly.
    """
    ttl = ttl_hours * 3600
    cards: list[Card] = []
    seen: set[str] = set()
    page = 1
    while len(cards) < cap:
        data = http.get_json(
            SEARCH_URL,
            ttl=ttl,
            params={"q": query, "order": "cmc", "page": page, "per_page": 100},
        )
        batch = data.get("data") or []
        if not batch:
            break
        for raw in batch:
            card = _parse_card(raw)
            if card is None or card.oracle_id in seen:
                continue
            seen.add(card.oracle_id)
            cards.append(card)
            if len(cards) >= cap:
                break
        if not data.get("has_more"):
            break
        page += 1
    return cards


def tag_search(http: Http, query: str, *, cap: int = 500) -> list[Card]:
    """Tag-mode card search with a helpful error on zero results."""
    cards = search_cards(http, query, cap=cap)
    if not cards:
        raise SystemExit(
            f"error: Scryfall search {query!r} returned no cards; "
            "check the tag syntax (e.g. otag:draw, atag:tokens)"
        )
    return cards
