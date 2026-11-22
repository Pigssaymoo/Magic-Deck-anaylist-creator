"""MTG Singles NZ pricing with Scryfall-MV/FX fallback.

Verified endpoint:
  GET https://api.mtgsingles.co.nz/MtgSingle?query={name}&page=1&pageSize=20&Country=1
Requirements: browser-like User-Agent + Referer, >= ~2s between requests
(their documented BAN warning). Prices are strings like "$15.87";
``availableQty`` is stock.

Fallback when a card is unmatched or out of stock and no substitution works:
Scryfall market USD x frankfurter.app USD->NZD, flagged ``estimated=True``
(never presented as verified buyable).
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from .cards import Card
from .config import Config
from .httpclient import Http, SourceBlocked

MTGS_BASE = "https://api.mtgsingles.co.nz/MtgSingle"
MTGS_HOST = "api.mtgsingles.co.nz"
FX_URL = "https://api.frankfurter.app/latest"


@dataclass
class PriceInfo:
    name: str
    price_nzd: float | None
    condition: str
    url: str
    stock: int
    estimated: bool = False
    source: str = "mtgsingles"
    printing: str | None = None
    note: str = ""


def parse_price(value: object) -> float | None:
    """'$15.87' -> 15.87; handles plain numbers and None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    s = str(value).replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _norm_name(s: str) -> str:
    return " ".join(s.lower().split())


_PAREN = re.compile(r"\s*\([^)]*\)")


def _clean_name(s: str) -> str:
    """Normalize for listing matching: lowercase, collapse whitespace, and
    drop parenthetical printing markers ('(Oversized)', '(Foil)', ...)."""
    return " ".join(_PAREN.sub(" ", s).lower().split())


class Pricer:
    """Rate-limited MTG Singles NZ lookup + FX-estimated fallback."""

    def __init__(self, http: Http, cfg: Config, free_names: frozenset[str] | None = None) -> None:
        self.http = http
        self.cfg = cfg
        http.set_interval(MTGS_HOST, cfg.pricing.mtgs.min_interval_sec)
        # Circuit breaker: after this many consecutive source failures (blocked,
        # non-JSON, down) stop probing MTGS at all -- a fully-down source
        # should not burn rate-limited dead time across hundreds of later probes.
        self._breaker_threshold = max(1, cfg.pricing.mtgs.block_breaker)
        self._consecutive_blocks = 0
        self._breaker_open = False
        # Free cards (--owned / --free-basics): the user already owns them, so
        # they cost $0 and are NEVER probed (no MTGS request, no cap consumption).
        # Keys are casefolded card names.
        self.free_names = frozenset(n.casefold() for n in (free_names or ()))
        # Distinct card names that have triggered a live MTGS lookup this run.
        # Repeat/cache hits (already in this set) do not re-count, so the cap
        # bounds UNIQUE names, not call count.
        self._probed: set[str] = set()

    # ------------------------------------------------------------- free cards
    def is_free(self, name: str) -> bool:
        return name.casefold() in self.free_names

    def free_price(self, name: str) -> PriceInfo:
        """A free (user-owned) card: in stock, NZ$0.00, no probe, not an estimate."""
        return PriceInfo(
            name=name,
            price_nzd=0.0,
            condition="any",
            url="",
            stock=1,
            estimated=False,
            source="owned",
            note="free card (--owned/--free-basics): user owns it, no price, no probe",
        )

    # --------------------------------------------------------------- probe cap
    def _probe_allowed(self, name: str) -> bool:
        """Whether ``name`` may spend a live MTGS lookup under the per-run cap.

        The cap bounds DISTINCT names (``pricing.max_probes_per_run``); a name
        already probed this run (a cache hit) never re-counts. ``0`` disables.
        """
        cap = max(0, int(self.cfg.pricing.max_probes_per_run))
        key = name.casefold()
        if cap == 0 or key in self._probed:
            return True
        return len(self._probed) < cap

    # ------------------------------------------------------------------ MTGS
    def _note_ok(self) -> None:
        self._consecutive_blocks = 0

    def _note_blocked(self) -> None:
        self._consecutive_blocks += 1
        if self._consecutive_blocks >= self._breaker_threshold:
            self._breaker_open = True

    def _mtgs_items(self, name: str, page: int, page_size: int) -> list[dict]:
        if self._breaker_open:
            raise SourceBlocked(
                "MTG Singles NZ circuit breaker open "
                f"(>= {self._breaker_threshold} consecutive source failures)"
            )
        params = {"query": name, "page": page, "pageSize": page_size, "Country": 1}
        headers = {
            "User-Agent": self.cfg.pricing.mtgs.user_agent,
            "Referer": self.cfg.pricing.mtgs.referer,
        }
        data = self.http.get_json(
            MTGS_BASE,
            ttl=self.cfg.http.price_cache_hours * 3600,
            params=params,
            headers=headers,
        )
        return self._items_from(data)

    @staticmethod
    def _items_from(data: object) -> list[dict]:
        """Listings from a raw body: a list, or a {data|items|results|rows}
        wrapper. Anything else (non-object JSON like a string or number) is
        no listings -- tolerated, never a crash."""
        if isinstance(data, list):
            return [it for it in data if isinstance(it, dict)]
        if isinstance(data, dict):
            for key in ("data", "items", "results", "rows"):
                value = data.get(key)
                if isinstance(value, list):
                    return [it for it in value if isinstance(it, dict)]
        return []

    @staticmethod
    def _match(name: str, listing_name: str) -> bool:
        a, b = _clean_name(name), _clean_name(listing_name)
        if not a or not b:
            return False
        if a == b:
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.92

    def _condition_ok(self, listing_condition: str, wanted: str) -> bool:
        if not wanted or wanted.casefold() in ("any", ""):
            return True
        return wanted.casefold() in (listing_condition or "").casefold()

    def _select_listing(
        self, name: str, items: list[dict], condition: str
    ) -> tuple[dict, float, int] | None:
        """Cheapest in-stock listing for (name, condition) -> (item, price, qty).

        Shared by :meth:`price_card` (live) and :meth:`cached_price`
        (probe-free) so both apply identical selection rules: fuzzy name
        match, condition filter, stock > 0, parseable price, lowest wins.
        """
        best: tuple[dict, float, int] | None = None
        for it in items:
            listing_name = str(
                it.get("title")
                or it.get("Title")
                or it.get("name")
                or it.get("Name")
                or ""
            )
            if not listing_name or not self._match(name, listing_name):
                continue
            if not self._condition_ok(str(it.get("condition") or it.get("Condition") or ""), condition):
                continue
            try:
                qty = int(it.get("availableQty") if it.get("availableQty") is not None else (it.get("qty") or 0))
            except (TypeError, ValueError):
                qty = 0
            if qty <= 0:
                continue
            price = parse_price(it.get("price", it.get("Price")))
            if price is None:
                continue
            if best is None or price < best[1]:
                best = (it, price, qty)
        return best

    def cached_price(self, name: str, condition: str = "any") -> float | None:
        """Cheapest in-stock NZD price for ``name`` read from the TTL cache; None on any miss.

        Probe-free by design: it reads only what a previous live lookup stored
        (the same cache key ``price_card`` writes), so the budget-floor
        precheck can price a whole deck without a single MTGS request, rate-limit
        wait, or breaker trip. Selection rules are identical to ``price_card``.
        A free (user-owned) card is NZ$0.00 by construction.
        """
        if self.is_free(name):
            return 0.0
        raw = self.http.cached_body(
            MTGS_BASE,
            ttl=self.cfg.price_ttl_seconds(),
            params={
                "query": name,
                "page": 1,
                "pageSize": self.cfg.pricing.mtgs.page_size,
                "Country": 1,
            },
        )
        if raw is None:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        pick = self._select_listing(name, self._items_from(payload), condition)
        return pick[1] if pick else None

    def price_card(self, name: str, condition: str = "any") -> PriceInfo | None:
        """Cheapest in-stock MTGS listing matching name + condition, else None.

        Never raises for a source failure (blocked, non-JSON, down): those
        return None, which callers treat as "unverified" (the commander gate
        counts the probe as unavailable; substitution falls back to estimates).

        A free (user-owned) card returns NZ$0.00 without any probe. A name that
        would exceed the per-run probe cap (``pricing.max_probes_per_run``)
        returns None (unverified) rather than spending a network lookup -- the
        cap is a hard bound on distinct live lookups, not a soft hint.
        """
        if self.is_free(name):
            return self.free_price(name)
        if not self._probe_allowed(name):
            return None
        self._probed.add(name.casefold())
        for page in range(1, self.cfg.pricing.mtgs.max_pages + 1):
            try:
                items = self._mtgs_items(name, page, self.cfg.pricing.mtgs.page_size)
            except SourceBlocked:
                self._note_blocked()
                break
            self._note_ok()
            pick = self._select_listing(name, items, condition)
            if pick is not None:
                it, price, qty = pick
                return PriceInfo(
                    name=name,
                    price_nzd=price,
                    condition=str(it.get("condition") or it.get("Condition") or ""),
                    url=str(
                        it.get("link")
                        or it.get("url")
                        or f"https://www.mtgsingles.co.nz/?s={name.replace(' ', '+')}"
                    ),
                    stock=qty,
                    estimated=False,
                    source="mtgsingles",
                    printing=str(it.get("setName") or it.get("set") or it.get("printing") or ""),
                )
        return None

    # -------------------------------------------------------------------- FX
    def fx_rate(self) -> float | None:
        try:
            data = self.http.get_json(
                FX_URL,
                ttl=self.cfg.pricing.fx.cache_hours * 3600,
                params={"from": self.cfg.pricing.fx.base, "to": self.cfg.pricing.fx.quote},
            )
            rate = (data.get("rates") or {}).get(self.cfg.pricing.fx.quote)
            return float(rate) if rate else None
        except (SourceBlocked, TypeError, ValueError):
            return None

    def estimate(self, card: Card) -> PriceInfo:
        """Scryfall market USD -> NZD via frankfurter.app (flagged estimated).

        A free (user-owned) card is NZ$0.00 -- it is never estimated.
        """
        if self.is_free(card.name):
            return self.free_price(card.name)
        rate = self.fx_rate()
        price = round(card.mv_usd * rate, 2) if (card.mv_usd is not None and rate) else None
        return PriceInfo(
            name=card.name,
            price_nzd=price,
            condition="any",
            url="https://scryfall.com/search?q=!%s" % card.name.replace(" ", "+"),
            stock=0,
            estimated=True,
            source="scryfall-mv-x-fx",
            printing=", ".join(card.printed_in[:3]) or None,
            note="estimated from Scryfall market USD x frankfurter.app rate; not verified buyable",
        )
