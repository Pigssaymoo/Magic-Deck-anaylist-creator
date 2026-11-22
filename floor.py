"""Budget floor precheck: the cheapest legal 100-card deck a budget could buy.

The floor is a LOWER bound on what any legal deck costs for these commanders:
commander price + companion (if any) + flat shipping + the cheapest legal
option per remaining slot. It is computed BEFORE any source fetch, solver run,
or rate-limited MTGS probe, so an impossible budget fails fast with the
minimum buildable budget instead of burning minutes on work that cannot
succeed.

Contract: if ``floor.total_nzd <= budget`` the build MUST succeed -- the
degradation ladder (R1-R4) guarantees a legal, in-budget deck exists. Only
``floor.total_nzd > budget`` is a legitimate budget failure.

Pricing rule (deterministic, probe-free):
  * the cheapest in-stock cached price for the run's condition (a previous
    live lookup, if any); else
  * Scryfall market USD x FX rate x ``pricing.estimate_margin`` -- an
    unverified price is never trusted at face value; else
  * 0.0 (no data at all).

No MTGS request is ever made here: the floor reads only the TTL cache, so it
cannot trip the rate limiter or the source circuit breaker.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cards import Card
from .config import Config
from .pricing import Pricer
from .scryfall import ScryfallDB


class FloorInfeasible(RuntimeError):
    """The identity has fewer than 99 legal cards (or not enough basic lands).

    No budget can buy a deck for these commanders; the pipeline converts this
    into a clean exit-1 with the reason.
    """


@dataclass(frozen=True)
class FloorBreakdown:
    """Cheapest possible legal deck cost, split into auditable parts."""

    commander_nzd: float
    companion_nzd: float
    shipping_nzd: float
    cards_nzd: float
    total_nzd: float

    def minimum_budget_message(self) -> str:
        """'commander NZ$X [+ companion NZ$Y] + shipping NZ$7.50 + 99 budget
        cards NZ$Z = NZ$W minimum' -- the deterministic minimum-buildable line."""
        parts = [f"commander NZ${self.commander_nzd:,.2f}"]
        if self.companion_nzd > 0:
            parts.append(f"companion NZ${self.companion_nzd:,.2f}")
        parts.append(f"shipping NZ${self.shipping_nzd:,.2f}")
        parts.append(f"99 budget cards NZ${self.cards_nzd:,.2f}")
        return " + ".join(parts) + f" = NZ${self.total_nzd:,.2f} minimum"


def _round2(v: float) -> float:
    return float(round(v + 1e-9, 2))


def _card_price(card: Card, pricer: Pricer | None, cfg: Config, condition: str, rate: float | None) -> float:
    """Cached verified price, else margined MV estimate, else 0.0. No network."""
    if pricer is not None:
        cached = pricer.cached_price(card.name, condition)
        if cached is not None:
            return cached
    mv = card.mv_usd
    if mv and pricer is not None and rate:
        return _round2(mv * rate * cfg.pricing.estimate_margin)
    return 0.0


def compute_floor(
    db: ScryfallDB,
    cfg: Config,
    commanders: list[Card],
    companion: Card | None,
    pricer: Pricer | None,
    condition: str,
    free_names: frozenset[str] | None = None,
) -> FloorBreakdown:
    """Cheapest legal 100-card deck (commander/companion + 99 slots + shipping).

    Legality pool: identity-legal cards (``color_identity`` subset of the
    commanders' union), excluding the commanders and the companion and any
    other Companion. The final deck must carry at least one basic land per
    identity color (a deck that cannot cast a color's spells is not the deck
    the builder promises), so the 99 budget slots must include >= N basic
    lands where N = number of identity colors. With all prices >= 0, the
    minimum-cost choice under that constraint is exactly: the N cheapest
    basic lands, then the 99-N cheapest remaining legal cards (exchange
    argument: any deck missing one of the N cheapest basics can swap in a
    pricier or equal basic it already carries, never increasing cost).

    ``free_names`` (casefolded card names the user already owns via
    --owned/--free-basics) are priced at NZ$0.00 -- they cost nothing and are
    never probed, so a deck built entirely from owned cards has a floor of
    shipping only.

    Raises :class:`FloorInfeasible` when the identity has fewer than 99 legal
    cards or fewer than N basic lands -- no budget can help in that case.
    """
    idset: set[str] = set()
    for c in commanders:
        idset.update(c.color_identity)
    excluded = {c.oracle_id for c in commanders}
    if companion is not None:
        excluded.add(companion.oracle_id)

    pool = [
        c
        for c in db.all
        if c.commander_legal
        and set(c.color_identity) <= idset
        and c.oracle_id not in excluded
        and not c.is_companion
    ]
    basics = [c for c in pool if c.is_basic_land]
    n_basics = len(idset)

    if len(pool) < 99:
        raise FloorInfeasible(
            f"not enough legal cards for identity {''.join(sorted(idset)) or 'colorless'}: "
            f"{len(pool)} candidate cards, need 99 (excluding {len(commanders)} commander(s)"
            f"{', companion' if companion is not None else ''})"
        )
    if len(basics) < n_basics:
        raise FloorInfeasible(
            f"not enough basic lands to cover identity "
            f"{''.join(sorted(idset)) or 'colorless'}: {len(basics)} basic land(s), need {n_basics}"
        )

    rate: float | None = None
    if pricer is not None:
        rate = pricer.fx_rate()

    # Free cards (--owned/--free-basics) cost NZ$0.00 in the floor, too.
    free = frozenset(n.casefold() for n in (free_names or ()))

    def _price(c: Card | None) -> float:
        if c is not None and c.name.casefold() in free:
            return 0.0
        return _card_price(c, pricer, cfg, condition, rate)

    # Greedy optimum: N cheapest basics, then 99-N cheapest of what remains.
    chosen_basics = sorted(basics, key=lambda c: (_price(c), c.name.casefold()))
    n_basics_needed = min(n_basics, len(basics))
    chosen = list(chosen_basics[:n_basics_needed])
    chosen_names = {c.oracle_id for c in chosen}
    rest = sorted(
        (c for c in pool if c.oracle_id not in chosen_names),
        key=lambda c: (_price(c), c.name.casefold()),
    )
    chosen.extend(rest[: 99 - len(chosen)])

    cards_nzd = _round2(sum(_price(c) for c in chosen))
    commander_nzd = _round2(sum(_price(c) for c in commanders))
    companion_nzd = _price(companion) if companion is not None else 0.0
    shipping_nzd = cfg.pricing.shipping_nzd
    total_nzd = _round2(commander_nzd + companion_nzd + cards_nzd + shipping_nzd)
    return FloorBreakdown(
        commander_nzd=commander_nzd,
        companion_nzd=companion_nzd,
        shipping_nzd=shipping_nzd,
        cards_nzd=cards_nzd,
        total_nzd=total_nzd,
    )
