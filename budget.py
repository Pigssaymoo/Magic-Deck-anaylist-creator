"""Unified pricing + budget + substitution (Phase 3).

Replaces the old two-stage flow:

* ``substitution.price_deck``  - live-priced EVERY deck card (hundreds of MTGS
  calls) and swapped out-of-stock cards on the fly;
* ``budget.enforce_budget``    - up to 40 greedy swap passes over live prices
  to claw the deck back under budget.

New flow (estimate-first, verify-last):

1. ``deckbuilder.assembly.assemble`` has already chosen the deck under the
   budget, using ESTIMATED prices (``mv_usd * FX``) and a binary search on the
   price penalty ``lambda`` (see ``deckbuilder.assembly._budget_search``).
2. :func:`resolve` verifies ONLY the selected cards against MTGS (sequential,
   TTL-cached), substitutes out-of-stock non-commanders with the first
   in-stock alternative (bounded probes), and - if the verified total still
   blows the budget - re-solves the assembly with a strictly higher ``lambda``
   (warm-started above the previous one), at most ``assembly.max_resolves``
   times. Out-of-stock cards are excluded from the re-solved pools.

Verified prices are cached for the lifetime of a :func:`resolve` call so that
re-solve passes never re-price an already-verified in-stock card (MTGS request
count stays bounded). Only in-stock hits are cached; out-of-stock results are
not (they are excluded from the pool instead).

Every deck change is recorded in one :class:`Substitution` log stream; reasons
are ``"stock: ..."`` or ``"budget: ..."`` (R2 pair-swap, R1/R3/R4 rung saves).
Commanders are never substituted: an out-of-stock commander stays in the deck
at its estimated price, logged as protected.

When the verified total still exceeds the budget after the re-solve budget is
exhausted, the degradation ladder degrades card quality in a fixed order
(rungs are logged in ``BudgetResult.degradation_rungs_applied``, a rung only
when it changed the deck): R1 re-prices the same cards under the widest
condition; R2 is the role-preserving re-solves + verified swap pass; R3
relaxes the non-structural quotas (theme -> any cheaper verified non-land,
non-basic land -> cheaper verified basic); R4 is floor fill at the cheapest
verified card per slot role. R3/R4 use the verified price cache only - the
ladder never triggers an MTGS probe.

Deterministic: stable sorts (value desc, est asc, name.casefold), probe order
recorded in the log, no randomness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .assembly import AssembleResult, assemble
from .cards import Card
from .config import Config
from .pricing import PriceInfo, Pricer
from .roles import tag
from .scoring import Candidate
from .scryfall import ScryfallDB

# Sentinel replacement labels (kept out of the deck; logged only).
COMMANDER_PROTECTED = "(no substitute - commander protected)"
NO_ALT = "(no substitute)"

_BUDGET_PREFIX = "budget:"


@dataclass
class Substitution:
    """One logged deck change (stock substitution or budget re-solve pairing).

    ``score_delta`` is ``new - old`` (shown signed). ``estimated`` is True when
    ``price_replacement`` is an estimate (MV x FX) rather than a verified
    in-stock MTGS price.
    """

    original: str
    replacement: str
    role: str
    reason: str
    score_delta: float
    price_original: float | None
    price_replacement: float | None
    estimated: bool = False


@dataclass
class BudgetResult:
    """Final budget position for the resolved deck."""

    budget_nzd: float | None
    cards_total_nzd: float
    shipping_nzd: float
    total_nzd: float
    within_budget: bool
    gap_nzd: float = 0.0
    swaps: list[Substitution] = field(default_factory=list)
    resolves: int = 0  # budget re-solves performed (0..assembly.max_resolves)
    # Degradation ladder rungs (R1..R4) that actually produced >=1 change,
    # in ladder order; no-op rungs are not logged.
    degradation_rungs_applied: list[str] = field(default_factory=list)


def cards_total(prices: dict[str, PriceInfo]) -> float:
    """Sum of non-None ``price_nzd`` values, rounded to cents."""
    return round(sum(p.price_nzd for p in prices.values() if p.price_nzd is not None), 2)


@dataclass
class ResolveResult:
    """Output of :func:`resolve` (verified prices + unified substitution log)."""

    asm: AssembleResult
    prices: dict[str, PriceInfo]  # deck card name -> verified or estimated PriceInfo
    subs: list[Substitution]  # unified log stream (stock + budget)
    budget: BudgetResult
    resolves: int  # budget re-solves performed (0..assembly.max_resolves)
    oos_excluded: list[str]  # OOS non-commander names excluded from re-solves


# --------------------------------------------------------------------- helpers
def _est_price(card: Card | None, fx: float | None) -> float:
    """Estimated NZ$ = ``mv_usd * FX``. Missing data -> 0 (budget-neutral)."""
    if card is None or card.mv_usd is None or fx is None:
        return 0.0
    return round(card.mv_usd * fx, 2)


def _value_of(cands: dict[str, Candidate], cfg: Config, card: Card | None) -> float:
    """A card's solver value: its consensus score if a source ranked it, else
    the flat filler prior (``assembly.filler_value``)."""
    if card is None:
        return 0.0
    c = cands.get(card.name)
    return c.score if (c is not None and c.score is not None) else cfg.assembly.filler_value


def _meets_quality_floor(
    cands: dict[str, Candidate], card: Card | None, min_quality: float
) -> bool:
    """Quality floor for the degradation ladder's replacement cards (R2/R3).

    Bounds only RANKED replacements: a card a source ranked (a consensus
    ``score`` exists) must score at least ``min_quality``. Unranked cards
    (flat filler prior, no measured quality) are exempt, mirroring the
    solver's filler semantics. ``min_quality <= 0`` disables the floor.
    """
    if min_quality <= 0:
        return True
    if card is None:
        return True
    c = cands.get(card.name)
    if c is None or c.score is None:
        return True
    return c.score >= min_quality


def _estimate_fallback(
    pricer: Pricer, card: Card | None, name: str, condition: str
) -> PriceInfo:
    """An estimated PriceInfo, safe when ``card`` is ``None``.

    ``Pricer.estimate`` dereferences ``card`` (and would raise ``AttributeError``
    on ``None``); for a card we have no Scryfall data for we synthesize an
    unpriced estimate instead.
    """
    if card is not None:
        return pricer.estimate(card)
    return PriceInfo(
        name=name,
        price_nzd=None,
        condition=condition,
        url="",
        stock=0,
        estimated=True,
        source="unpriced",
        note="no Scryfall card data; estimated price unavailable",
    )


def _in_stock(info: PriceInfo | None) -> bool:
    """A usable in-stock listing: present AND carrying a price.

    Both the real ``Pricer`` (returns ``None`` when OOS) and the test fake
    (returns ``PriceInfo(price_nzd=None)`` when OOS) are handled.
    """
    return info is not None and info.price_nzd is not None


def _lookup(
    pricer: Pricer, condition: str, verified: dict[str, PriceInfo], name: str
) -> PriceInfo | None:
    """Price ``name`` via the shared cache (in-stock hits cached, OOS not).

    A cache hit costs no MTGS call; a miss makes one sequential request and
    caches the result only when the card is in stock.
    """
    hit = verified.get(name)
    if hit is not None:
        return hit
    info = pricer.price_card(name, condition=condition)
    if _in_stock(info):
        verified[name] = info
    return info


# ------------------------------------------------------------------- verify
def _role_candidates(
    role: str,
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    idset: set[str],
    fx: float | None,
    used: set[str],
    exclude: frozenset[str],
) -> dict[str, tuple[Card, float, float]]:
    """All in-role alternatives for ``role`` as ``{name.casefold: (card, value, est)}``.

    Source-ranked cards keep their real scores; generic DB fillers use the
    flat filler value. For role "land" the pool is in-identity legal lands;
    otherwise in-identity legal non-lands matching the role tag. Cards already
    in ``used`` or ``exclude`` are not offered. Order-agnostic: callers sort.
    """
    seen: set[str] = set()
    items: dict[str, tuple[Card, float, float]] = {}

    def consider(card: Card | None) -> None:
        if card is None or not card.commander_legal:
            return
        if set(card.color_identity) - idset:
            return
        key = card.name.casefold()
        if key in seen or key in used or key in exclude:
            return
        if role == "land":
            if not card.is_land:
                return
        else:
            if card.is_land:
                return
            t = tag(card)
            if role == "ramp":
                ok = bool(t.get("ramp"))
            elif role == "draw":
                ok = bool(t.get("draw"))
            elif role == "removal":
                ok = bool(t.get("single_removal") or t.get("wipe") or t.get("counter"))
            else:  # theme / unknown: any non-land card
                ok = True
            if not ok:
                return
        seen.add(key)
        items[key] = (card, _value_of(cands, cfg, card), _est_price(card, fx))

    for name, cand in cands.items():
        consider(cand.card)
    for card in db.all:
        consider(card)
    return items


def _stock_alternatives(
    role: str,
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    idset: set[str],
    fx: float | None,
    used: set[str],
    exclude: frozenset[str],
) -> list[tuple[Card, float, float]]:
    """In-role alternatives for an OOS card, best-first, probe-capped.

    Sorted (value desc, est asc, name.casefold); capped at
    ``pricing.substitution.max_candidates``.
    """
    cap = max(1, int(cfg.pricing.substitution.max_candidates))
    items = _role_candidates(role, db, cands, cfg, idset, fx, used, exclude)
    ordered = sorted(items.values(), key=lambda p: (-p[1], p[2], p[0].name.casefold()))
    return ordered[:cap]


def _budget_alternatives(
    role: str,
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    idset: set[str],
    fx: float | None,
    used: set[str],
    exclude: frozenset[str],
    verified: dict[str, PriceInfo],
) -> list[tuple[Card, float, float]]:
    """In-role alternatives for the budget swap pass, cheapest-first.

    Verified (in-cache, therefore in-stock) candidates come first, ordered
    (verified price asc, name.casefold); unverified candidates follow, ordered
    (est asc, name.casefold). The swap pass takes the FIRST in-stock listing
    in this order, so a verified cheap alternative is never shadowed by an
    unprobed card, and an MTGS probe happens only when no verified candidate
    exists. Uncapped: a cheap alternative is never hidden behind the
    value-based probe cap.
    """
    items = _role_candidates(role, db, cands, cfg, idset, fx, used, exclude)
    known: list[tuple[Card, float, float]] = []
    unknown: list[tuple[Card, float, float]] = []
    for p in items.values():
        v = verified.get(p[0].name)
        if v is not None and v.price_nzd is not None:
            known.append(p)
        else:
            unknown.append(p)
    known.sort(key=lambda p: (verified[p[0].name].price_nzd, p[0].name.casefold()))
    unknown.sort(key=lambda p: (p[2], p[0].name.casefold()))
    return known + unknown


def _verify_deck(
    deck,
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    pricer: Pricer,
    fx: float | None,
    condition: str,
    exclude: frozenset[str],
    verified: dict[str, PriceInfo],
) -> tuple[dict[str, PriceInfo], list[Substitution], list[str]]:
    """Live-price every deck card (sequential, TTL-cached); fix OOS cards.

    Returns ``(prices, stock_subs, oos_excluded)`` and mutates ``deck`` in
    place: OOS non-commanders are replaced by the first in-stock alternative.
    Commanders are never replaced (logged as protected, priced estimated).

    ``verified`` is a shared price cache (in-stock hits only) so re-solve
    passes never re-price a card already confirmed in stock.
    """
    prices: dict[str, PriceInfo] = {}
    subs: list[Substitution] = []
    oos_excluded: list[str] = []
    used = {a.name.casefold() for a in deck.all_cards}
    idset = set(deck.identity)
    card_by_name = {c.name.casefold(): c for c in db.all}

    # -- non-commander assignments
    for i, a in enumerate(list(deck.assignments)):
        info = _lookup(pricer, condition, verified, a.name)
        if _in_stock(info):
            prices[a.name] = info
            continue

        # Out of stock (or no listing).
        est = _estimate_fallback(pricer, a.card, a.name, condition)
        oos_excluded.append(a.name)

        alts = _stock_alternatives(a.role, db, cands, cfg, idset, fx, used, exclude)
        probed: list[str] = []
        alt_name: str | None = None
        alt_info: PriceInfo | None = None
        for card, _v, _e in alts:
            probed.append(card.name)
            ainfo = _lookup(pricer, condition, verified, card.name)
            if _in_stock(ainfo):
                alt_name = card.name
                alt_info = ainfo
                break

        if alt_name is None:
            prices[a.name] = est
            subs.append(
                Substitution(
                    original=a.name,
                    replacement=NO_ALT,
                    role=a.role,
                    reason=f"stock: out of stock, no in-stock alternative (probed: {', '.join(probed)})",
                    score_delta=0.0,
                    price_original=est.price_nzd,
                    price_replacement=est.price_nzd,
                    estimated=True,
                )
            )
            continue

        # In-stock alternative: swap into the same slot.
        used.add(alt_name.casefold())
        alt_card = card_by_name.get(alt_name.casefold())
        new_score = _value_of(cands, cfg, alt_card)
        deck.assignments[i] = type(a)(
            name=alt_name,
            card=alt_card if alt_card is not None else a.card,
            role=a.role,
            package=a.package,
            score=new_score,
            note=f"stock substitution for {a.name}",
        )
        prices[alt_name] = alt_info
        subs.append(
            Substitution(
                original=a.name,
                replacement=alt_name,
                role=a.role,
                reason=f"stock: out of stock - first in-stock alternative (probed: {', '.join(probed)})",
                score_delta=round(new_score - a.score, 4),
                price_original=est.price_nzd,
                price_replacement=alt_info.price_nzd,
                estimated=False,
            )
        )

    # -- commanders (protected, never substituted)
    for c in deck.commanders:
        info = _lookup(pricer, condition, verified, c.name)
        if _in_stock(info):
            prices[c.name] = info
            continue
        est = _estimate_fallback(pricer, c.card, c.name, condition)
        prices[c.name] = est
        subs.append(
            Substitution(
                original=c.name,
                replacement=COMMANDER_PROTECTED,
                role="commander",
                reason="stock: out of stock (commander protected, kept at estimated price)",
                score_delta=0.0,
                price_original=est.price_nzd,
                price_replacement=est.price_nzd,
                estimated=True,
            )
        )

    return prices, subs, oos_excluded


# ------------------------------------------------------------------ pairing
def _budget_pair_subs(
    old: AssembleResult,
    new: AssembleResult,
    cands: dict[str, Candidate],
    cfg: Config,
    old_prices: dict[str, PriceInfo],
    fx: float | None,
) -> list[Substitution]:
    """Pair per-role dropped/added cards of a budget re-solve into log entries.

    For each role, cards dropped in the re-solve are paired (by index, both
    lists sorted by estimated price ascending then ``name.casefold`` — the
    price-based tie-break) with cards added; the per-pair saving is
    ``max(0, price_original - price_replacement)`` where the original uses its
    verified price (fallback: its estimate) and the replacement uses its
    estimate (it has not been live-verified yet).
    """
    subs: list[Substitution] = []
    old_by_name = {a.name.casefold(): a for a in old.deck.all_cards}
    new_by_name = {a.name.casefold(): a for a in new.deck.all_cards}

    for role in ("land", "ramp", "draw", "removal", "theme"):
        old_names = [a.name for a in old.deck.assignments if a.role == role]
        new_names = [a.name for a in new.deck.assignments if a.role == role]
        new_set = {n.casefold() for n in new_names}
        old_set = {n.casefold() for n in old_names}
        dropped = sorted(
            (n for n in old_names if n.casefold() not in new_set),
            key=lambda n: (old.est_prices.get(n) or 0.0, n.casefold()),
        )
        added = sorted(
            (n for n in new_names if n.casefold() not in old_set),
            key=lambda n: (_est_price(new_by_name[n.casefold()].card, fx) or 0.0, n.casefold()),
        )
        for old_name, new_name in zip(dropped, added):
            old_a = old_by_name[old_name.casefold()]
            new_a = new_by_name[new_name.casefold()]
            po = old_prices.get(old_name)
            price_original = po.price_nzd if (po is not None and po.price_nzd is not None) else (
                old.est_prices.get(old_name) or None
            )
            price_replacement = _est_price(new_a.card, fx) or None
            saving = round(max(0.0, (price_original or 0.0) - (price_replacement or 0.0)), 2)
            subs.append(
                Substitution(
                    original=old_name,
                    replacement=new_name,
                    role=role,
                    reason=f"budget: save NZ${saving:.2f}",
                    score_delta=round(new_a.score - old_a.score, 4),
                    price_original=price_original,
                    price_replacement=price_replacement,
                    estimated=True,
                )
            )
    return subs


# ---------------------------------------------------------------- budget swap
def _budget_swap_pass(
    deck,
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    pricer: Pricer,
    fx: float | None,
    condition: str,
    prices: dict[str, PriceInfo],
    verified: dict[str, PriceInfo],
    exclude: frozenset[str],
    target_cards_nzd: float,
    min_quality: float = 0.0,
) -> list[Substitution]:
    """Last-resort deterministic swaps that claw a verified deck under budget.

    Runs only when the re-solve budget is exhausted and the deck still blows
    the budget. Repeatedly applies the single most-saving swap: drop one
    verified-priced non-commander card, add the first in-stock in-role
    alternative that is cheaper.

    Per slot, alternatives come cheapest-first (see ``_budget_alternatives``):
    verified (cached, in-stock) candidates cost no MTGS call; unverified ones
    are probed only while their estimate is below the card they would replace
    (a probe can't reveal a saving otherwise), bounded at
    ``pricing.substitution.max_candidates`` probes per slot, and the scan
    stops at the first in-stock listing. ``min_quality`` bounds RANKED
    replacements (see ``_meets_quality_floor``); the check applies before
    any probe is spent on a candidate.

    Deterministic and terminating: the best swap is chosen by
    (-saving, name.casefold, slot); every applied swap strictly lowers the
    verified card total (saving > 0, prices are cent-quantized); the
    commander is never touched; ``used``/``exclude`` are respected.
    """
    subs: list[Substitution] = []
    idset = set(deck.identity)
    card_by_name = {c.name.casefold(): c for c in db.all}
    probe_cap = max(1, int(cfg.pricing.substitution.max_candidates))

    while cards_total(prices) > target_cards_nzd:
        used = {a.name.casefold() for a in deck.all_cards}
        best: tuple[tuple[float, str, int], str, int, Card, PriceInfo] | None = None

        for i, a in enumerate(deck.assignments):
            cur = prices.get(a.name)
            if cur is None or cur.price_nzd is None or a.role == "commander":
                continue
            probes = 0
            for card, _v, est in _budget_alternatives(
                a.role, db, cands, cfg, idset, fx, used, exclude, verified
            ):
                if not _meets_quality_floor(cands, card, min_quality):
                    continue
                v = verified.get(card.name)
                free = v is not None and v.price_nzd is not None
                if not free:
                    # Unknowns are est-ascending: nothing later can reveal a
                    # saving, and probes are capped.
                    if est >= cur.price_nzd or probes >= probe_cap:
                        break
                    probes += 1
                info = _lookup(pricer, condition, verified, card.name)
                if not _in_stock(info):
                    continue
                # The first in-stock listing settles this slot (verified
                # candidates are in-stock and price-ascending, so scanning
                # past one can never improve the slot).
                saving = round(cur.price_nzd - info.price_nzd, 2)
                if saving > 0:
                    key = (-saving, a.name.casefold(), i)
                    if best is None or key < best[0]:
                        best = (key, a.name, i, card, info)
                break

        if best is None:
            break  # no improving swap anywhere: infeasible

        _key, old_name, i, card, info = best
        a = deck.assignments[i]
        cur = prices[old_name]
        alt_card = card_by_name.get(card.name.casefold())
        new_score = _value_of(cands, cfg, card)
        deck.assignments[i] = type(a)(
            name=card.name,
            card=alt_card if alt_card is not None else a.card,
            role=a.role,
            package=a.package,
            score=new_score,
            note=f"budget swap for {a.name}",
        )
        del prices[old_name]
        prices[card.name] = info
        subs.append(
            Substitution(
                original=old_name,
                replacement=card.name,
                role=a.role,
                reason=f"budget: save NZ${round(cur.price_nzd - info.price_nzd, 2):.2f}",
                score_delta=round(new_score - a.score, 4),
                price_original=cur.price_nzd,
                price_replacement=info.price_nzd,
                estimated=False,
            )
        )

    return subs


# ------------------------------------------------------------- ladder rungs
def _rung_r1(
    deck,
    prices: dict[str, PriceInfo],
    verified: dict[str, PriceInfo],
    pricer: Pricer,
) -> list[Substitution]:
    """Rung R1: the cheapest in-stock printing of the SAME cards.

    The user constrained the buyable condition (e.g. ``near-mint``); this
    rung relaxes that constraint to the widest one (``any``) and re-prices
    every deck card, adopting a listing only when it is in stock AND strictly
    cheaper than the price currently charged for that card. The deck itself
    does not change - only the listing behind a price does.

    The MTGS response itself is condition-blind (the condition filter is
    client-side), so a card already verified this run is re-selected from
    cache at no extra MTGS request.
    """
    subs: list[Substitution] = []

    def try_reprice(name: str, role: str) -> None:
        cur = prices.get(name)
        if cur is None or cur.price_nzd is None:
            return
        info = pricer.price_card(name, condition="any")
        if not _in_stock(info):
            return
        saving = round(cur.price_nzd - info.price_nzd, 2)
        if saving <= 0:
            return
        prices[name] = info
        verified[name] = info
        subs.append(
            Substitution(
                original=name,
                replacement=name,
                role=role,
                reason=f"budget: R1 save NZ${saving:.2f}",
                score_delta=0.0,
                price_original=cur.price_nzd,
                price_replacement=info.price_nzd,
                estimated=False,
            )
        )

    for a in deck.assignments:
        try_reprice(a.name, a.role)
    for c in deck.commanders:
        try_reprice(c.name, "commander")
    return subs


_BASIC_BY_PIP = {
    "w": "plains",
    "u": "island",
    "b": "swamp",
    "r": "mountain",
    "g": "forest",
}


def _basic_matches_pips(card: Card, idset: set[str]) -> bool:
    """True when a basic land's name names one of the deck's identity pips."""
    name = card.name.casefold()
    idset = {c.casefold() for c in idset}
    return any(basic in name for pip, basic in _BASIC_BY_PIP.items() if pip in idset)


def _apply_rung_swap(
    deck,
    card_by_name: dict[str, Card],
    cands: dict[str, Candidate],
    cfg: Config,
    i: int,
    a,
    card: Card,
    info: PriceInfo,
    prices: dict[str, PriceInfo],
    used: set[str],
    rung: str,
) -> Substitution:
    """Swap slot ``i`` to ``card`` at its verified price; log the rung sub."""
    old_name = a.name
    cur = prices[old_name]
    alt_card = card_by_name.get(card.name.casefold())
    new_score = _value_of(cands, cfg, alt_card)
    deck.assignments[i] = type(a)(
        name=card.name,
        card=alt_card if alt_card is not None else a.card,
        role=a.role,
        package=a.package,
        score=new_score,
        note=f"{rung} degradation for {a.name}",
    )
    del prices[old_name]
    prices[card.name] = info
    used.add(card.name.casefold())
    return Substitution(
        original=old_name,
        replacement=card.name,
        role=a.role,
        reason=f"budget: {rung} save NZ${round(cur.price_nzd - info.price_nzd, 2):.2f}",
        score_delta=round(new_score - a.score, 4),
        price_original=cur.price_nzd,
        price_replacement=info.price_nzd,
        estimated=False,
    )


def _deck_wipes(deck) -> int:
    return sum(1 for a in deck.assignments if a.card is not None and tag(a.card).get("wipe"))


def _rung_r3_pass(
    deck,
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    fx: float | None,
    prices: dict[str, PriceInfo],
    verified: dict[str, PriceInfo],
    exclude: frozenset[str],
    target_cards_nzd: float,
    min_quality: float,
) -> list[Substitution]:
    """Rung R3: relax the non-structural quotas (verified cache only).

    (a) A theme slot may become ANY cheaper verified non-land card (theme
        fillers are interchangeable by construction; the theme quota is the
        soft one).
    (b) A non-basic land slot may become a cheaper verified BASIC land.

    Structural invariants are untouched: legality comes from
    ``_role_candidates`` (commander_legal, identity subset, not already in
    the deck, not excluded), the wipe cap (<= 2) is enforced, the singleton
    rule holds because a card is never offered twice. The quality floor
    (``min_quality``) bounds ranked replacements.

    Verified-only: a candidate is eligible only if it already carries a
    verified in-stock price - this rung never probes. One swap per slot,
    strictly cheaper, cheapest by (price, name.casefold); the pass stops as
    soon as the verified total is within ``target_cards_nzd``.
    """
    subs: list[Substitution] = []
    idset = set(deck.identity)
    card_by_name = {c.name.casefold(): c for c in db.all}
    used = {a.name.casefold() for a in deck.all_cards}
    wipes = _deck_wipes(deck)

    for i, a in enumerate(list(deck.assignments)):
        if cards_total(prices) <= target_cards_nzd:
            break
        if a.role not in ("theme", "land"):
            continue
        cur = prices.get(a.name)
        if cur is None or cur.price_nzd is None:
            continue
        if a.role == "land" and a.card is not None and a.card.is_basic_land:
            continue  # (b) applies to non-basic land slots only
        best: tuple[tuple[float, str], Card, PriceInfo] | None = None
        for card, _v, _e in _role_candidates(
            a.role, db, cands, cfg, idset, fx, used, exclude
        ).values():
            if a.role == "land" and not card.is_basic_land:
                continue
            info = verified.get(card.name)
            if info is None or info.price_nzd is None:
                continue  # verified-only pool
            if info.price_nzd >= cur.price_nzd:
                continue  # strictly cheaper only
            if not _meets_quality_floor(cands, card, min_quality):
                continue
            if a.role == "theme" and wipes >= 2 and tag(card).get("wipe"):
                continue  # never a third wipe
            key = (info.price_nzd, card.name.casefold())
            if best is None or key < best[0]:
                best = (key, card, info)
        if best is None:
            continue
        subs.append(_apply_rung_swap(
            deck, card_by_name, cands, cfg, i, a, best[1], best[2],
            prices, used, "R3",
        ))
        wipes = _deck_wipes(deck)
    return subs


def _rung_r4_pass(
    deck,
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    fx: float | None,
    prices: dict[str, PriceInfo],
    verified: dict[str, PriceInfo],
    exclude: frozenset[str],
    target_cards_nzd: float,
) -> list[Substitution]:
    """Rung R4: floor fill - the cheapest verified card per slot's role.

    Any non-commander slot may take the cheapest verified in-role card:
    ramp <- ramp, draw <- draw, removal <- removal, theme <- any non-land,
    land <- any land, preferring (on price ties) the basic that names one of
    the deck's identity pips, then ``name.casefold``. No quality bound (the
    floor is the floor): legality is ``_role_candidates`` plus the wipe cap.

    Verified-only, strictly cheaper, one swap per slot; stops as soon as the
    verified total is within ``target_cards_nzd``.
    """
    subs: list[Substitution] = []
    idset = set(deck.identity)
    card_by_name = {c.name.casefold(): c for c in db.all}
    used = {a.name.casefold() for a in deck.all_cards}
    wipes = _deck_wipes(deck)

    for i, a in enumerate(list(deck.assignments)):
        if cards_total(prices) <= target_cards_nzd:
            break
        cur = prices.get(a.name)
        if cur is None or cur.price_nzd is None:
            continue
        best: tuple[tuple[float, int, str], Card, PriceInfo] | None = None
        for card, _v, _e in _role_candidates(
            a.role, db, cands, cfg, idset, fx, used, exclude
        ).values():
            info = verified.get(card.name)
            if info is None or info.price_nzd is None:
                continue  # verified-only pool
            if info.price_nzd >= cur.price_nzd:
                continue  # strictly cheaper only
            if a.role != "land" and wipes >= 2 and tag(card).get("wipe"):
                continue  # never a third wipe
            pref = 0 if (a.role == "land" and card.is_basic_land
                         and _basic_matches_pips(card, idset)) else 1
            key = (info.price_nzd, pref, card.name.casefold())
            if best is None or key < best[0]:
                best = (key, card, info)
        if best is None:
            continue
        subs.append(_apply_rung_swap(
            deck, card_by_name, cands, cfg, i, a, best[1], best[2],
            prices, used, "R4",
        ))
        wipes = _deck_wipes(deck)
    return subs


# -------------------------------------------------------------------- solve
def resolve(
    asm: AssembleResult,
    db: ScryfallDB,
    cfg: Config,
    cands: dict[str, Candidate],
    commanders: list[Card],
    pricer: Pricer,
    *,
    budget_nzd: float | None = None,
    condition: str | None = None,
    themes: list[str] | None = None,
    companion: Card | None = None,
    fx: float | None = None,
    free_names: frozenset[str] | None = None,
) -> ResolveResult:
    """Verify the selected deck, then close budget/stock gaps (ladder).

    Pass: verify every deck card (MTGS, sequential, TTL-cached) -> substitute
    OOS non-commanders (first in-stock alternative, bounded probes) -> if the
    verified total (incl. shipping) is within budget, done. Otherwise the
    degradation ladder runs in fixed order, each rung logged only when it
    produced at least one change:

    * R1: re-price every deck card under the widest condition (only when the
      user constrained one), keeping a strictly cheaper in-stock listing of
      the SAME card;
    * R2: role-preserving budget pressure - warm-started re-solves at a
      strictly higher ``lambda`` (OOS cards excluded), at most
      ``assembly.max_resolves`` times, then the verified-price swap pass;
    * R3: relax the non-structural quotas (theme -> any cheaper verified
      non-land; non-basic land -> cheaper verified basic), verified cache
      only;
    * R4: floor fill - the cheapest verified card per slot's role, verified
      cache only.

    The last (closest) deck is reported with its gap; rungs are never
    applied out of order and no-op rungs are not logged.

    A single verified-price cache spans every pass, so an already-verified
    in-stock card is never priced twice.
    """
    condition = (condition or cfg.pricing.condition or "any").strip()
    verified: dict[str, PriceInfo] = {}
    subs: list[Substitution] = []
    oos_seen: set[str] = set()
    resolves = 0
    current = asm
    rungs: list[str] = []
    min_quality = cfg.budget.min_quality
    shipping = cfg.pricing.shipping_nzd
    r1_done = False
    # Config toggles: the quality-degrading rungs (R1-R3) are gated on
    # ``degradation_ladder``; the last-resort R4 floor fill on ``floor_fill``.
    ladder = cfg.budget.degradation_ladder
    floor_fill = cfg.budget.floor_fill

    def _budget_count() -> int:
        return sum(1 for s in subs if s.reason.startswith(_BUDGET_PREFIX))

    def _totals() -> tuple[float, float]:
        cards = cards_total(prices)
        return cards, round(cards + shipping, 2)

    # ------------------------------------------------- R1 -> R2 (re-solves)
    while True:
        exclude = frozenset(oos_seen)
        prices, stock_subs, oos = _verify_deck(
            current.deck, db, cands, cfg, pricer, fx, condition, exclude, verified
        )
        subs.extend(stock_subs)
        oos_seen |= {n.casefold() for n in oos}

        cards_total_nzd, total_nzd = _totals()
        if budget_nzd is None or total_nzd <= budget_nzd:
            break
        if not ladder:
            break  # ladder disabled: no R1-R3 budget pressure, report the gap

        # Rung R1 (once): relax the condition to the cheapest in-stock
        # printing of the same cards, before any card leaves the deck.
        if not r1_done:
            r1_done = True
            if condition not in ("any", ""):
                r1_subs = _rung_r1(current.deck, prices, verified, pricer)
                if r1_subs:
                    subs.extend(r1_subs)
                    rungs.append("R1")
                    cards_total_nzd, total_nzd = _totals()
                    if total_nzd <= budget_nzd:
                        break

        if resolves >= cfg.assembly.max_resolves:
            # Rung R2, last step: the verified-price swap pass (the
            # role-preserving, score-per-dollar rung at the floor of the
            # re-solve budget).
            n_before = _budget_count()
            target = budget_nzd - shipping
            swap_subs = _budget_swap_pass(
                current.deck, db, cands, cfg, pricer, fx, condition,
                prices, verified, exclude, target, min_quality=min_quality,
            )
            if swap_subs:
                subs.extend(swap_subs)
                cards_total_nzd, total_nzd = _totals()
            if _budget_count() > n_before and "R2" not in rungs:
                rungs.append("R2")
            break

        # Rung R2: budget re-solve - strictly higher lambda, OOS cards
        # excluded, and the solver priced against the verified prices we
        # already know (the accumulated cache, in-stock hits) instead of
        # MV x FX estimates.
        n_before = _budget_count()
        prev_lam = current.lambda_final
        min_lambda = max(prev_lam * 2.0, prev_lam + 1.0) if prev_lam > 0 else 1.0
        price_map = {
            k.casefold(): v.price_nzd for k, v in verified.items() if v.price_nzd is not None
        }
        new_asm = assemble(
            db,
            cfg,
            commanders,
            cands,
            themes=themes,
            companion=companion,
            budget_nzd=budget_nzd,
            fx=fx,
            min_lambda=min_lambda,
            exclude=exclude,
            price_map=price_map,
            free_names=free_names,
        )
        resolves += 1
        subs.extend(_budget_pair_subs(current, new_asm, cands, cfg, prices, fx))
        if _budget_count() > n_before and "R2" not in rungs:
            rungs.append("R2")
        current = new_asm

    # ------------------------------------------- R3 -> R4 (ladder tail)
    if budget_nzd is not None and total_nzd > budget_nzd:
        exclude = frozenset(oos_seen)
        target = budget_nzd - shipping
        # Rung R3: relax the non-structural quotas (verified cache only) -
        # gated on the degradation-ladder toggle.
        if ladder:
            r3_subs = _rung_r3_pass(
                current.deck, db, cands, cfg, fx, prices, verified,
                exclude, target, min_quality,
            )
            if r3_subs:
                subs.extend(r3_subs)
                rungs.append("R3")
                cards_total_nzd, total_nzd = _totals()

        # Rung R4: floor fill (verified cache only) - gated on the
        # floor-fill toggle.
        if floor_fill and total_nzd > budget_nzd:
            r4_subs = _rung_r4_pass(
                current.deck, db, cands, cfg, fx, prices, verified, exclude, target,
            )
            if r4_subs:
                subs.extend(r4_subs)
                rungs.append("R4")
                cards_total_nzd, total_nzd = _totals()

    # ------------------------------------------------------------ final return
    within = budget_nzd is None or total_nzd <= budget_nzd
    budget_subs = [s for s in subs if s.reason.startswith(_BUDGET_PREFIX)]
    return ResolveResult(
        asm=current,
        prices=prices,
        subs=subs,
        budget=BudgetResult(
            budget_nzd=budget_nzd,
            cards_total_nzd=cards_total_nzd,
            shipping_nzd=shipping,
            total_nzd=total_nzd,
            within_budget=within,
            gap_nzd=0.0 if within else round(total_nzd - budget_nzd, 2),
            swaps=budget_subs,
            resolves=resolves,
            degradation_rungs_applied=rungs,
        ),
        resolves=resolves,
        oos_excluded=sorted(oos_seen, key=str.casefold),
    )
