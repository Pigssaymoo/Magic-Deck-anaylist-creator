"""Deck assembly against the commandertemplate.com quota guide.

Quotas (default): 36 lands, 12 ramp, 10 draw, 10 removal (mostly single-target,
max 2 wipes), 32 theme (4 packages x 8, commanders included in the theme slot).
Lands flex 34-38 with curve/ramp density.

Phase 3 — joint slot<->card assignment (estimate-first):
  * Role slots (ramp/draw/removal) and theme slots are filled by a JOINT solver
    (``scipy.optimize.linear_sum_assignment``) over a truncated pool of
    source-ranked cards + DB fillers, maximizing ``sum(value - lambda*est_price)``.
  * A card's value is its consensus score (0..1.25) when a source ranked it,
    else a flat filler prior (``assembly.filler_value``, 0.05). ``est_price`` is
    ``mv_usd * FX`` (a single cached frankfurter rate); missing data -> 0.
  * The budget is enforced by a binary search on ``lambda``: doubling an upper
    bound until the estimated total (incl. shipping) fits, then bisecting to the
    boundary (or ``tolerance_nzd``). Higher ``lambda`` makes the solver prefer
    cheaper cards, so the search is monotone.
  * PROTECTED = commanders + the top-N highest-value theme cards are pre-assigned
    and never re-optimized; the solver fills the remaining theme slots.
  * Lands are chosen outside the solver (basics/fetches/duals/utility) and the
    land count is refined in a second pass from the ACTUAL non-land picks.
  * If scipy is absent (or ``assembly.solver.enabled`` is false) a deterministic
    greedy fill (best-cost first) is used instead.

Deterministic: stable sorts (value desc, est asc, name.casefold), no randomness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card
from .config import Config
from .roles import tag
from .scoring import Candidate
from .scryfall import ScryfallDB

DEFAULT_THEMES = ("Creatures", "Spells", "Enablers", "Utility")

# Cost-matrix penalties (solver path) are computed per-solve (see _penalties):
# a "generic filler" (an untagged card filling a role slot) and a padding
# (dummy row/column) must both cost MORE than any tagged assignment, whose cost
# is lam*est - value and grows with lam. Fixed sentinels break at high lam, so
# the penalties are derived from lam and the pool's max estimated price.


@dataclass
class Assignment:
    name: str
    card: Card
    role: str  # commander | land | ramp | draw | removal | theme
    package: str | None
    score: float
    note: str = ""


@dataclass
class Deck:
    commanders: list[Assignment] = field(default_factory=list)
    assignments: list[Assignment] = field(default_factory=list)
    identity: list[str] = field(default_factory=list)
    companion: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def all_cards(self) -> list[Assignment]:
        return self.commanders + self.assignments

    def names(self) -> list[str]:
        return [a.name for a in self.all_cards]


@dataclass
class AssembleResult:
    """Output of :func:`assemble` (estimate-first, before live verification)."""

    deck: Deck
    est_prices: dict[str, float]  # card name -> effective NZ$ (verified price if known, else mv_usd * FX)
    lambda_final: float
    solver_used: str  # "scipy.linear_sum_assignment" | "greedy-fallback"
    within_budget: bool
    budget_gap: float  # max(0, est_total_incl_shipping - budget)


# --------------------------------------------------------------------- helpers
def _sort_key(items: list[Candidate]) -> list[Candidate]:
    return sorted(items, key=lambda c: (-c.score, c.name.casefold()))


def _basic_land_name(color: str) -> str:
    return {
        "W": "Basic Plains Land",
        "U": "Basic Island Land",
        "B": "Basic Swamp Land",
        "R": "Basic Mountain Land",
        "G": "Basic Forest Land",
    }[color]


def _est_price(card: Card | None, fx: float | None) -> float:
    """Estimated NZ$ price = mv_usd * FX. Missing data -> 0 (budget-neutral)."""
    if card is None or card.mv_usd is None or fx is None:
        return 0.0
    return round(card.mv_usd * fx, 2)


def _price(card: Card | None, fx: float | None, price_map: dict[str, float] | None = None) -> float:
    """Effective solver price for a card.

    A verified (live-verified) NZ$ price from ``price_map`` (keyed by
    ``name.casefold()``) wins when present; otherwise fall back to the
    Scryfall-MV/FX estimate. This lets budget re-solves optimize against the
    ACTUAL prices we already know instead of re-guessing from MV.
    """
    if card is not None and price_map is not None:
        v = price_map.get(card.name.casefold())
        if v is not None:
            return v
    return _est_price(card, fx)


def _role_match(role: str, card: Card) -> bool:
    t = tag(card)
    if role == "ramp":
        return bool(t.get("ramp"))
    if role == "draw":
        return bool(t.get("draw"))
    if role == "removal":
        return bool(t.get("single_removal") or t.get("wipe") or t.get("counter"))
    return True  # theme / unknown: any non-land card


def _is_role_card(card: Card) -> bool:
    """True if the card is tagged for a quota role (ramp / draw / removal).

    Theme is the residual role ("everything else fills the theme packages"):
    a role-tagged card belongs in its role's slots, so it must never be
    pre-assigned to the protected theme block.
    """
    return _role_match("ramp", card) or _role_match("draw", card) or _role_match("removal", card)


def _have_scipy() -> bool:
    try:
        import scipy.optimize  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - scipy is an optional dependency
        return False


def _land_count_from_deck(cfg: Config, deck: Deck) -> int:
    """Pass-2 land count derived from the deck's ACTUAL non-land picks."""
    non_land = [a for a in deck.assignments if a.card is not None and not a.card.is_land]
    if not non_land:
        return cfg.quotas.lands_default
    avg_cmc = sum(a.card.cmc for a in non_land) / len(non_land)
    ramp_count = sum(1 for a in non_land if a.card is not None and tag(a.card).get("ramp"))
    if ramp_count >= 16 and avg_cmc <= 3.0:
        return cfg.quotas.lands_low
    if avg_cmc >= cfg.assembly.high_cmc_threshold:
        return cfg.quotas.lands_high
    return cfg.quotas.lands_default


def _justification(cand: Candidate) -> str:
    """One-line deterministic justification: scoring components + source ranks.

    Shows the consensus ``base`` (shrinkage blend before multipliers), the
    source ``coverage`` (weight of present sources / weight of all configured),
    the pool-relative ``theme_fit`` (when a theme was active), and the raw
    per-source ranks. No lift-bonus line: lift is not a scoring input.
    """
    bits: list[str] = [f"base {cand.base:.3f}", f"coverage {cand.coverage:.2f}"]
    if cand.theme_fit is not None:
        bits.append(f"theme_fit {cand.theme_fit:+.3f}")
    er = cand.signals.get("edhrec")
    if er is not None:
        bits.append(f"EDHREC rank {er.rank}/{er.n}")
    rc = cand.signals.get("recommander")
    if rc is not None:
        bits.append(f"Recommander rank {rc.rank}/{rc.n}")
    et = cand.signals.get("edhtools")
    if et is not None:
        bits.append(f"edh.tools rank {et.rank}/{et.n}")
    return " · ".join(bits)


def _theme_package(card: Card, themes: list[str]) -> str:
    t = card.type_line.lower()
    if "creature" in t:
        return themes[0]
    if "instant" in t or "sorcery" in t:
        return themes[1]
    if "enchantment" in t or "planeswalker" in t:
        return themes[2]
    return themes[3]


# --------------------------------------------------------------------- lands
def _build_lands(
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    identity: list[str],
    idset: set[str],
    lands_n: int,
    exclude: frozenset[str] = frozenset(),
) -> tuple[list[Assignment], set[str]]:
    """Deterministic land build: basics per color, fetches, duals, utility fill.

    Returns (land assignments, the set of casefolded names consumed).
    ``exclude`` (casefolded names) is never picked (e.g. OOS cards from a
    previous verify pass).
    """
    land_pool: dict[str, Candidate] = {}
    for c in cands.values():
        if (
            c.card is not None
            and c.card.is_land
            and set(c.card.color_identity) <= idset
            and c.name.casefold() not in exclude
        ):
            land_pool[c.name.casefold()] = c
    # Full legal-land fallback from Scryfall (covers source failures).
    for card in db.all:
        if (
            card.is_land
            and card.commander_legal
            and set(card.color_identity) <= idset
            and card.name.casefold() not in exclude
        ):
            key = card.name.casefold()
            if key not in land_pool:
                land_pool[key] = Candidate(name=card.name, card=card, roles=tag(card), score=0.0)

    lands: list[Assignment] = []
    used: set[str] = set()

    def add(entry: Candidate, note: str) -> None:
        if len(lands) >= lands_n:
            return
        if entry.name.casefold() in used:
            return
        used.add(entry.name.casefold())
        lands.append(
            Assignment(name=entry.name, card=entry.card, role="land", package=None, score=entry.score, note=note)
        )

    # Basics, one per identity color
    for color in identity:
        basic_name = _basic_land_name(color)
        hit = db.by_name.get(basic_name.casefold())
        if hit is None:
            continue
        entry = land_pool.get(basic_name.casefold()) or Candidate(
            name=hit.name, card=hit, roles=tag(hit), score=0.0
        )
        add(entry, "basic land")

    # Fetches / basic-land tutors (land-tagged ramp)
    fetches = [
        c
        for c in land_pool.values()
        if not c.card.is_basic_land and c.roles.get("ramp") and set(c.card.color_identity) <= idset
    ]
    for c in _sort_key(fetches)[:2]:
        add(c, "mana fixity fetch")

    # Duals / identity lands within budget
    dual_budget = cfg.assembly.dual_land_budget.get(len(identity), 20)
    duals = [
        c
        for c in land_pool.values()
        if not c.card.is_basic_land
        and len(set(c.card.colors)) == min(2, max(1, len(identity)))
        and set(c.card.colors) <= idset
    ]
    for c in _sort_key(duals)[:dual_budget]:
        if len(lands) >= lands_n:
            break
        add(c, "dual/identity land")

    # Fill remainder with best-scoring utility lands
    rest = [c for c in land_pool.values() if c.name.casefold() not in used]
    for c in _sort_key(rest):
        if len(lands) >= lands_n:
            break
        add(c, "utility land fill")

    # Last-resort basics of any color (colorless/snow edge cases)
    for c in _sort_key(land_pool.values()):
        if len(lands) >= lands_n:
            break
        add(c, "land fill (fallback)")

    return lands, used


# ------------------------------------------------------------- solver / pools
def _all_noland(
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    idset: set[str],
    fx: float | None,
    exclude: frozenset[str] = frozenset(),
    price_map: dict[str, float] | None = None,
) -> list[tuple[Card, float, float]]:
    """All Commander-legal, in-identity, non-land cards as (card, value, est).

    value = consensus score (0..1.25) for source-ranked cards, else the filler
    prior; est = mv_usd * FX (0 if unknown), or a verified price from
    ``price_map`` when one is known. Deterministic dedup by name.
    ``exclude`` (casefolded names) is never offered.
    """
    filler_value = cfg.assembly.filler_value
    out: dict[str, tuple[Card, float, float]] = {}

    def consider(card: Card | None) -> None:
        if card is None or not card.commander_legal or card.is_land:
            return
        if set(card.color_identity) - idset:
            return
        key = card.name.casefold()
        if key in out or key in exclude:
            return
        c = cands.get(card.name)
        value = c.score if (c is not None and c.score is not None) else filler_value
        out[key] = (card, value, _price(card, fx, price_map))

    for name, cand in cands.items():
        if cand.card is not None:
            consider(cand.card)
    for card in db.all:
        consider(card)
    return list(out.values())


def _pool_sort_key(item: tuple[Card, float, float]) -> tuple[float, float, str]:
    card, value, est_v = item
    return (-value, est_v, card.name.casefold())


def _solver_pool(
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    idset: set[str],
    fx: float | None,
    exclude: frozenset[str] = frozenset(),
    price_map: dict[str, float] | None = None,
    max_est: float | None = None,
) -> list[tuple[Card, float, float]]:
    """Truncated solver column pool: top-N per role + top theme, unioned.

    Bounds the assignment problem while guaranteeing each role has candidates.
    A value-only truncation can drop EVERY near-free card from a column (cheap
    fillers rank last by score), after which no lambda penalty can make the
    deck affordable. So each column also retains its ``cheap_pool_size``
    cheapest legal options by effective price - derived from the local feed,
    never hardcoded names; ties break on ``name.casefold()``.

    ``max_est`` is the per-slot price ceiling (``budget x
    pricing.max_single_card_fraction``): value-truncated candidates whose
    effective price exceeds it are pruned BEFORE solving, so one card can
    never eat ``1/fraction`` of the budget. A tiny relative slack absorbs
    float noise so a card exactly at the ceiling survives. The cheap tail is
    exempt - if even the cheapest legal option of a role clears the ceiling,
    that role still needs candidates (a legal 100-card deck beats the
    single-card guardrail; the budget search still caps the TOTAL).
    """
    allpool = _all_noland(db, cands, cfg, idset, fx, exclude, price_map)
    cheap_k = max(0, int(cfg.assembly.cheap_pool_size))

    def role_items(role: str) -> list[tuple[Card, float, float]]:
        return allpool if role == "theme" else [p for p in allpool if _role_match(role, p[0])]

    def top_by_value(items: list[tuple[Card, float, float]], limit: int) -> list[tuple[Card, float, float]]:
        return sorted(items, key=_pool_sort_key)[:limit]

    def cheapest(items: list[tuple[Card, float, float]], k: int) -> list[tuple[Card, float, float]]:
        return sorted(items, key=lambda p: (p[2], p[0].name.casefold()))[:k]

    slack = max_est * 1e-9 if max_est is not None else 0.0

    def under_ceiling(item: tuple[Card, float, float]) -> bool:
        return max_est is None or item[2] <= max_est + slack

    union: dict[str, tuple[Card, float, float]] = {}
    for role, value_limit in (("ramp", 40), ("draw", 40), ("removal", 40), ("theme", 100)):
        items = role_items(role)
        tail = cheapest(items, cheap_k) if cheap_k else []
        tail_names = {t[0].name.casefold() for t in tail}
        value_top = [i for i in top_by_value(items, value_limit) if under_ceiling(i) or i[0].name.casefold() in tail_names]
        for item in value_top + tail:
            union[item[0].name.casefold()] = item
    return list(union.values())


def _protected_theme(
    db: ScryfallDB,
    cands: dict[str, Candidate],
    cfg: Config,
    idset: set[str],
    used: set[str],
    fx: float | None,
    n: int,
    exclude: frozenset[str] = frozenset(),
    price_map: dict[str, float] | None = None,
) -> list[tuple[Card, float, float]]:
    """Top-N highest-value theme cards (pre-assigned, never re-optimized).

    Only untagged ("everything else") cards qualify. A card tagged for a
    quota role belongs in that role's slots: pre-assigning it to the theme
    block would steal it from the solver pool, and under estimate ties the
    role rows would then be filled with filler, shuffling unrelated cards
    into role slots and corrupting the re-solve pairing.
    """
    if n <= 0:
        return []
    pool = [
        p
        for p in _all_noland(db, cands, cfg, idset, fx, exclude, price_map)
        if p[0].name.casefold() not in used and not _is_role_card(p[0])
    ]
    pool.sort(key=_pool_sort_key)
    return pool[:n]


# ---------------------------------------------------------------------- solve
def _penalties(lam: float, max_est: float) -> tuple[float, float]:
    """Return (filler, padding) costs that dominate any tagged assignment.

    A tagged pair costs ``lam*est - value`` (<= ``lam*max_est``). A filler pair
    (untagged card in a role slot) keeps its real economics and adds a penalty
    of ``lam*max_est + 2`` so it is always pricier than any tagged card, while
    still preferring the cheapest filler. Padding (dummy rows/cols) sits above
    every real and filler cost so it is only used when a row has no column.
    """
    base = lam * max_est
    filler = base + 2.0
    padding = 2.0 * base + 4.0
    return filler, padding


def _cost_for(role: str, card: Card, value: float, est_v: float, lam: float, filler: float) -> float:
    if role == "theme" or _role_match(role, card):
        return -(value - lam * est_v)
    return (lam * est_v - value) + filler  # generic filler: real cost + dominance penalty


def _solve_scipy(rows: list[str], cols: list[tuple[Card, float, float]], lam: float) -> dict[int, int]:
    import numpy as np

    from scipy.optimize import linear_sum_assignment

    n, m = len(rows), len(cols)
    max_est = max((est for _c, _v, est in cols), default=0.0)
    filler, padding = _penalties(lam, max_est)
    cost = np.full((n, m), filler, dtype=float)
    for i, role in enumerate(rows):
        for j, (card, value, est_v) in enumerate(cols):
            cost[i, j] = _cost_for(role, card, value, est_v, lam, filler)
    size = max(n, m)
    padded = np.full((size, size), padding, dtype=float)
    padded[:n, :m] = cost
    row_ind, col_ind = linear_sum_assignment(padded)
    assignment: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            assignment[int(r)] = int(c)
    return assignment


def _solve_greedy(rows: list[str], cols: list[tuple[Card, float, float]], lam: float) -> dict[int, int]:
    """Deterministic fallback: assign (slot, card) pairs best-cost-first."""
    max_est = max((est for _c, _v, est in cols), default=0.0)
    filler, _padding = _penalties(lam, max_est)
    pairs: list[tuple[float, float, str, int, int]] = []
    for i, role in enumerate(rows):
        for j, (card, value, est_v) in enumerate(cols):
            pairs.append((_cost_for(role, card, value, est_v, lam, filler), est_v, card.name.casefold(), i, j))
    pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    used_r: set[int] = set()
    used_c: set[int] = set()
    assignment: dict[int, int] = {}
    for _c, _e, _name, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        assignment[i] = j
    return assignment


def _solve(rows: list[str], cols: list[tuple[Card, float, float]], lam: float, cfg: Config) -> tuple[dict[int, int], str]:
    if cfg.assembly.solver.enabled and _have_scipy():
        return _solve_scipy(rows, cols, lam), "scipy.linear_sum_assignment"
    return _solve_greedy(rows, cols, lam), "greedy-fallback"


# -------------------------------------------------------------- budget search
def _budget_search(
    rows: list[str],
    cols: list[tuple[Card, float, float]],
    cfg: Config,
    budget_nzd: float | None,
    protected_est_total: float,
    min_lambda: float,
    solve,
) -> tuple[float, dict[int, int], bool, float]:
    """Binary-search lambda so the estimated total fits the budget.

    Returns (lambda_final, assignment, within_budget, gap_nzd). ``min_lambda``
    warm-starts the lower bound (used by the estimate->verify re-solve loop).
    """
    shipping = cfg.pricing.shipping_nzd
    cache: dict[float, dict[int, int]] = {}

    def total_cards(lam: float) -> float:
        if lam not in cache:
            cache[lam] = solve(lam)
        pairs = cache[lam]
        t = protected_est_total
        for _r, c in pairs.items():
            t += cols[c][2]
        return round(t, 2)

    if budget_nzd is None:
        lam = 0.0
        return lam, solve(0.0), True, 0.0

    lo = max(0.0, min_lambda)
    if total_cards(lo) + shipping <= budget_nzd:
        return lo, cache.get(lo) or solve(lo), True, 0.0

    hi = max(lo, 1.0)
    found = False
    for _ in range(cfg.assembly.solver.max_doublings):
        if total_cards(hi) + shipping <= budget_nzd:
            found = True
            break
        lo = hi
        hi *= 2.0
    if not found:
        lam = hi
        return lam, solve(lam), False, round(total_cards(lam) + shipping - budget_nzd, 2)

    for _ in range(cfg.assembly.solver.bisections):
        mid = (lo + hi) / 2
        if total_cards(mid) + shipping <= budget_nzd:
            hi = mid
        else:
            lo = mid
        if (budget_nzd - (total_cards(hi) + shipping)) < cfg.assembly.solver.tolerance_nzd:
            break
    lam = hi
    return lam, solve(lam), True, 0.0


# -------------------------------------------------------------------- assemble
def _assemble_once(
    db: ScryfallDB,
    cfg: Config,
    commanders: list[Card],
    cands: dict[str, Candidate],
    themes: list[str],
    companion: Card | None,
    idset: set[str],
    identity: list[str],
    lands_n: int,
    budget_nzd: float | None,
    fx: float | None,
    min_lambda: float,
    exclude: frozenset[str] = frozenset(),
    price_map: dict[str, float] | None = None,
) -> AssembleResult:
    quotas = cfg.quotas
    ramp_n = quotas.ramp
    draw_n = quotas.draw
    removal_n = quotas.removal
    theme_n = 100 - lands_n - ramp_n - draw_n - removal_n - len(commanders)

    deck = Deck(identity=identity, companion=companion.name if companion else None)

    # ----------------------------------------------------------- commanders
    for cmd in commanders:
        deck.commanders.append(
            Assignment(
                name=cmd.name,
                card=cmd,
                role="commander",
                package=themes[0],
                score=1.0,
                note="commander",
            )
        )

    used: set[str] = {cmd.name.casefold() for cmd in commanders}
    if companion is not None:
        used.add(companion.name.casefold())

    # --------------------------------------------------------------- lands
    lands, land_used = _build_lands(db, cands, cfg, identity, idset, lands_n, exclude)
    used |= land_used
    deck.assignments = list(lands)

    # ------------------------------------------------- protected theme (top-N)
    prot_n = min(cfg.budget.protect_theme_top, max(0, theme_n))
    protected = _protected_theme(db, cands, cfg, idset, used, fx, prot_n, exclude, price_map)
    # Fixed cost = commanders + companion + lands + top-N theme (all pre-assigned,
    # outside the solver); the budget search penalizes only the solved slots.
    # The companion is outside the 100 but still bought and paid for, so it is
    # reserved with the rest of the basket. Lands are counted here too: they are
    # part of the deck the buyer pays for, so leaving them out silently
    # overstates the budget headroom by the land cost.
    protected_est_total = sum(_price(c, fx, price_map) for c in commanders)
    if companion is not None:
        protected_est_total += _price(companion, fx, price_map)
    protected_est_total += sum(_price(a.card, fx, price_map) for a in lands)
    protected_assignments: list[Assignment] = []
    for card, value, est_v in protected:
        used.add(card.name.casefold())
        protected_est_total += est_v
        protected_assignments.append(
            Assignment(
                name=card.name,
                card=card,
                role="theme",
                package=_theme_package(card, themes),
                score=value,
                note="protected theme (top value)",
            )
        )

    # ------------------------------------------------------------- role slots
    theme_slots = max(0, theme_n - len(protected))
    rows: list[str] = (
        ["ramp"] * ramp_n + ["draw"] * draw_n + ["removal"] * removal_n + ["theme"] * theme_slots
    )
    # Per-slot price ceiling: with commander + companion + protected theme +
    # lands + shipping already reserved, no single solved slot may be worth
    # more than ``max_single_card_fraction`` of the budget (default 15%) -
    # otherwise one card eats the basket and the remaining slots cannot be
    # filled legally. Candidates above it are pruned from the pool BEFORE
    # solving; the budget search still enforces the total.
    max_est: float | None = None
    if budget_nzd is not None and cfg.pricing.max_single_card_fraction > 0:
        max_est = budget_nzd * cfg.pricing.max_single_card_fraction

    # Commanders and protected cards are already placed (and counted in
    # protected_est_total), so they must stay OUT of the solver pool: a
    # one-to-one matching would otherwise hand them to a role/theme row too
    # and the deck would contain duplicates.
    cols = _solver_pool(
        db,
        cands,
        cfg,
        idset,
        fx,
        exclude
        | {cmd.name.casefold() for cmd in commanders}
        | {a.name.casefold() for a in protected_assignments},
        price_map,
        max_est,
    )

    def solve(lam: float) -> dict[int, int]:
        return _solve(rows, cols, lam, cfg)[0]

    lam, pairs, within, gap = _budget_search(rows, cols, cfg, budget_nzd, protected_est_total, min_lambda, solve)
    solver_used = "scipy.linear_sum_assignment" if (cfg.assembly.solver.enabled and _have_scipy()) else "greedy-fallback"

    # ----------------------------------------------------- assign role/theme
    assigned: list[Assignment] = []
    for i, role in enumerate(rows):
        j = pairs.get(i)
        if j is None:
            continue
        card, value, est_v = cols[j]
        note = _justification(cands[card.name]) if card.name in cands else "generic legal fill (not in any source response)"
        if role == "theme":
            assigned.append(
                Assignment(
                    name=card.name,
                    card=card,
                    role="theme",
                    package=_theme_package(card, themes),
                    score=value,
                    note=note,
                )
            )
        else:
            assigned.append(
                Assignment(
                    name=card.name,
                    card=card,
                    role=role,
                    package=None,
                    score=value,
                    note=note,
                )
            )

    # Preserve the template's role order: lands, removal, draw, ramp, theme
    # (protected theme pieces come first within the theme block).
    removal = [a for a in assigned if a.role == "removal"]
    draw = [a for a in assigned if a.role == "draw"]
    ramp = [a for a in assigned if a.role == "ramp"]
    theme = protected_assignments + [a for a in assigned if a.role == "theme"]
    deck.assignments = list(lands) + removal + draw + ramp + theme

    # ------------------------------------------------------------- est prices
    # Effective per-card price the solver worked with: verified (price_map)
    # where known, MV x FX estimate otherwise.
    est_prices: dict[str, float] = {}
    for a in deck.all_cards:
        est_prices[a.name] = _price(a.card, fx, price_map)

    deck.meta = {
        "quotas": {
            "lands": (lands_n, len(lands)),
            "ramp": (ramp_n, len(ramp)),
            "draw": (draw_n, len(draw)),
            "removal": (removal_n, len(removal)),
            "theme": (theme_n + len(commanders), len(theme) + len(commanders)),
        },
        "themes": themes,
        "land_count_reason": "curve/ramp density heuristic (two-pass: derived from actual non-land picks)",
        "lambda_final": lam,
        "solver_used": solver_used,
        # Per-card effective price (verified where known, MV x FX otherwise).
        "est_prices": est_prices,
    }

    return AssembleResult(
        deck=deck,
        est_prices=est_prices,
        lambda_final=lam,
        solver_used=solver_used,
        within_budget=within,
        budget_gap=gap,
    )


def assemble(
    db: ScryfallDB,
    cfg: Config,
    commanders: list[Card],
    cands: dict[str, Candidate],
    themes: list[str] | None = None,
    companion: Card | None = None,
    budget_nzd: float | None = None,
    fx: float | None = None,
    min_lambda: float = 0.0,
    exclude: frozenset[str] | None = None,
    price_map: dict[str, float] | None = None,
    free_names: frozenset[str] | None = None,
) -> AssembleResult:
    """Assemble a 100-card deck (estimate-first).

    Two-pass lands: solve at ``lands_default`` (36), then re-derive the land
    count (34/36/38) from the ACTUAL non-land picks and re-solve once if it
    changed. ``fx`` is the USD->NZD rate (``None`` disables estimates, lambda
    stays 0). ``min_lambda`` warm-starts the budget search (re-solve loop).
    ``exclude`` is a set of casefolded card names to keep out of every pool
    (e.g. out-of-stock cards found during verification). ``price_map`` maps
    ``name.casefold()`` -> verified NZ$ price; when given, those prices replace
    the MV x FX estimates in the solver and the budget search, so re-solves
    optimize against the live prices we already know. ``free_names`` (casefolded
    card names the user already owns) are priced at NZ$0.00 in the solver: they
    are merged into ``price_map`` (which wins over MV x FX), so the budget is
    spent only on cards that must be bought.
    """
    exclude = frozenset((n.casefold() for n in exclude) if exclude else ())
    if free_names:
        pm = dict(price_map) if price_map is not None else {}
        for n in free_names:
            pm[n.casefold()] = 0.0
        price_map = pm
    themes = list(themes) if themes and len(themes) == 4 else list(DEFAULT_THEMES)
    identity: list[str] = []
    for c in commanders:
        for color in c.color_identity:
            if color not in identity:
                identity.append(color)
    idset = set(identity)

    # Pass 1: default land count.
    lands_n = cfg.quotas.lands_default
    result = _assemble_once(
        db, cfg, commanders, cands, themes, companion, idset, identity, lands_n,
        budget_nzd, fx, min_lambda, exclude, price_map,
    )

    # Pass 2: derive land count from the actual non-land picks; re-solve once.
    lands_n2 = _land_count_from_deck(cfg, result.deck)
    if lands_n2 != lands_n:
        result = _assemble_once(
            db, cfg, commanders, cands, themes, companion, idset, identity, lands_n2,
            budget_nzd, fx, min_lambda, exclude, price_map,
        )

    return result
