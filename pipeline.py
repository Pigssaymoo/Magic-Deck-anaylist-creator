"""End-to-end pipeline: resolve -> fetch -> score -> assemble -> price -> budget -> outputs.

Graceful degradation: each recommendation source is fetched independently; a
failure is recorded in ``source_status`` and the deck is still built from the
remaining sources plus the Scryfall legal pool. The pipeline is deterministic
given the same cached data (stable sorts, no randomness).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .assembly import DEFAULT_THEMES, Deck, assemble
from .budget import BudgetResult, Substitution, resolve
from .cards import Card
from .commander import (
    CommanderScore,
    candidates_from_search,
    candidates_from_theme_page,
    fetch_commander_page,
    score_commanders,
    winner_commanders,
)
from .config import Config
from .edhrec import EdhrecData, resolve_theme_page, slugify, fetch as fetch_edhrec
from .edhtools import EdhToolsData, fetch as fetch_edhtools
from .floor import FloorBreakdown, FloorInfeasible, compute_floor
from .httpclient import Http, SourceBlocked
from .pricing import Pricer, PriceInfo
from .recommander import RecommanderData, fetch as fetch_recommander
from .report import write_outputs
from .roles import tag as tag_roles
from .scoring import Candidate, consensus_score, merge_candidates
from .scryfall import ScryfallDB, find_card, find_commander, tag_search


@dataclass
class CommanderSelection:
    """Theme-mode commander auto-selection result (explained in report.md)."""

    slug: str  # theme slug (tag mode) or the Scryfall tag query (tags mode)
    theme_names: list[str] = field(default_factory=list)
    required_colors: list[str] = field(default_factory=list)
    mode: str = "tag"  # "tag" | "tags"
    winner_name: str = ""
    winner: CommanderScore | None = None
    runners: list[CommanderScore] = field(default_factory=list)
    budget_nzd: float | None = None


@dataclass
class BuildResult:
    deck: Deck
    cands: dict[str, Candidate] = field(default_factory=dict)
    prices: dict[str, PriceInfo] = field(default_factory=dict)
    stock_subs: list[Substitution] = field(default_factory=list)
    budget: BudgetResult | None = None
    source_status: dict[str, str] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)
    commanders: list[Card] = field(default_factory=list)
    elapsed_sec: float = 0.0
    selection: CommanderSelection | None = None
    # Phase 3: assembly + pricing stats (estimate-first, verify-last).
    lambda_final: float = 0.0
    solver_used: str = ""
    estimated_count: int = 0
    verified_count: int = 0
    resolves: int = 0


def _fetch_source(status: dict[str, str], name: str, fn) -> object | None:
    try:
        data = fn()
        n = len(getattr(data, "signals", {})) if data is not None else 0
        status[name] = f"ok ({n} cards)" if n else "ok (empty response)"
        return data
    except SourceBlocked as exc:
        status[name] = f"degraded: {exc}"
        return None
    except Exception as exc:  # noqa: BLE001 - never let one source kill the build
        status[name] = f"degraded: {type(exc).__name__}: {exc}"
        return None


def _resolve_commanders(db: ScryfallDB, cfg: Config, names: list[str]) -> list[Card]:
    resolved: list[Card] = []
    for name in names:
        cmd = find_commander(db, name, cfg.assembly.fuzzy_match_threshold)
        if cmd is None:
            raise SystemExit(
                f"could not resolve commander {name!r} (not a legal Commander card in the Scryfall bulk data)"
            )
        if not cmd.commander_legal:
            raise SystemExit(f"commander {cmd.name!r} is not Commander (OCG) legal")
        if any(cmd.name == r.name for r in resolved):
            raise SystemExit(f"commander {name!r} repeated")
        resolved.append(cmd)
    return resolved


def _detect_companion(db: ScryfallDB, commanders: list[Card], companion_name: str | None) -> Card | None:
    """Resolve the deck's Companion (recorded outside the 100-card limit).

    An explicit ``companion_name`` always wins and must carry the Companion
    ability. Otherwise this is best-effort: if a commander's oracle text
    mentions a Companion, look for a named card reference in that text that
    carries the Companion ability. Returns None when undetermined (a deck is
    allowed to have no Companion).
    """
    if companion_name:
        card = find_card(db, companion_name)
        if card is None:
            raise SystemExit(f"companion {companion_name!r} could not be resolved")
        if not card.is_companion:
            raise SystemExit(f"{card.name!r} does not have the Companion ability")
        return card
    for cmd in commanders:
        if "companion" not in cmd.oracle_text.lower():
            continue
        # Best-effort: a Companion is a separate card, so only accept one that
        # is *named* in the commander's text and carries the ability.
        best: Card | None = None
        for name, card in db.by_name.items():
            if card.is_companion and len(name) >= 3 and name in cmd.oracle_text.casefold():
                if best is None or len(name) > len(best.name):
                    best = card
        return best
    return None


def _enforce_budget(
    budget: BudgetResult | None,
    budget_nzd: float | None,
    resolves: int,
    on_budget_fail: str = "degrade",
) -> None:
    """Budget policy for an over-budget result (``degrade`` | ``fail`` | ``emit``).

    The budget is a cap, not a target. ``resolve`` runs the degradation
    ladder (R1 condition relaxation, R2 role-preserving re-solves + swap
    pass, R3 quota relaxation, R4 floor fill) and reports the closest deck
    with its gap. When that deck STILL exceeds the budget:

    * ``degrade`` (default) - the ladder is exhausted; fail the build (exit
      1) with a message pointing at ``--on-budget-fail emit``;
    * ``fail`` - the legacy hard gate: same refusal, worded as the budget's
      own impossibility;
    * ``emit`` - accept the over-budget deck: no raise; the deck is written
      with ``within_budget: false`` and the caller prints a loud banner.

    In every mode an over-budget deck is never shipped silently.
    """
    if budget_nzd is None or budget is None or budget.within_budget:
        return
    if on_budget_fail == "emit":
        return
    if on_budget_fail == "fail":
        raise SystemExit(
            f"budget of NZ${budget_nzd:,.2f} could not be met: the closest deck "
            f"costs NZ${budget.total_nzd:,.2f} (NZ${budget.gap_nzd:,.2f} over) after "
            f"{resolves} re-solve(s) and stock substitution. The budget is a hard "
            f"cap, so no over-budget deck was written - raise --budget or pick a "
            f"cheaper commander."
        )
    # degrade (default): the ladder ran to exhaustion and could not close the gap.
    raise SystemExit(
        f"degradation ladder exhausted: final basket NZ${budget.total_nzd:,.2f} "
        f"exceeds budget NZ${budget_nzd:,.2f} by NZ${budget.gap_nzd:,.2f}; "
        f"re-run with --on-budget-fail emit to accept the over-budget deck"
    )


def _merge_theme_pool(
    db: ScryfallDB,
    cfg: Config,
    identity: list[str],
    cands: dict[str, Candidate],
    pool: dict[str, tuple[float, float]],  # name -> (synergy, lift)
    commander_names: set[str],
) -> int:
    """Attach a pool-relative ``theme_fit`` modifier to candidates.

    Theme affinity is a MODIFIER on the consensus value, not a pseudo-source:
    each pool card gets ``theme_fit`` = pool-relative percentile of its
    (synergy, lift), so ``consensus_score`` multiplies its value by
    (1 + theme_fit_beta * theme_fit). A pool with no synergy data at all (raw
    tag search) is neutral: candidates keep ``theme_fit = None`` and the score
    is unchanged (factor 1.0). Deterministic: same pool -> same fit values.
    """
    idset = set(identity)
    n = len(pool)
    if n == 0:
        return 0
    has_signal = any(syn > 0 or lift > 0 for syn, lift in pool.values())
    if has_signal:
        ordered = sorted(pool.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0].casefold()))
        rank = {nm: i + 1 for i, (nm, _v) in enumerate(ordered)}
        fit = {nm: (n - rank[nm] + 1) / n for nm in pool}
    else:  # no synergy data (raw tag search) -> neutral, no modifier
        fit = {nm: None for nm in pool}

    merged = 0
    touched: list[Candidate] = []
    for raw_name in pool:
        card = find_card(db, raw_name, threshold=0.9)
        if card is None or not card.commander_legal or not set(card.color_identity) <= idset:
            continue
        canon = card.name
        if canon.casefold() in commander_names:
            continue
        c = cands.get(canon)
        if c is None:
            c = Candidate(name=canon, card=card, roles=tag_roles(card))
            cands[canon] = c
        c.theme_fit = fit[raw_name]
        touched.append(c)
        merged += 1
    # Re-score any candidate touched (theme_fit changes the value multiplier).
    for c in touched:
        c.score = consensus_score(c, cfg)
    return merged


def _package_labels(
    first_labels: list[str],
    commanders: list[Card],
    http: Http,
    cfg: Config,
) -> list[str]:
    """Exactly 4 package labels: requested theme name(s) first, then the
    commanders' own tag_counts values (count desc), then DEFAULT_THEMES."""
    labels: list[str] = []
    seen: set[str] = set()

    def add(label: str) -> None:
        if label and label.casefold() not in seen:
            labels.append(label)
            seen.add(label.casefold())

    for label in first_labels:
        add(label)
    for cmd in commanders:
        page = fetch_commander_page(http, cfg, slugify(cmd.name))
        if page is None:
            continue
        ordered = sorted(page.tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for slug, _count in ordered:
            add(page.tag_values.get(slug, slug))
    for label in DEFAULT_THEMES:
        add(label)
    return labels[:4]


def build(
    cfg: Config,
    http: Http,
    commander_names: list[str],
    *,
    budget_nzd: float | None = None,
    condition: str | None = None,
    themes: list[str] | None = None,
    companion_name: str | None = None,
    output_dir: str | Path = "output",
    verbose: bool = True,
    theme_slugs: list[str] | None = None,
    tag_query: str | None = None,
    color_identity: set[str] | None = None,
    n_candidates: int | None = None,
    on_budget_fail: str = "degrade",
    owned_names: list[str] | None = None,
    free_basics: bool = False,
) -> BuildResult:
    t0 = time.monotonic()
    condition = (condition or cfg.pricing.condition or "any").strip()
    status: dict[str, str] = {}
    n_candidates = n_candidates or cfg.theme.candidates_default

    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"== loading Scryfall bulk data (canonicalization) ==")
    try:
        db = ScryfallDB.load(http, cfg)
    except SourceBlocked as exc:
        raise SystemExit(f"error: could not load the Scryfall card database: {exc}") from exc
    log(f"   {len(db.all)} unique cards, {len(db.commanders)} commanders")

    # -------------------------------------------------------------- free cards
    # --owned: cards the user already owns are NZ$0.00: they cost nothing and
    # are NEVER probed. Names are casefold-matched against the canonical DB;
    # unmatched names are warned and ignored. --free-basics adds the canonical
    # basic lands found in the feed (derived, never hardcoded staple names).
    free_names: frozenset[str] = frozenset()
    if owned_names:
        matched: set[str] = set()
        for raw in owned_names:
            name = raw.strip()
            if not name:
                continue
            card = db.by_name.get(name.casefold())
            if card is None:
                log(f"   owned: {name!r} not in the card database (ignored)")
            else:
                matched.add(card.name.casefold())
        free_names |= frozenset(matched)
        log(
            f"   owned: {len(matched)}/{len(owned_names)} name(s) matched -> "
            "free (NZ$0.00, no probe)"
        )
    if free_basics:
        basics = sorted({c.name.casefold() for c in db.all if c.is_basic_land})
        if not basics:
            log("   free-basics: no basic lands in the card database (ignored)")
        else:
            free_names |= frozenset(basics)
            log(
                f"   free-basics: {len(basics)} basic land(s) free (NZ$0.00, no "
                f"probe): {', '.join(basics)}"
            )

    # ------------------------------------------------------ theme resolution
    mode: str | None = None  # "tag" | "tags"
    pages: list = []
    search_cards: list[Card] = []
    pool: dict[str, tuple[float, float]] = {}  # name -> (synergy, lift)
    theme_names: list[str] = []
    tag_slugs: list[str] = []

    if theme_slugs:
        mode = "tag"
        log(f"== resolving theme(s): {', '.join(theme_slugs)} ==")
        for slug in theme_slugs:
            page = resolve_theme_page(http, cfg, slug)
            pages.append(page)
            theme_names.append(page.name or page.slug)
            for tc in page.cards:
                syn, lift = pool.get(tc.name, (0.0, 0.0))
                pool[tc.name] = (max(syn, tc.synergy), max(lift, tc.lift))
            tag_slugs.append(page.slug)
            log(
                f"   {page.slug} ({page.kind}): {len(page.commanders)} commander "
                f"candidates, {len(page.cards)} pool cards"
            )
    elif tag_query:
        mode = "tags"
        log(f"== Scryfall tag search: {tag_query!r} ==")
        search_cards = tag_search(http, tag_query)
        theme_names.append(tag_query)
        for c in search_cards:
            pool.setdefault(c.name, (0.0, 0.0))
        log(f"   {len(search_cards)} cards in the tag pool")

    # ------------------------------------------------------- commander source
    selection: CommanderSelection | None = None
    pricer: Pricer | None = None
    if not commander_names:
        if mode is None:
            raise SystemExit(
                "error: no commander given and no --theme/--tags to pick one from; "
                'pass a commander name, e.g. python -m deckbuilder "River Song"'
            )
        log("== theme mode: selecting the commander ==")
        if mode == "tag":
            candidates: list = []
            seen: set[str] = set()
            for page in pages:
                for cand in candidates_from_theme_page(page, db):
                    if cand.slug not in seen:
                        seen.add(cand.slug)
                        candidates.append(cand)
        else:
            candidates = candidates_from_search(search_cards, limit=30)
        log(f"   {len(candidates)} Commander-legal candidates")

        pricer = Pricer(http, cfg, free_names=free_names)  # shared with the later pricing pass
        scored = score_commanders(
            db,
            http,
            cfg,
            mode=mode,
            candidates=candidates,
            pool=set(pool),
            theme_slugs=tuple(tag_slugs),
            required_colors=color_identity,
            n_candidates=n_candidates,
            pricer=pricer,
            condition=condition,
            log=log,
        )
        winner = scored[0]  # score_commanders returns the gate winner first
        winner_names = winner_commanders(scored)
        gate_txt = "gate passed" if winner.gate_passed else "gate not passed"
        log(f"   selected: {winner.name} (score {winner.prelim_score:.4f}, {gate_txt})")
        for i, s in enumerate(scored[1:], 2):
            log(f"   runner-up {i}: {s.name} (score {s.prelim_score:.4f})")
        selection = CommanderSelection(
            slug=", ".join(tag_slugs) if mode == "tag" else tag_query or "",
            theme_names=theme_names,
            required_colors=sorted(color_identity or ()),
            mode=mode,
            winner_name=winner.name,
            winner=winner,
            runners=scored[1:],
            budget_nzd=budget_nzd,
        )
        commanders = _resolve_commanders(db, cfg, winner_names)
    else:
        log("== resolving commanders ==")
        commanders = _resolve_commanders(db, cfg, commander_names)
    for c in commanders:
        log(f"   {c.name} ({''.join(c.color_identity) or 'colorless'})")

    # Companion is resolved here (it only touches the card DB), because the
    # budget floor below must account for it -- and the floor runs before any
    # source fetch or pricing work.
    companion = _detect_companion(db, commanders, companion_name)
    if companion is not None:
        log(f"   companion: {companion.name} (recorded outside the 100-card limit)")

    # ------------------------------------------------ budget floor precheck
    # The floor is the cheapest legal 100-card deck this identity can buy:
    # commander + companion + shipping + 99 cheapest legal cards. If the
    # budget is below it, no solver run can succeed -- fail fast with the
    # minimum buildable budget instead of burning minutes on impossible work.
    # If the budget covers it, the build MUST succeed (the degradation ladder
    # degrades quality, never refuses).
    floor: FloorBreakdown | None = None
    if budget_nzd is not None:
        pricer = pricer or Pricer(http, cfg, free_names=free_names)
        log("== budget floor precheck (cheapest legal deck, probe-free) ==")
        try:
            floor = compute_floor(db, cfg, commanders, companion, pricer, condition,
                                  free_names=free_names)
        except FloorInfeasible as exc:
            raise SystemExit(str(exc)) from exc
        if floor.total_nzd > budget_nzd:
            raise SystemExit(
                f"budget NZ${budget_nzd:,.2f} is below the minimum buildable cost: "
                f"{floor.minimum_budget_message()}"
            )
        log(
            f"   floor: {floor.minimum_budget_message()} "
            f"<= budget NZ${budget_nzd:,.2f}; build must succeed"
        )

    log("== fetching recommendation sources (independent, degraded-safe) ==")
    edhrec: EdhrecData | None = _fetch_source(
        status, "edhrec", lambda: fetch_edhrec(http, cfg, [c.name for c in commanders])
    )
    rc: RecommanderData | None = _fetch_source(
        status,
        "recommander",
        lambda: fetch_recommander(
            http, cfg, commanders[0].name, commanders[1].name if len(commanders) > 1 else None
        ),
    )
    et: EdhToolsData | None = _fetch_source(
        status, "edhtools", lambda: fetch_edhtools(http, cfg, [c.oracle_id for c in commanders])
    )
    for name, st in status.items():
        log(f"   {name}: {st}")

    identity: list[str] = []
    for c in commanders:
        for col in c.color_identity:
            if col not in identity:
                identity.append(col)

    log("== scoring consensus candidates ==")
    cands = merge_candidates(
        db, cfg, identity, commanders, edhrec, rc, et,
    )
    log(f"   {len(cands)} canonical candidates with >= 1 source signal")

    if pool:
        merged = _merge_theme_pool(
            db, cfg, identity, cands, pool,
            {c.name.casefold() for c in commanders},
        )
        log(f"   theme pool: {len(pool)} cards, {merged} carry a theme_fit modifier into scoring")

    # Package labels: theme mode builds them (requested name(s) + the winner's
    # own top tags); otherwise the explicit --themes list or the defaults.
    labels = themes
    if theme_slugs or tag_query:
        labels = _package_labels(theme_names, commanders, http, cfg)
        log(f"   theme packages: {', '.join(labels)}")
    elif labels and len(labels) != 4:
        labels = None

    pricer = pricer or Pricer(http, cfg, free_names=free_names)

    log("== assembling deck (estimate-first budget, joint solver) ==")
    fx = pricer.fx_rate()
    if budget_nzd is not None and fx is None:
        log("   WARNING: FX rate unavailable; estimated prices are 0 and the budget penalty is off")
    asm = assemble(
        db,
        cfg,
        commanders,
        cands,
        themes=labels,
        companion=companion,
        budget_nzd=budget_nzd,
        fx=fx,
        free_names=free_names,
    )
    deck = asm.deck
    log(
        f"   lands={deck.meta['quotas']['lands'][1]} ramp={deck.meta['quotas']['ramp'][1]} "
        f"draw={deck.meta['quotas']['draw'][1]} removal={deck.meta['quotas']['removal'][1]} "
        f"theme={deck.meta['quotas']['theme'][1]} total={len(deck.all_cards)}"
    )
    if len(deck.all_cards) != 100:
        log(f"   WARNING: deck is {len(deck.all_cards)} cards, not 100")
    log(
        f"   solver={asm.solver_used} lambda={asm.lambda_final:.6g} "
        f"estimated total NZ${sum(asm.est_prices.values()):,.2f}"
    )

    log(f"== verifying prices via MTG Singles NZ (condition={condition!r}) ==")
    rr = resolve(
        asm,
        db,
        cfg,
        cands,
        commanders,
        pricer,
        budget_nzd=budget_nzd,
        condition=condition,
        themes=labels,
        companion=companion,
        fx=fx,
        free_names=free_names,
    )
    deck = rr.asm.deck
    prices = rr.prices
    stock_subs = rr.subs
    budget = rr.budget
    verified_count = sum(
        1 for p in prices.values() if p and not p.estimated and p.price_nzd is not None
    )
    estimated_count = sum(1 for p in prices.values() if p and p.estimated)
    log(f"   verified {verified_count} in-stock, {estimated_count} estimated ({len(prices)} total)")
    log(
        f"   budget: total NZ${budget.total_nzd:,.2f} -> "
        f"{'WITHIN' if budget.within_budget else 'OVER'} (gap NZ${budget.gap_nzd:,.2f}), "
        f"re-solves={rr.resolves}"
    )

    _enforce_budget(budget, budget_nzd, rr.resolves, on_budget_fail)

    if budget is not None and not budget.within_budget:
        log(
            f"!! OVER-BUDGET DECK ACCEPTED (--on-budget-fail emit): "
            f"total NZ${budget.total_nzd:,.2f} exceeds the budget of "
            f"NZ${budget_nzd:,.2f} by NZ${budget.gap_nzd:,.2f}; "
            f"the deck is written with within_budget=false"
        )

    log(f"== writing outputs to {output_dir} ==")
    outputs = write_outputs(
        output_dir, deck, cands, prices, stock_subs, budget, status, cfg, selection,
        floor=floor,
    )
    for p in outputs:
        log(f"   {p}")

    return BuildResult(
        deck=deck,
        cands=cands,
        prices=prices,
        stock_subs=stock_subs,
        budget=budget,
        source_status=status,
        outputs=outputs,
        commanders=commanders,
        elapsed_sec=round(time.monotonic() - t0, 1),
        selection=selection,
        lambda_final=rr.asm.lambda_final,
        solver_used=rr.asm.solver_used,
        estimated_count=estimated_count,
        verified_count=verified_count,
        resolves=rr.resolves,
    )
