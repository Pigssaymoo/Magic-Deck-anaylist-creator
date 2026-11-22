"""Deterministic commander auto-selection for theme mode (no LLM).

Two candidate sources:
  tag mode  - the Top/New Commanders lists of an EDHREC tag/theme page;
  tags mode - commander-legal creatures from a Scryfall tag search (otag:/atag:).

Every candidate gets a preliminary score from cached page fetches only:
  popularity  - log-scaled ``log(1+n)/log(1+n_max)`` of the candidate's
                num_decks (tags mode) or inclusion (tag mode);
  synergy_fit - synergy-weighted overlap of the candidate's own top cards
                with the theme pool (tag mode also blends the theme-tag fit);
  color_fit   - coverage of any required --color-identity (1.0 if none).
prelim = 0.375*popularity + 0.4375*synergy_fit + 0.1875*color_fit.

Buyability is a HARD feasibility gate, not a score component: candidates are
walked in preliminary order and each is probed (satisficing) against MTG
Singles NZ core-by-core until it either passes (core budget exhausted with
unavailability below the threshold) or fails (unavailability reaches the
threshold). The first candidate to pass wins and the search stops
immediately, which bounds live price requests. Scoring and probing are fully
deterministic: stable sorts, tie-breaks on ``name.casefold()``, no RNG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .cards import Card
from .config import Config
from .edhrec import PAGE_URL, ThemePage, slugify
from .httpclient import Http, SourceBlocked
from .scryfall import ScryfallDB, find_card


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# ----------------------------------------------------------------- page data
@dataclass
class CommanderPage:
    """The pieces of an EDHREC commander page the scorer needs."""

    name: str = ""
    num_decks: int = 0
    topcards: list[tuple[str, float]] = field(default_factory=list)  # (name, synergy), list order
    tag_counts: dict[str, int] = field(default_factory=dict)  # slug -> count
    tag_values: dict[str, str] = field(default_factory=dict)  # slug -> human tag name


def _parse_commander_page(data: dict[str, Any]) -> CommanderPage:
    """Parse the scorer-relevant fields of a raw EDHREC commander page (pure)."""
    inner = data.get("data")
    root = inner if isinstance(inner, dict) else data
    jd = (root.get("container") or {}).get("json_dict", {}) or {}
    card = jd.get("card") or {}
    topcards: list[tuple[str, float]] = []
    for cl in jd.get("cardlists") or []:
        tag = (cl.get("tag") or "").lower()
        if tag not in ("topcards", "highliftcards"):
            continue
        for cv in cl.get("cardviews") or []:
            name = str(cv.get("name") or "").strip()
            if not name:
                continue
            try:
                syn = float(cv.get("synergy") or 0)
            except (TypeError, ValueError):
                syn = 0.0
            topcards.append((name, syn))
    tag_counts: dict[str, int] = {}
    tag_values: dict[str, str] = {}
    for tc in root.get("tag_counts") or []:
        slug = str((tc or {}).get("slug") or "").strip()
        if not slug:
            continue
        tag_counts[slug] = max(tag_counts.get(slug, 0), _int((tc or {}).get("count")))
        value = str((tc or {}).get("value") or "").strip()
        if value and slug not in tag_values:
            tag_values[slug] = value
    return CommanderPage(
        name=str(card.get("name") or "").strip(),
        num_decks=_int(card.get("num_decks")),
        topcards=topcards,
        tag_counts=tag_counts,
        tag_values=tag_values,
    )


def fetch_commander_page(http: Http, cfg: Config, slug: str) -> CommanderPage | None:
    """Fetch one commander page (7-day cache); ``None`` on block/404."""
    try:
        data = http.get_json(PAGE_URL.format(slug=slug), ttl=cfg.http.rec_cache_days * 24 * 3600)
    except SourceBlocked:
        return None
    return _parse_commander_page(data)


# ----------------------------------------------------------------- candidates
@dataclass
class CandidateInfo:
    """One candidate entering the scorer (already legality-filtered)."""

    name: str  # canonical display name (partner: "A, Title // B, Title")
    slug: str  # EDHREC commander slug
    is_partner: bool = False
    partner_names: tuple[str, ...] = ()
    num_decks: int = 0  # tag-page value (tag mode) when available
    potential_decks: int = 0  # tag-page value (tag mode) when available

    @property
    def inclusion(self) -> float:
        if self.potential_decks <= 0:
            return 0.0
        return self.num_decks / self.potential_decks


def candidates_from_theme_page(page: ThemePage, db: ScryfallDB) -> list[CandidateInfo]:
    """Legality-filter candidates from an EDHREC tag/theme page.

    A candidate survives only if every partner member resolves in the
    Scryfall bulk DB and is Commander (OCG) legal. The canonical name is the
    Scryfall ``A // B`` join of the resolved members.
    """
    out: list[CandidateInfo] = []
    for tc in page.commanders:
        members = tc.partner_names if (tc.is_partner and tc.partner_names) else (tc.name,)
        resolved: list[Card] = []
        for m in members:
            card = find_card(db, m)
            if card is None or not card.commander_legal:
                resolved = []
                break
            resolved.append(card)
        if not resolved:
            continue  # not (fully) Commander legal -> drop
        out.append(
            CandidateInfo(
                name=" // ".join(c.name for c in resolved),
                slug=tc.slug or slugify(tc.name),
                is_partner=len(resolved) > 1,
                partner_names=tuple(c.name for c in resolved),
                num_decks=tc.num_decks,
                potential_decks=tc.potential_decks,
            )
        )
    return out


def candidates_from_search(cards: list[Card], limit: int = 30) -> list[CandidateInfo]:
    """Commander candidates from a Scryfall tag search (CMC-ascending).

    First ``limit`` results that are creatures and Commander-legal. Partners
    arrive as a single ``A // B`` oracle entry.
    """
    out: list[CandidateInfo] = []
    for c in cards:
        if not (c.is_creature and c.commander_legal):
            continue
        parts = tuple(p.strip() for p in c.name.split("//") if p.strip())
        out.append(
            CandidateInfo(
                name=c.name,
                slug=slugify(c.name),
                is_partner=len(parts) > 1,
                partner_names=parts,
            )
        )
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------- scoring
@dataclass
class CommanderScore:
    name: str
    slug: str
    is_partner: bool = False
    partner_names: tuple[str, ...] = ()
    popularity: float = 0.0
    synergy_fit: float = 0.0
    color_fit: float = 0.0
    prelim_score: float = 0.0
    # Buyability gate (hard feasibility, not a score component):
    probes: int = 0  # core theme cards actually probed for this candidate
    unavailable: int = 0  # probes that were missing / out of stock / unpriced
    gate_passed: bool | None = None  # None = not evaluated; True/False = outcome
    gate_probe_order: list[str] = field(default_factory=list)  # probe order recorded


def _log_scale(values: dict[str, float]) -> dict[str, float]:
    """``log(1+n)/log(1+n_max)`` per candidate; ``n_max`` over candidates.

    Deterministic; returns 0.0 for every candidate when ``n_max <= 0``.
    """
    vmax = max(values.values()) if values else 0.0
    if vmax <= 0:
        return {name: 0.0 for name in values}
    denom = math.log(1.0 + vmax)
    return {name: math.log(1.0 + max(0.0, v)) / denom for name, v in values.items()}


def _color_fit(
    db: ScryfallDB,
    cand: CandidateInfo,
    required: set[str] | None,
    penalty: float,
) -> float:
    if not required:
        return 1.0
    identity: set[str] = set()
    for n in (cand.partner_names or (cand.name,)):
        card = find_card(db, n)
        if card is not None:
            identity.update(card.color_identity)
    missing = len(required - identity)
    return max(0.0, 1.0 - penalty * missing)


def _pool_fit(page: CommanderPage, pool: set[str]) -> float:
    """Synergy-weighted overlap of the candidate's top-20 cards with the pool.

    ``sum(syn_i for i in top20 if i in pool) / sum(syn_i for i in top20)``;
    0.0 when the top-20 synergy mass is non-positive.
    """
    top20: list[tuple[str, float]] = []
    seen: set[str] = set()
    for nm, syn in page.topcards:
        if nm in seen:
            continue
        seen.add(nm)
        top20.append((nm, syn))
        if len(top20) >= 20:
            break
    total = sum(syn for _nm, syn in top20)
    if total <= 0:
        return 0.0
    overlap = sum(syn for nm, syn in top20 if nm in pool)
    return overlap / total


def _core_cards(page: CommanderPage, pool: set[str], cap: int) -> list[tuple[str, float]]:
    """The candidate's probeable core: topcards ∩ pool, deduped (keep the
    highest synergy), sorted synergy-desc then name-asc, capped at ``cap``."""
    best: dict[str, float] = {}
    for nm, syn in page.topcards:
        if nm not in pool:
            continue
        if nm not in best or syn > best[nm]:
            best[nm] = syn
    core = sorted(best.items(), key=lambda kv: (-kv[1], kv[0].casefold()))
    return core[: max(0, cap)]


def _gate(
    page: CommanderPage | None,
    pool: set[str],
    pricer: Any | None,
    condition: str,
    cfg: Config,
) -> tuple[bool | None, list[str], int, int]:
    """Satisficing buyability gate for one candidate.

    Probes the candidate's core theme cards (synergy desc, then name asc, at
    most ``core_probe_max``) sequentially against MTG Singles NZ. A probe is
    *unavailable* when it is missing, out of stock, or unpriced. The candidate
    FAILS as soon as unavailability reaches ``unavailable_core_threshold``
    (probing stops immediately); it PASSES once the probe budget is exhausted
    without failing.

    Returns ``(passed, probe_order, probes, unavailable)`` where ``passed`` is
    ``None`` when the gate could not be evaluated (no page, no pricer, or no
    core cards in the pool) — neither a pass nor a hard fail.
    """
    if page is None or pricer is None:
        return None, [], 0, 0
    core = _core_cards(page, pool, cfg.theme.core_probe_max)
    if not core:
        return None, [], 0, 0
    probe_order: list[str] = []
    unavailable = 0
    threshold = max(1, cfg.theme.unavailable_core_threshold)
    for nm, _syn in core:
        probe_order.append(nm)
        pi = pricer.price_card(nm, condition)
        if pi is None or pi.stock <= 0 or pi.price_nzd is None:
            unavailable += 1
            if unavailable >= threshold:
                return False, probe_order, len(probe_order), unavailable
    return True, probe_order, len(probe_order), unavailable


def score_commanders(
    db: ScryfallDB,
    http: Http,
    cfg: Config,
    *,
    mode: str,  # "tag" | "tags"
    candidates: list[CandidateInfo],
    pool: set[str],
    theme_slugs: tuple[str, ...] = (),
    required_colors: set[str] | None = None,
    n_candidates: int = 5,  # retained for CLI compatibility; the satisficing gate is not bounded by it
    pricer: Any | None = None,
    condition: str = "any",
    log: Callable[..., None] = print,
) -> list[CommanderScore]:
    """Score all candidates, run the buyability gate, and return the winner first.

    Deterministic: page fetches are cached, all sorts are stable with a
    ``name.casefold()`` tie-break, and the gate is satisficing (stop at the
    first passing candidate) so live price requests are bounded. The result
    list is ``[winner, *others in preliminary order]``; ``winner`` is the
    first candidate in preliminary order whose gate passes, or the
    top-preliminary candidate when no gate passes (degraded).
    """
    if not candidates:
        raise SystemExit(
            "error: no Commander-legal candidates found for this theme; "
            "try a different theme or explicit commander name"
        )
    t = cfg.theme
    del n_candidates  # retained for CLI compatibility; gate walk is satisficing

    # 1) fetch every candidate page (degraded-safe; 7-day cached)
    pages: dict[str, CommanderPage | None] = {}
    for cand in candidates:
        log(f"  commander candidate: {cand.name}")
        pages[cand.name] = fetch_commander_page(http, cfg, cand.slug)

    # 2) per-candidate preliminary score
    if mode == "tag":
        # popularity from the theme page's inclusion (num_decks / potential_decks)
        pop_values = {c.name: c.inclusion for c in candidates}
    else:
        # tags mode: each candidate's own page num_decks
        pop_values = {
            c.name: (pages[c.name].num_decks if pages[c.name] is not None else 0)
            for c in candidates
        }
    pop = _log_scale(pop_values)

    scores: list[CommanderScore] = []
    for cand in candidates:
        page = pages[cand.name]
        pool_fit = _pool_fit(page, pool) if page is not None else 0.0
        tag_fit = 0.0
        if mode == "tag" and page is not None and theme_slugs:
            total = sum(page.tag_counts.values())
            if total > 0:
                tag_fit = max(page.tag_counts.get(s, 0) for s in theme_slugs) / total
        synergy = (0.5 * pool_fit + 0.5 * tag_fit) if mode == "tag" else pool_fit
        color = _color_fit(db, cand, required_colors, t.color_penalty_per_missing)
        prelim = (
            t.weight_popularity * pop[cand.name]
            + t.weight_synergy_fit * synergy
            + t.weight_color * color
        )
        scores.append(
            CommanderScore(
                name=cand.name,
                slug=cand.slug,
                is_partner=cand.is_partner,
                partner_names=cand.partner_names,
                popularity=pop[cand.name],
                synergy_fit=synergy,
                color_fit=color,
                prelim_score=prelim,
            )
        )

    # 3) preliminary order (tie-break name.casefold())
    order = sorted(scores, key=lambda s: (-s.prelim_score, s.name.casefold()))

    # 4) buyability gate: walk in preliminary order, first pass wins
    winner: CommanderScore | None = None
    for s in order:
        passed, probe_order, probes, unavailable = _gate(
            pages[s.name], pool, pricer, condition, cfg
        )
        s.gate_passed = passed
        s.gate_probe_order = probe_order
        s.probes = probes
        s.unavailable = unavailable
        outcome = "pass" if passed else ("fail" if passed is False else "n/a")
        log(
            f"  gate {s.name}: {outcome} "
            f"(probes={probes}, unavailable={unavailable})"
        )
        if passed:
            winner = s
            break
    if winner is None:
        # degraded: nobody passed the gate -> best preliminary candidate wins
        winner = order[0]

    # 5) winner first, then the rest in preliminary order
    result = [winner] + [s for s in order if s is not winner]
    for i, s in enumerate(result, 1):
        log(
            f"  candidate {i}/{len(result)}: {s.name} "
            f"score={s.prelim_score:.4f} "
            f"(pop={s.popularity:.3f} syn={s.synergy_fit:.3f} "
            f"color={s.color_fit:.2f})"
        )
    return result


def winner_commanders(scored: list[CommanderScore]) -> list[str]:
    """The winner's individual commander names (partner pairs split apart)."""
    w = scored[0]
    return list(w.partner_names or (w.name,))
