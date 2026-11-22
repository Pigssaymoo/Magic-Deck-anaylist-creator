"""Output writers: decklist.txt, decklist.json, substitutions.md, report.md."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assembly import Deck
from .budget import BudgetResult
from .cards import COLOR_ORDER
from .config import Config
from .floor import FloorBreakdown
from .pricing import PriceInfo
from .scoring import Candidate
from .budget import Substitution

ROLE_ORDER = ("commander", "land", "ramp", "draw", "removal", "theme")


def _sig_dict(c: Candidate | None) -> dict[str, Any]:
    if c is None:
        return {}
    out: dict[str, Any] = {}
    for name, s in c.signals.items():
        out[name] = {
            "norm": round(s.norm, 4),
            "rank": s.rank,
            "n": s.n,
            **({k: round(v, 4) if isinstance(v, float) else v for k, v in s.detail.items()}),
        }
    return out


def _price_dict(pi: PriceInfo | None) -> dict[str, Any]:
    if pi is None:
        return {}
    return {
        "price_nzd": pi.price_nzd,
        "condition": pi.condition,
        "stock": pi.stock,
        "url": pi.url,
        "printing": pi.printing,
        "estimated": pi.estimated,
        "source": pi.source,
        "note": pi.note,
    }


def _price_source(pi: PriceInfo | None) -> str | None:
    """Semantic price origin for a deck card: ``verified`` (in-stock MTGS
    listing), ``estimate`` (Scryfall MV x FX), ``owned`` (NZ$0.00 via
    --owned/--free-basics), or None when the card has no price at all."""
    if pi is None or pi.price_nzd is None:
        return None
    if pi.estimated:
        return "estimate"
    if pi.source == "owned":
        return "owned"
    return "verified"


def _reserved_cost_nzd(floor: FloorBreakdown | None) -> float | None:
    """The reserved portion of the basket: commander(s) + companion + flat
    shipping -- the cost locked in before any of the 99 solved slots."""
    if floor is None:
        return None
    return round(floor.commander_nzd + floor.companion_nzd + floor.shipping_nzd, 2)


def curve(deck: Deck) -> dict[int, int]:
    """Non-land CMC histogram."""
    c: Counter[int] = Counter()
    for a in deck.assignments:
        if a.card and not a.card.is_land:
            c[a.card.cmc] += 1
    return dict(sorted(c.items()))


def pip_counts(deck: Deck) -> dict[str, int]:
    """Card count per color (multicolor counted once per color) + colorless."""
    c: Counter[str] = Counter()
    for a in deck.all_cards:
        if a.card is None:
            continue
        if not a.card.colors:
            c["colorless"] += 1
        else:
            for col in a.card.colors:
                c[col] += 1
    ordered = {k: c[k] for k in COLOR_ORDER if k in c}
    if "colorless" in c:
        ordered["colorless"] = c["colorless"]
    return ordered


def package_counts(deck: Deck) -> dict[str, int]:
    c: Counter[str] = Counter()
    for a in deck.all_cards:
        if a.package:
            c[a.package] += 1
    return dict(c)


def avg_mv(deck: Deck) -> float | None:
    mvs = [a.card.mv_usd for a in deck.all_cards if a.card and a.card.mv_usd is not None]
    if not mvs:
        return None
    return round(sum(mvs) / len(mvs), 2)


def _fmt_nzd(v: float | None) -> str:
    return f"NZ${v:,.2f}" if v is not None else "n/a"


# ----------------------------------------------------------------- decklist
def render_decklist_txt(deck: Deck) -> str:
    """Exact MTG Singles paste format: one ``1 <canonical name>`` line per card.

    No headers, comments, blank lines, set codes, prices, or markers. DFCs
    keep their `` // `` separator. Order:
      1. commander(s), in deck order (partner pairs stay together)
      2. all lands, A-Z
      3. ramp, A-Z
      4. draw, A-Z
      5. removal, A-Z
      6. theme packages in total-score order (cards A-Z within each)
    Exactly ``len(deck.all_cards)`` lines, LF-terminated, trailing newline.
    """
    by_role: dict[str, list] = {r: [] for r in ROLE_ORDER}
    for a in deck.all_cards:
        by_role[a.role].append(a)

    lines: list[str] = []
    # 1. commanders (canonical order; partner pairs stay in deck order)
    for a in deck.commanders:
        lines.append(f"1 {a.name}")
    # 2-5. fixed role blocks, each alphabetical
    for role in ("land", "ramp", "draw", "removal"):
        for a in sorted(by_role[role], key=lambda x: x.name.casefold()):
            lines.append(f"1 {a.name}")
    # 6. theme packages: packages in total-score order, cards A-Z within
    packages: dict[str, list] = {}
    for a in by_role["theme"]:
        packages.setdefault(a.package or "Theme", []).append(a)

    def pkg_key(item: tuple[str, list]) -> tuple[float, str]:
        _pkg, items = item
        return (-(sum(a.score for a in items)), _pkg.casefold())

    for _pkg, items in sorted(packages.items(), key=pkg_key):
        for a in sorted(items, key=lambda x: x.name.casefold()):
            lines.append(f"1 {a.name}")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- json
def render_decklist_json(
    deck: Deck,
    cands: dict[str, Candidate],
    prices: dict[str, PriceInfo],
    budget: BudgetResult | None,
    source_status: dict[str, str],
    floor: FloorBreakdown | None = None,
) -> dict[str, Any]:
    cards = []
    for a in deck.all_cards:
        cand = cands.get(a.name)
        entry: dict[str, Any] = {
            "name": a.name,
            "qty": 1,
            "role": a.role,
            "package": a.package,
            # Scoring breakdown (Phase 1): final value + components.
            "score": round(a.score, 4),
            "base": round(cand.base, 4) if cand else None,
            "coverage": round(cand.coverage, 4) if cand else None,
            "agreement_n": cand.agreement_n if cand else None,
            "theme_fit": round(cand.theme_fit, 4) if (cand and cand.theme_fit is not None) else None,
            "cmc": a.card.cmc if a.card else None,
            "type_line": a.card.type_line if a.card else None,
            "colors": a.card.colors if a.card else None,
            "signals": _sig_dict(cand),
            "price": _price_dict(prices.get(a.name)),
            "price_source": _price_source(prices.get(a.name)),
            "est_price_nzd": (deck.meta.get("est_prices") or {}).get(a.name),
            "verified_price_nzd": (
                prices[a.name].price_nzd
                if a.name in prices
                and prices[a.name] is not None
                and not prices[a.name].estimated
                and prices[a.name].price_nzd is not None
                else None
            ),
            "mv_usd": a.card.mv_usd if a.card else None,
            "justification": a.note,
        }
        cards.append(entry)

    verified = sum(1 for p in prices.values() if p and not p.estimated and p.price_nzd is not None)
    estimated = sum(1 for p in prices.values() if p and p.estimated)
    missing = sum(1 for p in prices.values() if p and p.price_nzd is None)

    return {
        "commanders": [c.name for c in deck.commanders],
        "companion": deck.companion,
        "identity": deck.identity,
        "card_count": len(deck.all_cards),
        "cards": cards,
        "curve": curve(deck),
        "pips": pip_counts(deck),
        "packages": package_counts(deck),
        "avg_mv_usd": avg_mv(deck),
        "totals": {
            "verified_count": verified,
            "estimated_count": estimated,
            "unpriced_count": missing,
            "cards_total_nzd": budget.cards_total_nzd if budget else None,
            "shipping_nzd": budget.shipping_nzd if budget else None,
            "total_nzd": budget.total_nzd if budget else None,
            "budget_nzd": budget.budget_nzd if budget else None,
            "within_budget": budget.within_budget if budget else None,
            "gap_nzd": budget.gap_nzd if budget else None,
            "lambda_final": deck.meta.get("lambda_final"),
            "solver_used": deck.meta.get("solver_used"),
            "resolves": budget.resolves if budget else None,
        },
        # Budget-as-constraint audit trail (commit f): the cheapest legal deck
        # that could exist for this identity, the cost reserved outside the
        # 99 solved slots, and which ladder rungs had to degrade the deck.
        "budget_floor": (
            {
                "commander_nzd": floor.commander_nzd,
                "companion_nzd": floor.companion_nzd,
                "shipping_nzd": floor.shipping_nzd,
                "cards_nzd": floor.cards_nzd,
                "total_nzd": floor.total_nzd,
                "reserved_cost": _reserved_cost_nzd(floor),
            }
            if floor is not None
            else None
        ),
        "reserved_cost": _reserved_cost_nzd(floor),
        "degradation_rungs_applied": (
            list(budget.degradation_rungs_applied) if budget else None
        ),
        "substitutions": [],  # filled by write_outputs() with stock + budget swaps
        "source_status": source_status,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------- markdowns
# Fixed rung order for the substitutions log: stock fixes first, then the
# degradation ladder in ladder order. No-op rungs never appear (they produce
# no Substitution rows).
_RUNG_ORDER = ("stock", "R1", "R2", "R3", "R4")

_RUNG_TITLES = {
    "stock": "Stock (in-stock fixes)",
    "R1": "R1 — cheapest in-stock printing / wider condition",
    "R2": "R2 — role-preserving swap (re-solve + verified swap pass)",
    "R3": "R3 — relaxed non-structural quotas (cheaper role cards / basics)",
    "R4": "R4 — floor fill (cheapest legal card per slot)",
}


def _rung_of(sub: Substitution) -> str:
    """The ladder rung a substitution belongs to, from its reason prefix.

    Rung reasons are stamped by the ladder itself: ``budget: R1 save ...``,
    ``budget: R3 save ...``, ``budget: R4 save ...``; a plain
    ``budget: save ...`` is an R2 swap (re-solve pairing or swap pass);
    everything else is a stock fix.
    """
    r = sub.reason
    if r.startswith("budget: R1"):
        return "R1"
    if r.startswith("budget: R3"):
        return "R3"
    if r.startswith("budget: R4"):
        return "R4"
    if r.startswith("budget:"):
        return "R2"
    return "stock"


def render_substitutions_md(subs: list[Substitution]) -> str:
    if not subs:
        return "# Substitutions\n\nNo substitutions were needed — every card matched in-stock stock.\n"
    lines = ["# Substitutions", ""]
    for rung in _RUNG_ORDER:
        group = [s for s in subs if _rung_of(s) == rung]
        if not group:
            continue
        lines.append(f"## {_RUNG_TITLES[rung]}")
        lines.append("")
        if rung == "stock":
            lines.append(f"**{len(group)} substitution(s)**")
        else:
            saving = round(
                sum(max(0.0, (s.price_original or 0.0) - (s.price_replacement or 0.0)) for s in group),
                2,
            )
            lines.append(
                f"**{len(group)} substitution(s)** · saved {_fmt_nzd(saving)}"
            )
        lines.append("")
        lines.append("| Original | Replacement | Role | Reason | Score Δ | Price (orig → new) |")
        lines.append("|---|---|---|---|---|---|")
        for s in group:
            po = _fmt_nzd(s.price_original)
            pn = _fmt_nzd(s.price_replacement)
            est = " *(estimated)*" if s.estimated else ""
            lines.append(
                f"| {s.original} | {s.replacement} | {s.role} | {s.reason} | "
                f"{s.score_delta:+.3f} | {po} → {pn}{est} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _rerun_command(sel: object, names: list[str]) -> str:
    """Exact command to rebuild a runner-up's deck with the same theme+budget."""
    parts = ["python -m deckbuilder"] + [f'"{n}"' for n in names]
    slug = str(getattr(sel, "slug", "") or "")
    if getattr(sel, "mode", "tag") == "tag" and slug:
        for s in slug.split(", "):
            parts.append(f"--theme {s}")
    elif slug:
        parts.append(f'--tags "{slug}"')
    budget = getattr(sel, "budget_nzd", None)
    if budget:
        parts.append(f"--budget {float(budget):g}")
    return " ".join(parts)


def _gate_label(gate_passed: object) -> str:
    """Human label for a candidate's buyability-gate outcome (None = not run)."""
    if gate_passed is True:
        return "pass"
    if gate_passed is False:
        return "fail"
    return "n/a"


def _selection_section(sel: object, deck: Deck, cfg: Config) -> list[str]:
    """The 'Commander selection' block (theme mode only). Duck-typed on
    ``pipeline.CommanderSelection`` to avoid a circular import."""
    winner = getattr(sel, "winner", None)
    if winner is None:
        return []
    t = cfg.theme
    lines: list[str] = []
    lines.append("## Commander selection (theme mode)")
    lines.append("")
    theme_ref = ", ".join(getattr(sel, "theme_names", []) or [str(getattr(sel, "slug", ""))])
    lines.append(f"Theme(s): **{theme_ref}**" + (f" · required identity: **{''.join(getattr(sel, 'required_colors', []) or [])}**" if getattr(sel, "required_colors", None) else ""))
    gate_txt = {
        True: "gate **passed**",
        False: "gate **not passed** (degraded fallback to top preliminary score)",
        None: "gate **not evaluated** (no price source)",
    }[getattr(winner, "gate_passed", None)]
    lines.append(f"**Winner: {winner.name}** — score **{winner.prelim_score:.4f}** ({gate_txt})")
    lines.append("")
    lines.append("| Component | Weight | Value | Contribution |")
    lines.append("|---|---|---|---|")
    rows = [
        ("popularity (log-scaled num_decks / inclusion)", t.weight_popularity, winner.popularity),
        ("synergy fit (top cards ∩ theme pool" + (", tag overlap" if getattr(sel, "mode", "") == "tag" else "") + ")",
         t.weight_synergy_fit, winner.synergy_fit),
        ("color identity fit", t.weight_color, winner.color_fit),
    ]
    for label, w, v in rows:
        lines.append(f"| {label} | {w:.4f} | {v:.4f} | +{w * v:.4f} |")
    lines.append("")

    # One-line deterministic rationale: the component with the largest weighted
    # contribution is the driver.
    drivers = {
        "popularity": "it ranked highest for deck inclusion among the tag's commander candidates",
        "synergy fit": "its top cards overlap the theme's card pool the most",
        "color identity fit": "its color identity best covers the required colors",
    }
    key, best = max(
        (
            ("popularity", winner.popularity * t.weight_popularity),
            ("synergy fit", winner.synergy_fit * t.weight_synergy_fit),
            ("color identity fit", winner.color_fit * t.weight_color),
        ),
        key=lambda kv: (kv[1], kv[0]),
    )
    detail = ""
    if winner.probes:
        detail = (
            f" Probed {winner.probes} core theme cards: {winner.probes - winner.unavailable} in stock, "
            f"{winner.unavailable} missing/out of stock."
        )
    order = getattr(winner, "gate_probe_order", None) or []
    if order:
        detail += f" Probe order: {' → '.join(order)}."
    lines.append(f"Rationale: {drivers[key].capitalize()}.{detail}")
    lines.append("")

    runners: list = getattr(sel, "runners", []) or []
    if runners:
        lines.append("### Runners-up")
        lines.append("")
        lines.append("| # | Commander | Score | pop | syn | color | gate | Re-run exactly |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, s in enumerate(runners, 2):
            names = list(getattr(s, "partner_names", None) or [s.name])
            lines.append(
                f"| {i} | {s.name} | {s.prelim_score:.4f} | {s.popularity:.3f} | {s.synergy_fit:.3f} "
                f"| {s.color_fit:.2f} | {_gate_label(getattr(s, 'gate_passed', None))} | `{_rerun_command(sel, names)}` |"
            )
        lines.append("")

    # Short heuristic deck-tech line derived from the assembled deck.
    labels = (deck.meta.get("themes") or [])[:4]
    theme_cards = [
        a.name for a in deck.assignments if a.role == "theme" and a.card and not a.card.is_land
    ]
    theme_cards.sort(key=lambda n: n.casefold())
    lead = ", ".join(theme_cards[:3]) if theme_cards else "(none)"
    tech = (
        f"Heuristic deck tech: a {''.join(deck.identity) or 'colorless'}-identity build around the "
        f"{', '.join(labels) or 'default'} packages; lead theme cards include {lead}. "
        "This is a deterministic heuristic, not a playtested strategy."
    )
    lines.append(f"### Deck tech\n\n{tech}")
    lines.append("")
    return lines


def _upgrade_paths(budget: BudgetResult | None) -> list[tuple[Substitution, float]]:
    """``(substitution, score-gain-per-NZ$)`` for every budget downgrade.

    The upgrade path answers "if the budget allowed it, what should the buyer
    pay back for?" For each downgrade (original -> replacement) the marginal
    cost is ``price_original - price_replacement`` and the score gain is
    ``-score_delta`` (``score_delta`` is recorded as replacement - original).
    Rows with no positive marginal cost or no score gain carry 0.0 for
    score-gain-per-dollar. Sorted by score-gain-per-dollar descending; ties
    broken by the downgraded card's name (casefold) so the order is
    deterministic.
    """
    if budget is None:
        return []
    rows: list[tuple[Substitution, float]] = []
    for s in budget.swaps:
        marginal = round((s.price_original or 0.0) - (s.price_replacement or 0.0), 2)
        gain = -s.score_delta
        per_dollar = round(gain / marginal, 6) if marginal > 0 and gain > 0 else 0.0
        rows.append((s, per_dollar))
    rows.sort(key=lambda r: (-r[1], r[0].replacement.casefold()))
    return rows


def render_report_md(
    deck: Deck,
    cands: dict[str, Candidate],
    prices: dict[str, PriceInfo],
    budget: BudgetResult | None,
    source_status: dict[str, str],
    cfg: Config,
    selection: object | None = None,
    floor: FloorBreakdown | None = None,
) -> str:
    q = deck.meta.get("quotas", {})
    verified = sum(1 for p in prices.values() if p and not p.estimated and p.price_nzd is not None)
    estimated = sum(1 for p in prices.values() if p and p.estimated)
    missing = sum(1 for p in prices.values() if p and p.price_nzd is None)
    non_land = [a for a in deck.assignments if a.card and not a.card.is_land]
    avg_cmc = round(sum(a.card.cmc for a in non_land) / len(non_land), 2) if non_land else 0.0

    lines: list[str] = []
    lines.append(f"# Deck report — {', '.join(c.name for c in deck.commanders)}")
    lines.append("")
    lines.append(f"- Identity: `{''.join(deck.identity) or 'colorless'}`")
    lines.append(f"- Card count: **{len(deck.all_cards)}** (singleton, Commander rules)")
    lines.append(f"- Non-land avg CMC: {avg_cmc}")
    mv = avg_mv(deck)
    if mv is not None:
        lines.append(f"- Average market value (Scryfall USD): ${mv}")
    if deck.companion:
        lines.append(f"- Companion (outside the 100): {deck.companion}")
    lines.append("")

    # Commander selection (theme mode only)
    if selection is not None:
        lines.extend(_selection_section(selection, deck, cfg))

    # Curve
    lines.append("## Mana curve (non-lands)")
    cv = curve(deck)
    if cv:
        mx = max(cv.values())
        for cmc, n in cv.items():
            bar = "█" * max(1, round(n / mx * 30)) if mx else ""
            lines.append(f"CMC {cmc:>2}: {bar} {n}")
    else:
        lines.append("(no non-land cards?)")
    lines.append("")

    # Pips
    lines.append("## Color distribution")
    pc = pip_counts(deck)
    for k, v in pc.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Quotas vs actual
    lines.append("## Category counts vs. commandertemplate.com guide")
    lines.append("")
    lines.append("| Category | Quota | Actual |")
    lines.append("|---|---|---|")
    for key in ("lands", "ramp", "draw", "removal", "theme"):
        if key in q:
            quota, actual = q[key]
            lines.append(f"| {key} | {quota} | {actual} |")
    lines.append("")
    lines.append("Theme packages: " + ", ".join(f"{k} ({v})" for k, v in package_counts(deck).items()))
    land_reason = deck.meta.get("land_count_reason")
    if land_reason:
        lines.append("")
        lines.append(f"Land count: {land_reason}")
    lines.append("")

    # Budget
    lines.append("## Pricing & budget (MTG Singles NZ, NZD)")
    if budget is not None:
        lines.append("")
        lines.append(f"- Verified in-stock cards: {verified}")
        lines.append(f"- Estimated (Scryfall MV × FX, not verified buyable): {estimated}")
        lines.append(f"- Unpriced: {missing}")
        lines.append(f"- Cards total: {_fmt_nzd(budget.cards_total_nzd)}")
        lines.append(f"- Flat shipping (in budget): {_fmt_nzd(budget.shipping_nzd)}")
        lines.append(f"- **Total: {_fmt_nzd(budget.total_nzd)}**")
        if floor is not None:
            lines.append(f"- Budget floor (cheapest legal deck, probe-free): {floor.minimum_budget_message()}")
            lines.append(f"- Reserved (commander + companion + shipping): {_fmt_nzd(_reserved_cost_nzd(floor))}")
        if budget.budget_nzd is not None:
            lines.append(f"- Budget: {_fmt_nzd(budget.budget_nzd)} → **{'WITHIN BUDGET' if budget.within_budget else 'OVER BUDGET'}**")
            if not budget.within_budget:
                lines.append(f"- Gap after all swaps: {_fmt_nzd(budget.gap_nzd)} (closest legal deck produced)")
            if budget.degradation_rungs_applied:
                lines.append(f"- Degradation ladder rungs applied: {', '.join(budget.degradation_rungs_applied)}")
            else:
                lines.append("- Degradation ladder rungs applied: none (within budget on the verified deck)")
        lines.append(f"- Budget swaps made: {len(budget.swaps)}")
        lam = deck.meta.get("lambda_final")
        if lam is not None:
            lines.append(
                f"- Budget re-solve: penalty λ = {lam:g} (solver: {deck.meta.get('solver_used') or 'n/a'}), "
                f"re-solves used {budget.resolves}/{cfg.assembly.max_resolves}"
            )
    else:
        lines.append("")
        lines.append(f"- Verified in-stock cards: {verified}; estimated: {estimated}; unpriced: {missing}")
    lines.append("")

    # Upgrade path: every downgrade made to fit the budget, ranked by how
    # much consensus score each NZ$ of the saving buys back.
    upgrade = _upgrade_paths(budget) if (budget is not None and budget.budget_nzd is not None) else []
    if upgrade:
        lines.append("## Upgrade path (how to spend the savings)")
        lines.append("")
        lines.append("Each downgrade made to fit the budget, with the cost to buy back the better "
                     "card — best score value per dollar first.")
        lines.append("")
        lines.append("| Downgraded card (in deck) | Upgrade to | Better price | Marginal NZ$ | Δscore | Score gain / NZ$ |")
        lines.append("|---|---|---|---|---|---|")
        for s, per_dollar in upgrade:
            marginal = round((s.price_original or 0.0) - (s.price_replacement or 0.0), 2)
            gain = -s.score_delta
            lines.append(
                f"| {s.replacement} | {s.original} | {_fmt_nzd(s.price_original)} | "
                f"{_fmt_nzd(marginal)} | {gain:+.3f} | {per_dollar:.4f} |"
            )
        lines.append("")

    # Non-obvious picks
    lines.append("## Non-obvious picks (one-line justification)")
    picks = [a for a in deck.assignments if a.note and "generic legal fill" in a.note]
    if picks:
        for a in picks[:25]:
            lines.append(f"- **{a.name}** ({a.role}): {a.note}")
        if len(picks) > 25:
            lines.append(f"- …and {len(picks) - 25} more")
    else:
        low = [a for a in deck.assignments if a.role not in ("land",) and a.score < 0.4 and a.note]
        for a in low[:20]:
            lines.append(f"- **{a.name}** ({a.role}): {a.note}")
        if not low:
            lines.append("- (all picks are sourced top-40% consensus picks or basics)")
    lines.append("")

    # Sources
    lines.append("## Source status")
    for name, status in source_status.items():
        lines.append(f"- **{name}**: {status}")
    lines.append("")

    lines.append("## Limitations & assumptions")
    lines.append(
        "- Legality gate is Scryfall `legalities.commander` (OCG). EDHREC's own banlist is not "
        "exposed via its JSON API, so format-specific bans beyond OCG are not enforced."
    )
    lines.append("- Companion (if present) is priced/listed separately and never inside the 100.")
    lines.append("- Recommander scores have no fixed scale; they are rank-normalized within the response.")
    lines.append("- `--marketplaces` is accepted for future use; MTG Singles NZ is the only live marketplace source.")
    lines.append("- Deterministic: same cached inputs → same deck, scores, and swaps (stable sorts, no RNG).")
    lines.append("")
    lines.append(f"Recommendations aggregated from EDHREC, Recommander (https://recommander.cards), and edh.tools.")
    return "\n".join(lines) + "\n"


def _selection_dict(sel: object) -> dict[str, Any]:
    """JSON-safe snapshot of the theme-mode commander selection (duck-typed)."""
    winner = getattr(sel, "winner", None)
    if winner is None:
        return {}
    out: dict[str, Any] = {
        "mode": getattr(sel, "mode", None),
        "slug": getattr(sel, "slug", None),
        "theme_names": list(getattr(sel, "theme_names", []) or []),
        "required_colors": list(getattr(sel, "required_colors", []) or []),
        "winner": winner.name,
        "winner_score": round(float(getattr(winner, "prelim_score", 0.0)), 6),
        "winner_gate_passed": getattr(winner, "gate_passed", None),
        "winner_probes": int(getattr(winner, "probes", 0)),
        "winner_unavailable": int(getattr(winner, "unavailable", 0)),
        "winner_gate_probe_order": list(getattr(winner, "gate_probe_order", []) or []),
        "runners_up": [
            {
                "name": s.name,
                "score": round(float(getattr(s, "prelim_score", 0.0)), 6),
                "gate_passed": getattr(s, "gate_passed", None),
            }
            for s in (getattr(sel, "runners", []) or [])
        ],
    }
    return out


# -------------------------------------------------------------------- main
def write_outputs(
    outdir: str | Path,
    deck: Deck,
    cands: dict[str, Candidate],
    prices: dict[str, PriceInfo],
    stock_subs: list[Substitution],
    budget: BudgetResult | None,
    source_status: dict[str, str],
    cfg: Config,
    selection: object | None = None,
    floor: FloorBreakdown | None = None,
) -> list[Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    # stock_subs is the UNIFIED substitution stream (stock fixes + budget
    # re-solve pairs); budget.swaps is a subset of it, so no second merge.
    all_subs = list(stock_subs)

    paths: list[Path] = []
    p1 = out / "decklist.txt"
    # Write bytes, not text: the contract is LF-terminated on every platform
    # (text mode would translate \n -> \r\n on Windows).
    p1.write_bytes(render_decklist_txt(deck).encode("utf-8"))
    paths.append(p1)

    p2 = out / "decklist.json"
    payload = render_decklist_json(deck, cands, prices, budget, source_status, floor=floor)
    payload["substitutions"] = [s.__dict__ for s in all_subs]
    if selection is not None:
        payload["commander_selection"] = _selection_dict(selection)
    p2.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    paths.append(p2)

    p3 = out / "substitutions.md"
    p3.write_text(render_substitutions_md(all_subs), encoding="utf-8")
    paths.append(p3)

    p4 = out / "report.md"
    p4.write_text(
        render_report_md(
            deck, cands, prices, budget, source_status, cfg,
            selection=selection, floor=floor,
        ),
        encoding="utf-8",
    )
    paths.append(p4)
    return paths
