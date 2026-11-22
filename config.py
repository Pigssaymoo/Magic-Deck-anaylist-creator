"""Configuration loading and typed access."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceCfg:
    enabled: bool = True
    weight: float = 0.3


@dataclass
class ScoringCfg:
    edhrec_signals: dict[str, float] = field(
        default_factory=lambda: {"inclusion": 0.4, "synergy": 0.3, "lift": 0.3}
    )
    # Shrinkage prior: a card's base = (sum_present w_i*s_i + w0*mu0) / (sum_all w_i + w0).
    # Missing/blocked sources contribute zero to the numerator but their weight
    # stays in the denominator, so a run with fewer working sources yields LOWER
    # (never inflated) scores. The prior keeps single-source cards on a sane
    # floor instead of collapsing to their raw norm.
    prior_mean: float = 0.15
    prior_weight: float = 0.10
    # "Top" for the agreement factor means within the top fraction of a
    # source's own list (e.g. 0.20 = top 20%).
    agreement_top_fraction: float = 0.20
    # Theme affinity is a MODIFIER, not a source: value * (1 + beta * theme_fit).
    theme_fit_beta: float = 0.15


@dataclass
class QuotaCfg:
    lands_default: int = 36
    lands_low: int = 34
    lands_high: int = 38
    ramp: int = 12
    draw: int = 10
    removal: int = 10
    removal_wipes_max: int = 2
    theme_packages: int = 4


@dataclass
class SolverCfg:
    """Joint slot/card assignment solver (see deckbuilder.assembly).

    ``enabled=False`` (or scipy missing) falls back to a deterministic greedy
    fill + swap loop. The budget is enforced by a binary search on the price
    penalty lambda: double an upper bound (<= max_doublings) until the deck fits,
    then bisect (<= bisections) until within ``tolerance_nzd``.
    """

    enabled: bool = True
    max_doublings: int = 32
    bisections: int = 20
    tolerance_nzd: float = 0.50


@dataclass
class AssemblyCfg:
    high_cmc_threshold: float = 4.5
    dual_land_budget: dict[int, int] = field(
        default_factory=lambda: {0: 6, 1: 10, 2: 18, 3: 22, 4: 24, 5: 26}
    )
    fuzzy_match_threshold: float = 0.85
    # Prior value for a DB "filler" card (ranked by no source): a flat floor so
    # fillers slot in behind real source cards but ahead of nothing.
    filler_value: float = 0.05
    # Max estimate->verify re-solves in the budget/stock resolution loop.
    max_resolves: int = 3
    # Guaranteed cheap tail per solver column (ramp/draw/removal/theme): the K
    # cheapest legal options by effective price always survive the value
    # truncation, so a tight budget always has near-free candidates to fill
    # slots with. 0 disables the guarantee (value truncation only).
    cheap_pool_size: int = 10
    solver: SolverCfg = field(default_factory=SolverCfg)


@dataclass
class MtgSCfg:
    min_interval_sec: float = 2.1
    page_size: int = 20
    max_pages: int = 2
    referer: str = "https://www.mtgsingles.co.nz/"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
    # Circuit breaker: after this many consecutive MTGS source failures
    # (blocked, non-JSON, down) stop probing it for the rest of the run.
    # A fully-down source must not burn rate-limited dead time on every probe.
    block_breaker: int = 3


@dataclass
class FxCfg:
    cache_hours: int = 24
    base: str = "USD"
    quote: str = "NZD"


@dataclass
class SubstitutionCfg:
    max_candidates: int = 5


@dataclass
class PricingCfg:
    shipping_nzd: float = 7.50
    condition: str = "any"
    mtgs: MtgSCfg = field(default_factory=MtgSCfg)
    fx: FxCfg = field(default_factory=FxCfg)
    substitution: SubstitutionCfg = field(default_factory=SubstitutionCfg)
    # Safety margin applied to Scryfall-MV-derived estimates when they are
    # treated as if they were buyable prices (budget floor, price ceilings):
    # an unverified number is never trusted at face value.
    estimate_margin: float = 1.10
    # Max share of the budget one single card may take (per-slot price
    # ceiling = budget x fraction; candidates above it are pruned from the
    # solver pool before solving). 0 disables the ceiling.
    max_single_card_fraction: float = 0.15
    # Global per-run probe cap: the maximum number of DISTINCT card names that
    # may trigger a live MTG Singles lookup in one run (repeat/cache hits do
    # not count). Bounds probe explosion (initial verify + R2 swap pass). 0
    # disables the cap.
    max_probes_per_run: int = 150


@dataclass
class BudgetCfg:
    protect_theme_top: int = 10
    # Quality floor for the degradation ladder's replacement cards (R2 swap
    # pass, R3 quota relaxation): a *ranked* replacement (a source gave it a
    # consensus score) must score at least this; unranked cards (flat filler
    # prior, no measured quality) are exempt. 0 disables the floor. Bounds only
    # the ladder swaps, never the solver's own picks.
    min_quality: float = 0.10
    # Master switch for the degradation ladder's quality-degrading rungs
    # R1-R3 (condition relaxation, role-preserving re-solves + swap pass, quota
    # relaxation). The ladder degrades card quality to close the budget gap;
    # when off, an over-budget result is only closable by the R4 floor fill.
    degradation_ladder: bool = True
    # Rung R4 (floor fill): when the ladder has exhausted every other option,
    # allow the cheapest verified in-role card per slot to close the gap.
    floor_fill: bool = True


@dataclass
class HttpCfg:
    cache_dir: str = ".cache"
    rec_cache_days: int = 7
    price_cache_hours: int = 24
    default_min_interval_sec: float = 1.0
    max_retries: int = 4
    timeout_sec: int = 30


@dataclass
class ThemeCfg:
    """Commander auto-selection (theme mode) — deterministic preliminary score
    plus a hard buyability gate.

    Weights blend three 0-1 components (renormalized to sum to 1.0 after
    buyability was promoted from a score component to a feasibility gate):
      popularity  - log-scaled ``log(1+n)/log(1+n_max)`` of the candidate's
                    num_decks (tags mode) or inclusion (tag mode)
      synergy_fit - synergy-weighted overlap of the candidate's own top cards
                    with the theme's card pool (tag mode also blends tag fit)
      color       - coverage of any required --color-identity (1.0 if none)
    Buyability is a hard gate, not a weighted component: candidates are
    walked in preliminary order and probed (satisficing) until one passes.
    """

    candidates_default: int = 5  # retained for CLI compatibility; gate is satisficing
    weight_popularity: float = 0.375
    weight_synergy_fit: float = 0.4375
    weight_color: float = 0.1875
    color_penalty_per_missing: float = 0.5
    unavailable_core_threshold: int = 3  # unavailable core cards before a candidate fails
    core_probe_max: int = 8  # max MTGS stock probes per candidate


@dataclass
class Config:
    sources: dict[str, SourceCfg] = field(default_factory=dict)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    quotas: QuotaCfg = field(default_factory=QuotaCfg)
    assembly: AssemblyCfg = field(default_factory=AssemblyCfg)
    pricing: PricingCfg = field(default_factory=PricingCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    http: HttpCfg = field(default_factory=HttpCfg)
    theme: ThemeCfg = field(default_factory=ThemeCfg)

    def rec_ttl_seconds(self) -> float:
        return float(self.http.rec_cache_days * 24 * 3600)

    def price_ttl_seconds(self) -> float:
        return float(self.http.price_cache_hours * 3600)

    def source_weight(self, name: str) -> float:
        return self.sources[name].weight if name in self.sources else 0.0


def _src(raw: Any, default: float = 0.3) -> SourceCfg:
    raw = raw or {}
    return SourceCfg(
        enabled=bool(raw.get("enabled", True)),
        weight=float(raw.get("weight", default)),
    )


def _dict_int(d: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in (d or {}).items()}


def _dict(d: Any) -> dict:
    return d if isinstance(d, dict) else {}


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml (or an explicit path); fall back to defaults per section."""
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        for cand in (Path.cwd() / "config.yaml", Path(__file__).parent.parent / "config.yaml"):
            if cand.exists():
                raw = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
                break

    sc = raw.get("scoring", {}) or {}
    qu = raw.get("quotas", {}) or {}
    asy = raw.get("assembly", {}) or {}
    pr = raw.get("pricing", {}) or {}
    mt = pr.get("mtgs", {}) or {}
    fxc = pr.get("fx", {}) or {}
    sub = pr.get("substitution", {}) or {}
    bu = raw.get("budget", {}) or {}
    hc = raw.get("http", {}) or {}
    th = raw.get("theme", {}) or {}

    return Config(
        sources={
            name: _src(raw.get("sources", {}).get(name, {}))
            for name in ("edhrec", "recommander", "edhtools")
        },
        scoring=ScoringCfg(
            edhrec_signals={
                k: float(v)
                for k, v in (sc.get("edhrec_signals") or {"inclusion": 0.4, "synergy": 0.3, "lift": 0.3}).items()
            },
            prior_mean=float(sc.get("prior_mean", 0.15)),
            prior_weight=float(sc.get("prior_weight", 0.10)),
            agreement_top_fraction=float(sc.get("agreement_top_fraction", 0.20)),
            theme_fit_beta=float(sc.get("theme_fit_beta", 0.15)),
        ),
        quotas=QuotaCfg(
            lands_default=int(qu.get("lands_default", 36)),
            lands_low=int(qu.get("lands_low", 34)),
            lands_high=int(qu.get("lands_high", 38)),
            ramp=int(qu.get("ramp", 12)),
            draw=int(qu.get("draw", 10)),
            removal=int(qu.get("removal", 10)),
            removal_wipes_max=int(qu.get("removal_wipes_max", 2)),
            theme_packages=int(qu.get("theme_packages", 4)),
        ),
        assembly=AssemblyCfg(
            high_cmc_threshold=float(asy.get("high_cmc_threshold", 4.5)),
            dual_land_budget=_dict_int(asy.get("dual_land_budget")),
            fuzzy_match_threshold=float(asy.get("fuzzy_match_threshold", 0.85)),
            filler_value=float(asy.get("filler_value", 0.05)),
            max_resolves=int(asy.get("max_resolves", 3)),
            cheap_pool_size=int(asy.get("cheap_pool_size", 10)),
            solver=SolverCfg(
                enabled=bool(_dict(asy.get("solver")).get("enabled", True)),
                max_doublings=int(_dict(asy.get("solver")).get("max_doublings", 32)),
                bisections=int(_dict(asy.get("solver")).get("bisections", 20)),
                tolerance_nzd=float(_dict(asy.get("solver")).get("tolerance_nzd", 0.50)),
            ),
        ),
        pricing=PricingCfg(
            shipping_nzd=float(pr.get("shipping_nzd", 7.50)),
            condition=str(pr.get("condition", "any")),
            mtgs=MtgSCfg(
                min_interval_sec=float(mt.get("min_interval_sec", 2.1)),
                page_size=int(mt.get("page_size", 20)),
                max_pages=int(mt.get("max_pages", 2)),
                referer=str(mt.get("referer", "https://www.mtgsingles.co.nz/")),
                user_agent=str(mt.get("user_agent", "Mozilla/5.0")),
                block_breaker=int(mt.get("block_breaker", 3)),
            ),
            fx=FxCfg(
                cache_hours=int(fxc.get("cache_hours", 24)),
                base=str(fxc.get("base", "USD")),
                quote=str(fxc.get("quote", "NZD")),
            ),
            substitution=SubstitutionCfg(
                max_candidates=int(sub.get("max_candidates", 5)),
            ),
            estimate_margin=float(pr.get("estimate_margin", 1.10)),
            max_single_card_fraction=float(pr.get("max_single_card_fraction", 0.15)),
            max_probes_per_run=int(pr.get("max_probes_per_run", 150)),
        ),
        budget=BudgetCfg(
            protect_theme_top=int(bu.get("protect_theme_top", 10)),
            min_quality=float(bu.get("min_quality", 0.10)),
            degradation_ladder=bool(bu.get("degradation_ladder", True)),
            floor_fill=bool(bu.get("floor_fill", True)),
        ),
        http=HttpCfg(
            cache_dir=str(hc.get("cache_dir", ".cache")),
            rec_cache_days=int(hc.get("rec_cache_days", 7)),
            price_cache_hours=int(hc.get("price_cache_hours", 24)),
            default_min_interval_sec=float(hc.get("default_min_interval_sec", 1.0)),
            max_retries=int(hc.get("max_retries", 4)),
            timeout_sec=int(hc.get("timeout_sec", 30)),
        ),
        theme=ThemeCfg(
            candidates_default=int(th.get("candidates_default", 5)),
            weight_popularity=float(th.get("weight_popularity", 0.375)),
            weight_synergy_fit=float(th.get("weight_synergy_fit", 0.4375)),
            weight_color=float(th.get("weight_color", 0.1875)),
            color_penalty_per_missing=float(th.get("color_penalty_per_missing", 0.5)),
            unavailable_core_threshold=int(th.get("unavailable_core_threshold", 3)),
            core_probe_max=int(th.get("core_probe_max", 8)),
        ),
    )


def override_source_weights(cfg: Config, weights_json: str | None) -> Config:
    """Apply a --weights JSON override, renormalizing over the named sources.

    Sources named in the override keep their (renormalized) weight; sources NOT
    named are dropped (weight 0.0) so the configured set is exactly the named
    sources. This keeps the shrinkage denominator consistent with the
    numerator: a dropped source contributes nothing.
    """
    if not weights_json:
        return cfg
    raw = json.loads(weights_json)
    unknown = set(raw) - set(cfg.sources)
    if unknown:
        raise ValueError(f"unknown source weights: {sorted(unknown)}")
    total = sum(float(v) for v in raw.values())
    if total <= 0:
        raise ValueError("--weights must sum to a positive number")
    for name, src in cfg.sources.items():
        src.weight = float(raw.get(name, 0.0)) / total
    return cfg
