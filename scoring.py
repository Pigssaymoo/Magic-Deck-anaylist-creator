"""Cross-source consensus scoring.

Each source signal is normalized to 0-1 *within its own response* (rank
percentile), then blended with a shrinkage prior:

    base = (sum_present w_i * s_i + w0 * mu0) / (sum_all w_i + w0)

where the sums run over ALL configured sources (blocked/absent sources
contribute zero to the numerator, no per-card renormalization), w0 =
``scoring.prior_weight``, mu0 = ``scoring.prior_mean``.

Multipliers:
  * agreement   - 1 + min(0.10, 0.05 * (n-1)); n = number of sources whose own
                  list puts the card in its top ``agreement_top_fraction``
  * theme fit   - 1 + theme_fit_beta * theme_fit; theme_fit is a pool-relative
                  modifier stored on the candidate (None -> factor 1.0)

The final value is clamped to [0.0, 1.25]. No LLM, no randomness: same
inputs -> same scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .cards import Card
from .config import Config
from .edhrec import EdhrecData
from .edhtools import EdhToolsData
from .recommander import RecommanderData
from .roles import tag
from .scryfall import ScryfallDB, find_card


@dataclass
class SourceSignal:
    source: str
    norm: float
    rank: int
    n: int
    detail: dict = field(default_factory=dict)


@dataclass
class Candidate:
    name: str
    card: Card
    signals: dict[str, SourceSignal] = field(default_factory=dict)
    roles: dict[str, bool] = field(default_factory=dict)
    score: float = 0.0
    package_hint: int | None = None
    # Scoring components (populated by consensus_score; reported per card):
    base: float = 0.0  # shrinkage consensus base, before multipliers
    coverage: float = 0.0  # weight of present sources / weight of all configured
    agreement_n: int = 0  # sources whose own top-fraction includes this card
    theme_fit: float | None = None  # pool-relative theme affinity; None = no data


def edhrec_blend(sig_norms: dict[str, float], cfg: Config) -> float:
    """Blend EDHREC's three normalized signals with config weights (renormalized)."""
    weights = cfg.scoring.edhrec_signals
    total_w = sum(float(weights.get(k, 0.0)) for k in sig_norms)
    if total_w <= 0:
        return 0.0
    return sum(sig_norms[k] * float(weights.get(k, 0.0)) for k in sig_norms) / total_w


def _is_top(sig: SourceSignal, top_fraction: float) -> bool:
    """True if the card is within the top ``top_fraction`` of its source list."""
    if sig.n <= 0 or sig.rank is None:
        return False
    cutoff = max(1, math.ceil(top_fraction * sig.n))
    return sig.rank <= cutoff


def consensus_score(cand: Candidate, cfg: Config) -> float:
    """Shrinkage consensus base x agreement factor x theme-fit factor.

    base = (sum_present w_i*s_i + w0*mu0) / (sum_all w_i + w0)

    The numerator sums only the sources that actually returned the card (their
    weight x normalized score); the denominator sums ALL configured sources
    (including blocked/absent ones) plus the prior weight, so a run with fewer
    working sources yields LOWER, never inflated, scores and no per-card
    renormalization happens. The prior (w0*mu0) keeps single-source cards on a
    sane floor.

    Multipliers: agreement rewards the card being in the top fraction of
    several sources' own lists; theme_fit is a bounded pool-relative modifier.
    Result clamped to [0.0, 1.25]. Components are stored on the candidate.
    """
    sc = cfg.scoring
    total_all = sum(cfg.source_weight(name) for name in cfg.sources)

    num = sc.prior_weight * sc.prior_mean
    present_w = 0.0
    for name, sig in cand.signals.items():
        w = cfg.source_weight(name)
        num += w * sig.norm
        present_w += w
    base = num / (total_all + sc.prior_weight)

    n = sum(1 for s in cand.signals.values() if _is_top(s, sc.agreement_top_fraction))
    agreement = 1.0 + min(0.10, 0.05 * max(0, n - 1))

    theme_fit = cand.theme_fit
    theme_factor = 1.0 + sc.theme_fit_beta * theme_fit if theme_fit is not None else 1.0

    value = base * agreement * theme_factor
    value = max(0.0, min(1.25, value))

    cand.base = base
    cand.coverage = (present_w / total_all) if total_all > 0 else 0.0
    cand.agreement_n = n
    return value


def _canonicalize(db: ScryfallDB, name: str, identity: set[str]) -> str | None:
    """Map a source's raw name to a canonical, in-identity, legal card name."""
    card = find_card(db, name, threshold=0.9)
    if card is None:
        return None
    if not card.commander_legal:
        return None
    if not set(card.color_identity) <= identity:
        return None
    return card.name


def merge_candidates(
    db: ScryfallDB,
    cfg: Config,
    identity: list[str],
    commanders: list[Card],
    edhrec: EdhrecData | None,
    recommander: RecommanderData | None,
    edhtools: EdhToolsData | None,
) -> dict[str, Candidate]:
    """Merge all source responses into canonical, scored candidates."""
    idset = set(identity)
    commander_names = {c.name.casefold() for c in commanders}
    cands: dict[str, Candidate] = {}

    def get(canon: str) -> Candidate:
        card = db.by_name[canon.casefold()]
        c = cands.get(canon)
        if c is None:
            c = Candidate(name=canon, card=card, roles=tag(card))
            cands[canon] = c
        return c

    if edhrec is not None:
        for raw_name, sig in edhrec.signals.items():
            canon = _canonicalize(db, raw_name, idset)
            if canon is None or canon.casefold() in commander_names:
                continue
            c = get(canon)
            if c.card is None:
                c.card = db.by_name[canon.casefold()]
            c.signals["edhrec"] = SourceSignal(
                source="edhrec",
                norm=edhrec_blend(
                    {
                        "inclusion": sig.norm_inclusion,
                        "synergy": sig.norm_synergy,
                        "lift": sig.norm_lift,
                    },
                    cfg,
                ),
                rank=sig.rank,
                n=sig.n,
                detail={"lift": sig.lift, "inclusion": sig.inclusion, "synergy": sig.synergy},
            )
            if c.package_hint is None and sig.package_hint is not None:
                c.package_hint = sig.package_hint

    if recommander is not None:
        for raw_name, sig in recommander.signals.items():
            canon = _canonicalize(db, raw_name, idset)
            if canon is None or canon.casefold() in commander_names:
                continue
            c = get(canon)
            c.signals["recommander"] = SourceSignal(
                source="recommander",
                norm=sig.norm,
                rank=sig.rank,
                n=sig.n,
                detail={"raw_score": sig.raw_score},
            )

    if edhtools is not None:
        for raw_name, sig in edhtools.signals.items():
            canon = _canonicalize(db, raw_name, idset)
            if canon is None or canon.casefold() in commander_names:
                continue
            c = get(canon)
            c.signals["edhtools"] = SourceSignal(
                source="edhtools",
                norm=sig.norm,
                rank=sig.rank,
                n=sig.n,
                detail={"raw_score": sig.score},
            )

    for c in cands.values():
        c.score = consensus_score(c, cfg)
    return cands
