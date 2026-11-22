"""Deterministic role tagging from oracle text / type line.

Roles (per the commandertemplate.com guide):
  ramp        - non-land mana acceleration (tutors, mana dorks, rocks)
  draw        - card advantage (draw, library search for a card, scry)
  removal     - single-target creature removal or counterspells
  wipe        - mass removal (counts toward the removal quota, capped)
  protection  - counter prevention / untargetable / protection
  (theme)     - everything else fills the theme packages
"""
from __future__ import annotations

from .cards import Card

_RAMP_MARKERS = (
    "tap: add",
    "tap and sacrifice",
    "sacrifice: add",
    "search your library for a basic land",
    "put a basic land card from your library",
    "put a land card from your library",
    "put onto the battlefield a basic land card",
    "you may add mana",
)

_DRAW_MARKERS = (
    "draw a card",
    "draw an additional card",
    "draw two cards",
    "draw three cards",
    "draw four cards",
    "draw 2 cards",
    "search your library for a non-land card",
    "search your library for a card",
    "scry",
)

_SINGLE_REMOVAL_MARKERS = (
    "destroy target creature",
    "put target creature on the bottom",
    "exile target creature",
    "target creature gets",
    "-x/-x until end of turn",
)

_COUNTER_MARKERS = ("counter target spell",)

_WIPE_MARKERS = (
    "all creatures",
    "each creature you control",
    "each creature an opponent controls",
    "destroy target creature and all other",
)

_PROTECTION_MARKERS = (
    "can't be countered",
    "can't be targeted",
    "can't be destroyed",
    "can't be exiled",
    "shroud",
    "hexproof",
    "protection from",
)


def tag(card: Card) -> dict[str, bool]:
    """Tag a card with every role it plausibly fills (heuristic, deterministic)."""
    text = (card.oracle_text or "").lower()
    type_line = card.type_line.lower()

    is_land = "land" in card.card_types
    ramp = (not is_land) and any(m in text for m in _RAMP_MARKERS)
    draw = any(m in text for m in _DRAW_MARKERS)
    single_removal = any(m in text for m in _SINGLE_REMOVAL_MARKERS)
    counter = any(m in text for m in _COUNTER_MARKERS)
    wipe = any(m in text for m in _WIPE_MARKERS)
    protection = any(m in text for m in _PROTECTION_MARKERS)

    # "target creature gets -X/-X" is only removal-ish when the effect is
    # actually negative; keep the simple marker but require "-x/-x".
    if single_removal and not any(
        m in text for m in ("destroy", "put target creature", "exile target creature", "-x/-x")
    ):
        single_removal = False

    return {
        "ramp": ramp,
        "draw": draw,
        "removal": single_removal or counter,
        "single_removal": single_removal,
        "counter": counter,
        "wipe": wipe,
        "protection": protection,
        "land": is_land,
    }
