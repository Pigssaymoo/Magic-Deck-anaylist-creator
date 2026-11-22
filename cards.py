"""Core card model shared by every module."""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical color ordering used everywhere (pips, identity).
COLOR_ORDER = ("W", "U", "B", "R", "G")


def ordered_identity(colors: list[str] | tuple[str, ...] | str) -> list[str]:
    """Return a color identity sorted in WUBRG order, deduplicated."""
    seen = {c for c in (colors or []) if c in COLOR_ORDER}
    return [c for c in COLOR_ORDER if c in seen]


@dataclass
class Card:
    """A single canonical card (one Scryfall card definition, all printings)."""

    name: str
    scryfall_id: str
    oracle_id: str
    mana_cost: str = ""
    cmc: int = 0
    type_line: str = ""
    colors: list[str] = field(default_factory=list)
    color_identity: list[str] = field(default_factory=list)
    card_types: list[str] = field(default_factory=list)
    oracle_text: str = ""
    mv_usd: float | None = None
    printed_in: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    commander_legal: bool = False

    # ------------------------------------------------------------ properties
    @property
    def is_land(self) -> bool:
        return "land" in self.card_types

    @property
    def is_basic_land(self) -> bool:
        return "basic" in self.type_line.lower() and self.is_land

    @property
    def is_creature(self) -> bool:
        return "creature" in self.card_types

    @property
    def is_commander_capable(self) -> bool:
        return self.commander_legal and self.is_creature and "legendary" in self.card_types

    @property
    def is_companion(self) -> bool:
        return "companion" in self.keywords or "companion —" in (self.oracle_text or "").lower()

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.name} ({self.type_line})"
