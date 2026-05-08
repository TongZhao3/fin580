"""
Multi-Agent Macro Trading System
=================================
Three rule-based agents that transform news headlines into tradable signals.

Pipeline:  Headline → Narrative → Macro Interpretation → Asset Mapping
"""

from dataclasses import dataclass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class NarrativeSignal:
    """Output of the Narrative Agent — macro regime classification."""
    headline: str
    inflation: int   #  1 = rising,  0 = neutral, -1 = falling
    growth: int      #  1 = strong,  0 = neutral, -1 = weak
    risk: int        #  1 = risk-on, 0 = neutral, -1 = risk-off


@dataclass
class MacroSignal:
    """Output of the Macro Interpretation Agent — directional views."""
    equity: int      #  1 = bullish, 0 = neutral, -1 = bearish
    bonds: int       #  1 = bullish, 0 = neutral, -1 = bearish
    gold: int        #  1 = bullish, 0 = neutral, -1 = bearish
    dollar: int      #  1 = bullish, 0 = neutral, -1 = bearish


@dataclass
class TradeSignal:
    """Output of the Asset Mapping Agent — per-ETF position sizing."""
    SPY: int         #  1 = long, 0 = flat, -1 = short
    TLT: int
    GLD: int
    UUP: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agent 1: Narrative Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scans a headline for keyword clusters and classifies the
#  macro environment along three axes: inflation, growth, risk.

# Keyword dictionaries — each maps to a (+1) or (-1) signal.
INFLATION_KEYWORDS = {
    +1: [
        "inflation rises", "cpi hot", "cpi beats", "prices surge",
        "inflation accelerates", "cost of living", "wage growth",
        "commodity rally", "oil spikes", "energy prices rise",
        "producer prices rise", "ppi hot", "inflation expectations",
        "stagflation", "food prices surge", "rent increases",
    ],
    -1: [
        "inflation falls", "cpi cools", "cpi misses", "prices drop",
        "deflation", "disinflation", "inflation eases",
        "oil prices fall", "commodity prices drop", "energy prices fall",
        "core inflation slows", "price stability",
    ],
}

GROWTH_KEYWORDS = {
    +1: [
        "gdp beats", "gdp grows", "economy expands", "jobs surge",
        "payrolls beat", "unemployment falls", "retail sales beat",
        "manufacturing expands", "ism beats", "consumer confidence rises",
        "housing starts rise", "industrial production up",
        "strong earnings", "revenue beats", "economic boom",
    ],
    -1: [
        "gdp misses", "gdp contracts", "recession", "jobs miss",
        "payrolls miss", "unemployment rises", "retail sales miss",
        "manufacturing contracts", "ism misses", "consumer confidence drops",
        "layoffs", "downturn", "economic slowdown", "weak earnings",
        "housing starts fall", "credit crunch",
    ],
}

RISK_KEYWORDS = {
    +1: [   # risk-on
        "rally", "all-time high", "stocks surge", "bull market",
        "risk appetite", "buyback", "ipo boom", "market optimism",
        "fed pivot", "rate cut", "stimulus", "dovish fed",
        "ceasefire", "trade deal", "peace talks",
    ],
    -1: [   # risk-off
        "crash", "selloff", "bear market", "volatility spikes",
        "vix surges", "panic", "flight to safety", "contagion",
        "war", "invasion", "sanctions", "tariffs", "trade war",
        "hawkish fed", "rate hike", "bank failure", "default",
        "geopolitical risk", "crisis", "shutdown",
    ],
}


def _score_axis(headline: str, keyword_map: dict[int, list[str]]) -> int:
    """
    Score a headline on a single axis by counting keyword matches.
    Returns +1, 0, or -1 based on the net direction.
    """
    headline_lower = headline.lower()
    score = 0
    for direction, keywords in keyword_map.items():
        for kw in keywords:
            if kw in headline_lower:
                score += direction
    # Clamp to {-1, 0, +1}
    if score > 0:
        return 1
    elif score < 0:
        return -1
    return 0


def narrative_agent(headline: str) -> NarrativeSignal:
    """
    Agent 1 — Narrative Agent
    Classifies a news headline into three macro dimensions.

    Example:
        >>> narrative_agent("CPI hot, inflation accelerates as oil spikes")
        NarrativeSignal(headline='...', inflation=1, growth=0, risk=0)
    """
    return NarrativeSignal(
        headline=headline,
        inflation=_score_axis(headline, INFLATION_KEYWORDS),
        growth=_score_axis(headline, GROWTH_KEYWORDS),
        risk=_score_axis(headline, RISK_KEYWORDS),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agent 2: Macro Interpretation Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Converts the three-axis narrative into directional views on
#  four macro asset classes using standard macro playbook logic.
#
#  Core macro logic:
#  ┌────────────────┬─────────┬───────┬──────┬────────┐
#  │ Regime         │ Equity  │ Bonds │ Gold │ Dollar │
#  ├────────────────┼─────────┼───────┼──────┼────────┤
#  │ Reflation      │   +1    │  -1   │  +1  │  -1    │
#  │ Goldilocks     │   +1    │   0   │   0  │   0    │
#  │ Stagflation    │   -1    │  -1   │  +1  │  -1    │
#  │ Deflation scare│   -1    │  +1   │  +1  │  +1    │
#  │ Risk-off shock │   -1    │  +1   │  +1  │  +1    │
#  │ Risk-on rally  │   +1    │  -1   │  -1  │  -1    │
#  └────────────────┴─────────┴───────┴──────┴────────┘

def _clamp(value: int) -> int:
    """Clamp an integer to {-1, 0, +1}."""
    return max(-1, min(1, value))


def macro_interpretation_agent(signal: NarrativeSignal) -> MacroSignal:
    """
    Agent 2 — Macro Interpretation Agent
    Translates narrative signals into asset-class directional views.

    The logic follows standard macro playbook relationships:
      - Rising inflation  → bearish bonds, bullish gold
      - Strong growth     → bullish equity, bearish bonds
      - Risk-off          → bearish equity, bullish bonds & gold & dollar
      - Risk-on           → bullish equity, bearish bonds & gold & dollar
    """
    inf, grw, rsk = signal.inflation, signal.growth, signal.risk

    # ── Equity: helped by growth & risk-on, hurt by inflation ──
    equity = _clamp(grw + rsk - inf)

    # ── Bonds: helped by weak growth & risk-off, hurt by inflation ──
    bonds = _clamp(-grw - rsk - inf)

    # ── Gold: helped by inflation & risk-off, hurt by strong growth ──
    gold = _clamp(inf - rsk - grw)

    # ── Dollar: helped by growth & hawkish (inflation), hurt by risk-on ──
    #    Rising rates (from inflation) attract capital → strong dollar
    #    But risk-on flows go to equities, weakening safe-haven dollar
    dollar = _clamp(inf + grw - rsk)

    return MacroSignal(equity=equity, bonds=bonds, gold=gold, dollar=dollar)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agent 3: Asset Mapping Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Maps abstract macro views to concrete ETF positions.
#
#  Mapping:
#    equity signal  →  SPY
#    bonds signal   →  TLT
#    gold signal    →  GLD
#    dollar signal  →  UUP

def asset_mapping_agent(macro: MacroSignal) -> TradeSignal:
    """
    Agent 3 — Asset Mapping Agent
    Converts macro directional views into tradable ETF signals.

    Direct 1:1 mapping from asset-class view to ETF position:
      equity → SPY | bonds → TLT | gold → GLD | dollar → UUP
    """
    return TradeSignal(
        SPY=macro.equity,
        TLT=macro.bonds,
        GLD=macro.gold,
        UUP=macro.dollar,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(headline: str) -> dict:
    """
    End-to-end pipeline: headline → narrative → macro → trades.
    Returns a dict with all intermediate outputs for transparency.
    """
    narrative = narrative_agent(headline)
    macro = macro_interpretation_agent(narrative)
    trades = asset_mapping_agent(macro)
    return {
        "headline":  headline,
        "narrative": narrative,
        "macro":     macro,
        "trades":    trades,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    test_headlines = [
        "CPI hot as oil spikes, inflation accelerates beyond expectations",
        "Fed signals rate cut, stocks surge to all-time high",
        "GDP contracts as unemployment rises, recession fears grow",
        "Stagflation fears mount: inflation rises while economy slows",
        "Trade deal reached, markets rally on renewed optimism",
        "Bank failure sparks contagion fears, VIX surges",
        "Goldilocks economy: GDP beats expectations, inflation eases",
    ]

    print("=" * 80)
    print("  MULTI-AGENT MACRO TRADING SYSTEM — Pipeline Demo")
    print("=" * 80)

    for hl in test_headlines:
        result = run_pipeline(hl)
        n = result["narrative"]
        m = result["macro"]
        t = result["trades"]

        print(f"\n📰 {hl}")
        print(f"   Narrative  →  inflation={n.inflation:+d}  growth={n.growth:+d}  risk={n.risk:+d}")
        print(f"   Macro      →  equity={m.equity:+d}  bonds={m.bonds:+d}  gold={m.gold:+d}  dollar={m.dollar:+d}")
        print(f"   Trades     →  SPY={t.SPY:+d}  TLT={t.TLT:+d}  GLD={t.GLD:+d}  UUP={t.UUP:+d}")

    print("\n" + "=" * 80)
