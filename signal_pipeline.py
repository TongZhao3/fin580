"""
Signal Pipeline
================
Applies the three-agent macro system to a news dataset and produces
daily trading signals for SPY, TLT, GLD, UUP.

Pipeline:
  news CSV → Narrative Agent → Macro Agent → Asset Mapping → daily signals

Missing-day handling: forward-fill from last known signal.
"""

import pandas as pd
from agents import narrative_agent, macro_interpretation_agent, asset_mapping_agent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 0: Sample News Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Replace this with your own CSV: pd.read_csv("news.csv")
#  Required columns: date, headline

SAMPLE_NEWS = [
    # ── 2024 Q1: Disinflation + soft landing optimism ──────────
    ("2024-01-05", "Jobs surge past expectations, unemployment falls to 3.7%"),
    ("2024-01-12", "CPI cools for third straight month, inflation eases"),
    ("2024-01-19", "Fed signals rate cut timeline, stocks surge"),
    ("2024-02-02", "Strong earnings from tech giants, revenue beats across the board"),
    ("2024-02-14", "Retail sales miss expectations, consumer spending weakens"),
    ("2024-03-01", "GDP beats expectations as economy expands at 3.2% pace"),
    ("2024-03-15", "CPI hot, prices surge on sticky shelter costs"),
    ("2024-03-22", "Fed holds rates, hawkish Fed warns cuts may be delayed"),

    # ── 2024 Q2: Inflation reignites + geopolitical risk ───────
    ("2024-04-05", "Oil spikes on Middle East tensions, energy prices rise"),
    ("2024-04-12", "PPI hot, producer prices rise more than expected"),
    ("2024-04-26", "GDP misses in Q1 revision, economic slowdown fears grow"),
    ("2024-05-03", "Payrolls miss badly, unemployment rises to 3.9%"),
    ("2024-05-17", "Inflation accelerates, CPI beats expectations again"),
    ("2024-06-07", "Stagflation fears mount: inflation rises while economy slows"),
    ("2024-06-14", "Trade deal reached between US and EU, markets rally"),
    ("2024-06-28", "Consumer confidence drops to 2-year low"),

    # ── 2024 Q3: Risk-off + recession scare ────────────────────
    ("2024-07-05", "Manufacturing contracts for fifth straight month"),
    ("2024-07-19", "Bank failure at regional lender sparks contagion fears"),
    ("2024-07-26", "VIX surges as selloff accelerates, panic selling"),
    ("2024-08-02", "Flight to safety: Treasury bonds rally as stocks crash"),
    ("2024-08-16", "Layoffs mount across tech sector, downturn deepens"),
    ("2024-08-30", "Fed pivot expected, rate cut priced in for September"),
    ("2024-09-06", "Ceasefire agreement reduces geopolitical risk"),
    ("2024-09-20", "GDP grows at 2.8%, recession fears fade"),

    # ── 2024 Q4: Recovery + Goldilocks ─────────────────────────
    ("2024-10-04", "Jobs surge, payrolls beat by wide margin"),
    ("2024-10-18", "Inflation falls to 2.1%, disinflation trend intact"),
    ("2024-10-25", "Stocks surge to all-time high on rate cut hopes"),
    ("2024-11-01", "Strong earnings across sectors, bull market confirmed"),
    ("2024-11-15", "ISM beats, manufacturing expands for first time in months"),
    ("2024-11-29", "Oil prices fall on weak global demand outlook"),
    ("2024-12-06", "Fed announces rate cut, dovish Fed boosts market optimism"),
    ("2024-12-20", "GDP beats, inflation eases — Goldilocks economy confirmed"),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 1: Process Headlines Through the Agent Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_news(news_data: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Run each headline through the full 3-agent pipeline.
    Returns a DataFrame with one row per headline and all
    intermediate signals for auditability.
    """
    records = []
    for date_str, headline in news_data:
        # Agent 1: Narrative
        narrative = narrative_agent(headline)

        # Agent 2: Macro Interpretation
        macro = macro_interpretation_agent(narrative)

        # Agent 3: Asset Mapping
        trades = asset_mapping_agent(macro)

        records.append({
            "date":           pd.Timestamp(date_str),
            "headline":       headline,
            # Narrative layer
            "inflation":      narrative.inflation,
            "growth":         narrative.growth,
            "risk":           narrative.risk,
            # Macro layer
            "equity_view":    macro.equity,
            "bonds_view":     macro.bonds,
            "gold_view":      macro.gold,
            "dollar_view":    macro.dollar,
            # Trade layer
            "SPY_signal":     trades.SPY,
            "TLT_signal":     trades.TLT,
            "GLD_signal":     trades.GLD,
            "UUP_signal":     trades.UUP,
        })

    return pd.DataFrame(records)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 2: Handle Multiple Headlines Per Day
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def aggregate_daily_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple headlines land on the same day, average their
    signals and snap to {-1, 0, +1} using sign().

    This simulates a "consensus" across the day's news flow.
    """
    signal_cols = ["SPY_signal", "TLT_signal", "GLD_signal", "UUP_signal"]

    daily = (
        df.groupby("date")[signal_cols]
        .mean()                     # average if multiple headlines
        .apply(lambda x: x.map(    # snap back to {-1, 0, +1}
            lambda v: int(pd.np.sign(v)) if v != 0 else 0
        ))
    )
    return daily


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 3: Build Full Daily Signal Grid with Forward Fill
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_daily_signals(
    news_data: list[tuple[str, str]],
    start: str = "2024-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Full pipeline:
      1. Process every headline through the 3-agent system
      2. Aggregate same-day signals
      3. Reindex to a continuous business-day calendar
      4. Forward-fill gaps (no news → hold previous signal)
      5. Fill any leading NaN with 0 (flat / no position)

    Returns a clean DataFrame: date × {SPY, TLT, GLD, UUP}_signal
    """
    # Run the agent pipeline
    raw = process_news(news_data)

    # Aggregate same-day headlines
    signal_cols = ["SPY_signal", "TLT_signal", "GLD_signal", "UUP_signal"]
    daily = raw.groupby("date")[signal_cols].mean()

    # Snap averaged values back to {-1, 0, +1}
    import numpy as np
    daily = daily.apply(lambda col: col.apply(
        lambda v: int(np.sign(v)) if v != 0 else 0
    ))

    # Reindex to full business-day calendar
    bdays = pd.bdate_range(start=start, end=end)
    daily = daily.reindex(bdays)

    # Forward-fill: no news today → keep yesterday's signal
    daily = daily.ffill()

    # Fill any leading NaN (before first headline) with 0
    daily = daily.fillna(0).astype(int)

    # Clean up index
    daily.index.name = "date"

    return daily


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # ── Run full pipeline ──────────────────────────────────────
    signals = build_daily_signals(SAMPLE_NEWS)

    # ── Save to CSV ────────────────────────────────────────────
    signals.to_csv("signals.csv")
    print(f"✅ Saved daily signals → signals.csv")
    print(f"   {len(signals)} business days from {signals.index[0].date()} to {signals.index[-1].date()}\n")

    # ── Preview ────────────────────────────────────────────────
    print("── First 20 days ──")
    print(signals.head(20))

    print("\n── Signal transitions (days where signal changed) ──")
    changes = signals.diff().abs().sum(axis=1)
    transition_days = signals[changes > 0]
    print(transition_days)

    # ── Summary stats ──────────────────────────────────────────
    print("\n── Signal distribution ──")
    for col in signals.columns:
        counts = signals[col].value_counts().sort_index()
        total = len(signals)
        print(f"  {col}:  "
              f"short={counts.get(-1, 0):3d} ({counts.get(-1, 0)/total:.0%})  "
              f"flat={counts.get(0, 0):3d} ({counts.get(0, 0)/total:.0%})  "
              f"long={counts.get(1, 0):3d} ({counts.get(1, 0)/total:.0%})")
