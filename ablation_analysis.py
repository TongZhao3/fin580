"""
Ablation Analysis
==================
Systematically removes components of the multi-agent system
to measure each agent's marginal contribution.

Variants:
  A) Full Model        — Narrative → Macro Interp → Asset Mapping  (with risk mgmt)
  B) No Macro Agent    — Narrative signals mapped directly to ETFs  (skip macro layer)
  C) No Risk Mgmt      — Full model but no txn costs, no position limits, allow leverage

Pipeline architecture being tested:
  ┌──────────┐     ┌───────────────┐     ┌──────────────┐     ┌────────────┐
  │ Headlines│ ──→ │ Narrative     │ ──→ │ Macro Interp │ ──→ │ Asset Map  │
  │          │     │ (sentiment)   │     │ (playbook)   │     │ (ETFs)     │
  └──────────┘     └───────────────┘     └──────────────┘     └────────────┘
                    always present        removed in (B)       always present
                                          risk mgmt removed in (C)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from agents import (
    narrative_agent,
    macro_interpretation_agent,
    asset_mapping_agent,
    NarrativeSignal,
    MacroSignal,
    TradeSignal,
    _clamp,
)
from signal_pipeline import SAMPLE_NEWS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TICKERS      = ["SPY", "TLT", "GLD", "UUP"]
WEIGHT       = 1.0 / len(TICKERS)            # 25% per asset
TC_PER_TURN  = 0.003                          # 30 bps
ANN_FACTOR   = 252
OUTPUT_PLOT  = "ablation_results.png"
OUTPUT_CSV   = "ablation_results.csv"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ablation Variant: No Macro Agent (B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Skip the Macro Interpretation Agent entirely.
#  Map narrative sentiment directly to ETFs using a naive heuristic:
#
#    risk > 0 (risk-on sentiment)    → long SPY
#    risk < 0 (risk-off sentiment)   → long TLT, long GLD, long UUP
#    inflation signal                → long GLD, long UUP
#    growth signal                   → long SPY
#
#  This tests whether the structured macro playbook (Agent 2)
#  adds value over a simple sentiment-to-asset direct mapping.

def naive_sentiment_mapping(narrative: NarrativeSignal) -> TradeSignal:
    """
    Bypass Macro Interpretation Agent.
    Map raw narrative axes directly to ETFs with simple heuristics:
      - SPY follows risk + growth  (risk-on & growth = buy stocks)
      - TLT follows inverse risk   (risk-off = buy bonds)
      - GLD follows inflation      (inflation hedge)
      - UUP follows growth         (strong economy = strong dollar)

    No cross-asset macro reasoning — each ETF responds to only one axis.
    """
    return TradeSignal(
        SPY=_clamp(narrative.risk + narrative.growth),
        TLT=_clamp(-narrative.risk),
        GLD=_clamp(narrative.inflation),
        UUP=_clamp(narrative.growth),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Signal Generation for Each Variant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_signals(news_data, pipeline_fn, start="2024-01-01", end="2024-12-31"):
    """
    Generic signal generator — takes a pipeline function
    that converts a headline → TradeSignal, then builds
    the full daily signal grid with forward fill.
    """
    records = []
    for date_str, headline in news_data:
        trade = pipeline_fn(headline)
        records.append({
            "date":       pd.Timestamp(date_str),
            "SPY_signal": trade.SPY,
            "TLT_signal": trade.TLT,
            "GLD_signal": trade.GLD,
            "UUP_signal": trade.UUP,
        })

    df = pd.DataFrame(records)
    signal_cols = ["SPY_signal", "TLT_signal", "GLD_signal", "UUP_signal"]

    daily = df.groupby("date")[signal_cols].mean()
    daily = daily.apply(lambda col: col.apply(
        lambda v: int(np.sign(v)) if v != 0 else 0
    ))

    bdays = pd.bdate_range(start=start, end=end)
    daily = daily.reindex(bdays).ffill().fillna(0).astype(int)
    daily.index.name = "date"
    daily.columns = TICKERS

    return daily


# Pipeline functions for each variant
def pipeline_full(headline: str) -> TradeSignal:
    """(A) Full model: Narrative → Macro → Asset Mapping"""
    n = narrative_agent(headline)
    m = macro_interpretation_agent(n)
    return asset_mapping_agent(m)


def pipeline_no_macro(headline: str) -> TradeSignal:
    """(B) No Macro Agent: Narrative → direct naive mapping"""
    n = narrative_agent(headline)
    return naive_sentiment_mapping(n)


def pipeline_no_risk(headline: str) -> TradeSignal:
    """(C) No Risk Mgmt: same signals as full model (risk mgmt is in the backtest)"""
    return pipeline_full(headline)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Backtesting Engine (with risk management toggle)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest_variant(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    apply_tc: bool = True,
    apply_position_limits: bool = True,
) -> pd.Series:
    """
    Backtest a signal set with optional risk management controls.

    Risk management controls:
      - Transaction costs (30 bps per turnover)
      - Position limits (signals clamped to {-1, 0, +1})
      - If risk mgmt OFF: no costs, signals can stack (sum > 1 per asset)

    Position at t uses signal from t-1 (always — look-ahead avoidance is fundamental).
    """
    # Lag signals by 1 day
    positions = signals.shift(1).fillna(0)

    if apply_position_limits:
        # Clamp each position to [-1, +1]
        positions = positions.clip(-1, 1).astype(int)

    # Weighted returns
    strat_returns = (positions * returns * WEIGHT).sum(axis=1)

    # Transaction costs
    if apply_tc:
        turnover = positions.diff().abs().fillna(0)
        tc = (turnover * WEIGHT * TC_PER_TURN).sum(axis=1)
        strat_returns = strat_returns - tc

    return strat_returns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(daily: pd.Series) -> dict:
    """Standard performance metrics."""
    total   = (1 + daily).prod() - 1
    n       = len(daily)
    ann_ret = (1 + total) ** (ANN_FACTOR / n) - 1
    ann_vol = daily.std() * np.sqrt(ANN_FACTOR)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum      = (1 + daily).cumprod()
    dd       = (cum / cum.cummax()) - 1
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (daily > 0).mean()

    return {
        "total_return":  total,
        "annual_return": ann_ret,
        "annual_vol":    ann_vol,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "calmar":        calmar,
        "win_rate":      win_rate,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plotting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_ablation(results: dict, output_path: str):
    """Three-panel ablation chart: cumulative, drawdown, bar comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0f0f14",
                             gridspec_kw={"height_ratios": [2, 1.2]})
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    palette = {
        "(A) Full Model":     "#00e676",
        "(B) No Macro Agent": "#ff9100",
        "(C) No Risk Mgmt":   "#e040fb",
    }
    bg   = "#0f0f14"
    grid = "#1a1a24"
    text = "#c0c0c8"

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor(bg)
            ax.tick_params(colors=text, labelsize=9)
            ax.grid(True, color=grid, linewidth=0.5, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color(grid)

    # ── Panel 1 (top-left): Cumulative returns ─────────────────
    ax1 = axes[0, 0]
    for name, data in results.items():
        cum = (1 + data["daily"]).cumprod()
        c = palette[name]
        ax1.plot(cum, color=c, linewidth=1.8, label=name, zorder=3)
        ax1.fill_between(cum.index, 1, cum.values, color=c, alpha=0.04)

    ax1.axhline(1.0, color=grid, linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_title("Cumulative Returns", fontsize=14, fontweight="bold",
                  color="white", pad=10)
    ax1.set_ylabel("Growth of $1", fontsize=10, color=text)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.4,
               facecolor=bg, edgecolor=grid, labelcolor=text)

    # ── Panel 2 (top-right): Drawdowns ─────────────────────────
    ax2 = axes[0, 1]
    for name, data in results.items():
        cum = (1 + data["daily"]).cumprod()
        dd = (cum / cum.cummax()) - 1
        c = palette[name]
        ax2.fill_between(dd.index, 0, dd.values, color=c, alpha=0.2, linewidth=0)
        ax2.plot(dd, color=c, linewidth=0.8, alpha=0.8, label=name)

    ax2.set_title("Drawdown Comparison", fontsize=14, fontweight="bold",
                  color="white", pad=10)
    ax2.set_ylabel("Drawdown", fontsize=10, color=text)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.4,
               facecolor=bg, edgecolor=grid, labelcolor=text)

    # ── Panel 3 (bottom-left): Sharpe bar chart ────────────────
    ax3 = axes[1, 0]
    names = list(results.keys())
    sharpes = [results[n]["metrics"]["sharpe"] for n in names]
    colors_bar = [palette[n] for n in names]

    bars = ax3.bar(range(len(names)), sharpes, color=colors_bar, alpha=0.85,
                   edgecolor="white", linewidth=0.5, width=0.6)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n.split(") ")[1] for n in names], fontsize=10, color=text)
    ax3.set_title("Sharpe Ratio", fontsize=14, fontweight="bold",
                  color="white", pad=10)
    ax3.axhline(0, color=text, linewidth=0.5, alpha=0.5)
    for bar, val in zip(bars, sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:+.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="white")

    # ── Panel 4 (bottom-right): Max Drawdown bar chart ─────────
    ax4 = axes[1, 1]
    max_dds = [results[n]["metrics"]["max_drawdown"] * 100 for n in names]

    bars = ax4.bar(range(len(names)), max_dds, color=colors_bar, alpha=0.85,
                   edgecolor="white", linewidth=0.5, width=0.6)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n.split(") ")[1] for n in names], fontsize=10, color=text)
    ax4.set_title("Max Drawdown", fontsize=14, fontweight="bold",
                  color="white", pad=10)
    ax4.axhline(0, color=text, linewidth=0.5, alpha=0.5)
    for bar, val in zip(bars, max_dds):
        y_pos = val - 0.3 if val < 0 else val + 0.1
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f"{val:.1f}%", ha="center", va="top",
                 fontsize=11, fontweight="bold", color="white")

    fig.suptitle("Ablation Analysis — Multi-Agent Macro System",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=bg, edgecolor="none")
    plt.close()
    print(f"📊 Chart saved → {output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reporting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_ablation_report(results: dict):
    """Print the ablation comparison table and analysis."""

    header = f"{'Metric':<22}"
    for name in results:
        header += f"  {name:>22}"
    sep = "─" * len(header)

    print()
    print("=" * len(header))
    print("  ABLATION ANALYSIS — Component Contribution")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    rows = [
        ("Total Return",   "total_return",  "{:+.2%}"),
        ("Annual Return",  "annual_return", "{:+.2%}"),
        ("Annual Vol",     "annual_vol",    "{:.2%}"),
        ("Sharpe Ratio",   "sharpe",        "{:+.3f}"),
        ("Max Drawdown",   "max_drawdown",  "{:.2%}"),
        ("Calmar Ratio",   "calmar",        "{:.3f}"),
        ("Win Rate",       "win_rate",      "{:.1%}"),
    ]

    for label, key, fmt in rows:
        row = f"  {label:<20}"
        for name in results:
            val = results[name]["metrics"][key]
            row += f"  {fmt.format(val):>22}"
        print(row)

    print(sep)

    # ── Component attribution ──────────────────────────────────
    m_full    = results["(A) Full Model"]["metrics"]
    m_nomacro = results["(B) No Macro Agent"]["metrics"]
    m_norisk  = results["(C) No Risk Mgmt"]["metrics"]

    macro_delta_sharpe = m_full["sharpe"] - m_nomacro["sharpe"]
    macro_delta_ret    = m_full["total_return"] - m_nomacro["total_return"]
    macro_delta_dd     = m_full["max_drawdown"] - m_nomacro["max_drawdown"]

    risk_delta_sharpe  = m_full["sharpe"] - m_norisk["sharpe"]
    risk_delta_ret     = m_full["total_return"] - m_norisk["total_return"]
    risk_delta_dd      = m_full["max_drawdown"] - m_norisk["max_drawdown"]

    print()
    print("── Component Attribution (delta from removing) ──")
    print()
    print(f"  {'Component':<24} {'Δ Return':>12} {'Δ Sharpe':>12} {'Δ MaxDD':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'Macro Interp Agent':<24} {macro_delta_ret:>+11.2%} {macro_delta_sharpe:>+12.3f} {macro_delta_dd:>+11.2%}")
    print(f"  {'Risk Management':<24} {risk_delta_ret:>+11.2%} {risk_delta_sharpe:>+12.3f} {risk_delta_dd:>+11.2%}")
    print()

    # ── Interpretation ─────────────────────────────────────────
    print("── Interpretation ──")
    print()

    # Macro Agent value
    if macro_delta_sharpe > 0:
        print("  ✅ Macro Interpretation Agent ADDS value:")
        print(f"     Removing it worsens Sharpe by {abs(macro_delta_sharpe):.3f}.")
        print("     The structured macro playbook (inflation→bonds, risk→equity)")
        print("     provides better cross-asset allocation than naive sentiment mapping.")
    elif macro_delta_sharpe < 0:
        print("  ⚠️  Macro Interpretation Agent HURTS performance:")
        print(f"     Removing it improves Sharpe by {abs(macro_delta_sharpe):.3f}.")
        print("     The playbook rules may be misaligned with the 2024 macro regime.")
        print("     Direct sentiment mapping captured asset moves more accurately.")
    else:
        print("  ➖ Macro Interpretation Agent has NEUTRAL impact.")

    print()

    # Risk Management value
    if risk_delta_dd > 0:
        print("  ✅ Risk Management ADDS value:")
        print(f"     Removing it deepens max drawdown by {abs(risk_delta_dd):.2%}.")
        print("     Transaction costs + position limits prevent overtrading and")
        print("     reduce whipsaw losses from frequent signal changes.")
    elif risk_delta_ret < 0:
        print("  ⚠️  Risk Management COSTS returns:")
        print(f"     Transaction costs drag returns by {abs(risk_delta_ret):.2%}.")
        if risk_delta_dd > 0:
            print("     But drawdown protection is worth the cost (lower MaxDD).")
        else:
            print("     And drawdown protection is minimal in this sample.")
    else:
        print("  ➖ Risk Management has NEUTRAL impact.")

    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":

    # ── Load price data ────────────────────────────────────────
    prices  = pd.read_csv("prices.csv", index_col=0, parse_dates=True)[TICKERS]
    returns = prices.pct_change().dropna()

    print(f"Price data: {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"Trading days: {len(returns)}\n")

    # ── Generate signals for each variant ──────────────────────
    print("Generating signals for each ablation variant...")

    sig_full     = generate_signals(SAMPLE_NEWS, pipeline_full)
    sig_no_macro = generate_signals(SAMPLE_NEWS, pipeline_no_macro)
    sig_no_risk  = generate_signals(SAMPLE_NEWS, pipeline_no_risk)  # same signals, diff backtest

    # Align to common dates
    common = returns.index.intersection(sig_full.index)
    ret = returns.loc[common]

    # ── Backtest each variant ──────────────────────────────────
    results = {}

    # (A) Full Model — all agents + risk management
    daily_a = backtest_variant(ret, sig_full.loc[common],
                               apply_tc=True, apply_position_limits=True)
    results["(A) Full Model"] = {"daily": daily_a, "metrics": compute_metrics(daily_a)}

    # (B) No Macro Agent — narrative directly to ETFs + risk management
    daily_b = backtest_variant(ret, sig_no_macro.loc[common],
                               apply_tc=True, apply_position_limits=True)
    results["(B) No Macro Agent"] = {"daily": daily_b, "metrics": compute_metrics(daily_b)}

    # (C) No Risk Mgmt — full model but no txn costs, no position limits
    daily_c = backtest_variant(ret, sig_no_risk.loc[common],
                               apply_tc=False, apply_position_limits=False)
    results["(C) No Risk Mgmt"] = {"daily": daily_c, "metrics": compute_metrics(daily_c)}

    # ── Report ─────────────────────────────────────────────────
    print_ablation_report(results)

    # ── Plot ───────────────────────────────────────────────────
    plot_ablation(results, OUTPUT_PLOT)

    # ── Save daily returns ─────────────────────────────────────
    out_df = pd.DataFrame({
        name: data["daily"] for name, data in results.items()
    })
    out_df.to_csv(OUTPUT_CSV)
    print(f"💾 Daily returns saved → {OUTPUT_CSV}")
