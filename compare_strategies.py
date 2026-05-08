"""
Strategy Comparison
====================
Compares three strategies over the same period:
  1. Multi-Agent Macro Strategy  (from signals.csv)
  2. Buy-and-Hold SPY
  3. 20-Day Momentum on SPY

All use prices.csv as the common price source.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TICKERS      = ["SPY", "TLT", "GLD", "UUP"]
WEIGHT       = 1.0 / len(TICKERS)
TC_PER_TURN  = 0.003
MOM_WINDOW   = 20       # lookback for momentum signal
ANN_FACTOR   = 252
OUTPUT_PLOT  = "strategy_comparison.png"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def metrics(daily: pd.Series) -> dict:
    """Compute standard performance metrics from a daily return series."""
    total    = (1 + daily).prod() - 1
    n        = len(daily)
    ann_ret  = (1 + total) ** (ANN_FACTOR / n) - 1
    ann_vol  = daily.std() * np.sqrt(ANN_FACTOR)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum       = (1 + daily).cumprod()
    drawdown  = (cum / cum.cummax()) - 1
    max_dd    = drawdown.min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    win_rate  = (daily > 0).mean()

    return {
        "total_return":  total,
        "annual_return": ann_ret,
        "annual_vol":    ann_vol,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "calmar":        calmar,
        "win_rate":      win_rate,
        "n_days":        n,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy 1: Multi-Agent (reuse existing backtest logic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def multi_agent_strategy(returns: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
    """
    Multi-agent macro strategy.
    Position at t uses signal from t-1. Equal weight. Includes txn costs.
    """
    positions = signals.shift(1).fillna(0).astype(int)
    gross = (positions * returns * WEIGHT).sum(axis=1)

    # Transaction costs
    turnover = positions.diff().abs().fillna(0)
    tc = (turnover * WEIGHT * TC_PER_TURN).sum(axis=1)

    return gross - tc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy 2: Buy-and-Hold SPY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def buy_and_hold_spy(returns: pd.DataFrame) -> pd.Series:
    """Simply hold SPY every day — no signals, no costs."""
    return returns["SPY"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy 3: 20-Day Momentum on SPY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def momentum_spy(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Momentum signal on SPY:
      - If cumulative return over the past 20 days > 0 → long SPY
      - Otherwise → cash (0% return)

    Signal is lagged by 1 day to avoid look-ahead bias.
    """
    # Past 20-day return at each point
    rolling_ret = prices["SPY"].pct_change(MOM_WINDOW)

    # Signal: 1 if positive momentum, 0 otherwise
    raw_signal = (rolling_ret > 0).astype(int)

    # Lag by 1 day
    position = raw_signal.shift(1).reindex(returns.index).fillna(0).astype(int)

    # Daily return when invested
    daily = position * returns["SPY"]

    # Transaction costs on entry/exit
    turnover = position.diff().abs().fillna(0)
    tc = turnover * TC_PER_TURN

    return daily - tc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plotting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_comparison(results: dict, output_path: str):
    """3-panel chart: cumulative returns, drawdowns, metrics table."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1.5]},
                             facecolor="#0f0f14")
    fig.subplots_adjust(hspace=0.30)

    palette = {
        "Multi-Agent":  "#00e676",
        "Buy & Hold SPY": "#5c6bc0",
        "20d Momentum": "#ff9100",
    }
    bg    = "#0f0f14"
    grid  = "#1a1a24"
    text  = "#c0c0c8"

    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=text, labelsize=9)
        ax.grid(True, color=grid, linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(grid)

    # ── Panel 1: Cumulative returns ────────────────────────────
    ax1 = axes[0]
    for name, data in results.items():
        cum = (1 + data["daily"]).cumprod()
        c = palette[name]
        ax1.plot(cum, color=c, linewidth=1.8, label=name, zorder=3)
        ax1.fill_between(cum.index, 1, cum.values, color=c, alpha=0.04)

    ax1.axhline(1.0, color=grid, linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_title("Strategy Comparison — Cumulative Returns",
                  fontsize=16, fontweight="bold", color="white", pad=12)
    ax1.set_ylabel("Growth of $1", fontsize=11, color=text)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.4,
               facecolor=bg, edgecolor=grid, labelcolor=text)

    # ── Panel 2: Drawdowns (overlaid) ──────────────────────────
    ax2 = axes[1]
    for name, data in results.items():
        cum = (1 + data["daily"]).cumprod()
        dd = (cum / cum.cummax()) - 1
        c = palette[name]
        ax2.fill_between(dd.index, 0, dd.values, color=c, alpha=0.25, linewidth=0)
        ax2.plot(dd, color=c, linewidth=0.8, alpha=0.8, label=name)

    ax2.set_ylabel("Drawdown", fontsize=11, color=text)
    ax2.set_xlabel("Date", fontsize=11, color=text)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.4,
               facecolor=bg, edgecolor=grid, labelcolor=text)

    # ── Metrics annotation on Panel 1 ─────────────────────────
    lines = []
    for name, data in results.items():
        m = data["metrics"]
        lines.append(
            f"{name:<17}  Return: {m['total_return']:+6.1%}  "
            f"Sharpe: {m['sharpe']:+5.2f}  "
            f"MaxDD: {m['max_drawdown']:6.1%}  "
            f"Calmar: {m['calmar']:5.2f}"
        )
    stats_text = "\n".join(lines)
    ax1.text(0.02, 0.04, stats_text, transform=ax1.transAxes,
             fontsize=9, color=text, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a24",
                       edgecolor=grid, alpha=0.92))

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=bg, edgecolor="none")
    plt.close()
    print(f"📊 Chart saved → {output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_comparison(results: dict):
    """Print a formatted side-by-side comparison table."""

    header = f"{'Metric':<22}"
    for name in results:
        header += f"  {name:>17}"
    sep = "─" * len(header)

    print()
    print("=" * len(header))
    print("  STRATEGY COMPARISON")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    rows = [
        ("Total Return",    "total_return",  "{:+.2%}"),
        ("Annual Return",   "annual_return", "{:+.2%}"),
        ("Annual Vol",      "annual_vol",    "{:.2%}"),
        ("Sharpe Ratio",    "sharpe",        "{:+.3f}"),
        ("Max Drawdown",    "max_drawdown",  "{:.2%}"),
        ("Calmar Ratio",    "calmar",        "{:.3f}"),
        ("Win Rate",        "win_rate",      "{:.1%}"),
    ]

    for label, key, fmt in rows:
        row = f"  {label:<20}"
        for name in results:
            val = results[name]["metrics"][key]
            row += f"  {fmt.format(val):>17}"
        print(row)

    print(sep)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":

    # ── Load data ──────────────────────────────────────────────
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)[TICKERS]
    returns = prices.pct_change().dropna()

    signals = pd.read_csv("signals.csv", index_col=0, parse_dates=True)
    signals.columns = [c.replace("_signal", "") for c in signals.columns]

    # Align all to common dates
    common = returns.index.intersection(signals.index)
    returns_aligned = returns.loc[common]
    signals_aligned = signals.loc[common]

    print(f"Evaluation period: {common[0].date()} → {common[-1].date()}")
    print(f"Trading days: {len(common)}\n")

    # ── Run strategies ─────────────────────────────────────────
    strats = {}

    # 1. Multi-Agent
    ma_daily = multi_agent_strategy(returns_aligned, signals_aligned)
    strats["Multi-Agent"] = {"daily": ma_daily, "metrics": metrics(ma_daily)}

    # 2. Buy & Hold SPY
    bh_daily = buy_and_hold_spy(returns_aligned)
    strats["Buy & Hold SPY"] = {"daily": bh_daily, "metrics": metrics(bh_daily)}

    # 3. 20-Day Momentum
    mom_daily = momentum_spy(returns_aligned, prices)
    strats["20d Momentum"] = {"daily": mom_daily, "metrics": metrics(mom_daily)}

    # ── Output ─────────────────────────────────────────────────
    print_comparison(strats)
    plot_comparison(strats, OUTPUT_PLOT)

    # Save comparison CSV
    comparison_df = pd.DataFrame({
        name: data["daily"] for name, data in strats.items()
    })
    comparison_df.to_csv("strategy_comparison.csv")
    print(f"💾 Daily returns saved → strategy_comparison.csv")
