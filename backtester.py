"""
Backtester
==========
Backtests a multi-asset macro trading strategy using:
  - prices.csv   (daily adjusted close for SPY, TLT, GLD, UUP)
  - signals.csv  (daily {-1, 0, +1} signals from the agent pipeline)

Key design choices:
  - Signals are lagged by 1 day to avoid look-ahead bias
  - Equal weight (25%) per asset
  - Transaction cost: 30 bps per unit of turnover
  - Benchmark: buy-and-hold equal-weight portfolio
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — save to file
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TICKERS        = ["SPY", "TLT", "GLD", "UUP"]
WEIGHT         = 1.0 / len(TICKERS)          # 25% each
TC_PER_TURN    = 0.003                       # 30 bps per unit of turnover
ANNUAL_FACTOR  = 252                         # trading days per year
PRICES_FILE    = "prices.csv"
SIGNALS_FILE   = "signals.csv"
OUTPUT_PLOT    = "backtest_results.png"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Load & Align Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load price and signal data, align on common dates."""

    # Prices
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    prices = prices[TICKERS]

    # Signals
    signals = pd.read_csv(SIGNALS_FILE, index_col=0, parse_dates=True)
    signal_cols = [f"{t}_signal" for t in TICKERS]
    signals = signals[signal_cols]
    # Rename columns to match tickers for easy alignment
    signals.columns = TICKERS

    # Daily simple returns
    returns = prices.pct_change().dropna()

    # Align to common dates
    common = returns.index.intersection(signals.index)
    returns = returns.loc[common]
    signals = signals.loc[common]

    return returns, signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Backtesting Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest(returns: pd.DataFrame, signals: pd.DataFrame) -> dict:
    """
    Run the backtest.

    Position logic:
      position(t) = signal(t-1)       ← 1-day lag, no look-ahead
      asset_return(t) = position(t) * return(t) * weight

    Transaction costs:
      turnover(t) = |position(t) - position(t-1)|
      cost(t) = sum(turnover across assets) * weight * TC_PER_TURN
    """
    # ── Lag signals by 1 day ───────────────────────────────────
    positions = signals.shift(1).fillna(0).astype(int)

    # ── Compute weighted asset returns ─────────────────────────
    # Each asset contributes WEIGHT * position * return
    strategy_asset_returns = positions * returns * WEIGHT

    # ── Transaction costs ──────────────────────────────────────
    turnover = positions.diff().abs().fillna(0)
    daily_tc = (turnover * WEIGHT * TC_PER_TURN).sum(axis=1)

    # ── Portfolio daily returns ────────────────────────────────
    strategy_daily = strategy_asset_returns.sum(axis=1) - daily_tc

    # ── Benchmark: buy-and-hold equal weight ───────────────────
    benchmark_daily = (returns * WEIGHT).sum(axis=1)

    # ── Cumulative returns (growth of $1) ──────────────────────
    strategy_cum = (1 + strategy_daily).cumprod()
    benchmark_cum = (1 + benchmark_daily).cumprod()

    # ── Performance metrics ────────────────────────────────────
    metrics = compute_metrics(strategy_daily, "Strategy")
    bench_metrics = compute_metrics(benchmark_daily, "Benchmark (EW Buy&Hold)")

    return {
        "strategy_daily":   strategy_daily,
        "benchmark_daily":  benchmark_daily,
        "strategy_cum":     strategy_cum,
        "benchmark_cum":    benchmark_cum,
        "positions":        positions,
        "turnover":         turnover,
        "metrics":          metrics,
        "bench_metrics":    bench_metrics,
    }


def compute_metrics(daily_returns: pd.Series, name: str) -> dict:
    """Compute standard performance metrics."""
    total_return   = (1 + daily_returns).prod() - 1
    annual_return  = (1 + total_return) ** (ANNUAL_FACTOR / len(daily_returns)) - 1
    annual_vol     = daily_returns.std() * np.sqrt(ANNUAL_FACTOR)
    sharpe         = annual_return / annual_vol if annual_vol > 0 else 0.0

    # Max drawdown
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate
    win_rate = (daily_returns > 0).mean()

    # Calmar ratio
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "name":            name,
        "total_return":    total_return,
        "annual_return":   annual_return,
        "annual_vol":      annual_vol,
        "sharpe":          sharpe,
        "max_drawdown":    max_dd,
        "calmar":          calmar,
        "win_rate":        win_rate,
        "n_days":          len(daily_returns),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plotting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_results(result: dict, output_path: str):
    """Generate a publication-quality backtest chart."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             facecolor="#0f0f14")
    fig.subplots_adjust(hspace=0.35)

    colors = {
        "bg":        "#0f0f14",
        "grid":      "#1a1a24",
        "text":      "#c0c0c8",
        "strategy":  "#00e676",
        "benchmark": "#5c6bc0",
        "drawdown":  "#ff1744",
        "turnover":  "#ffc107",
    }

    for ax in axes:
        ax.set_facecolor(colors["bg"])
        ax.tick_params(colors=colors["text"], labelsize=9)
        ax.grid(True, color=colors["grid"], linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(colors["grid"])

    # ── Panel 1: Cumulative returns ────────────────────────────
    ax1 = axes[0]
    ax1.plot(result["strategy_cum"],  color=colors["strategy"],
             linewidth=1.8, label="Multi-Agent Strategy", zorder=3)
    ax1.plot(result["benchmark_cum"], color=colors["benchmark"],
             linewidth=1.2, label="Equal-Weight Buy & Hold",
             alpha=0.7, linestyle="--", zorder=2)
    ax1.fill_between(result["strategy_cum"].index, 1,
                     result["strategy_cum"].values,
                     color=colors["strategy"], alpha=0.06)

    ax1.set_title("Multi-Agent Macro Strategy — Backtest",
                  fontsize=16, fontweight="bold", color="white", pad=12)
    ax1.set_ylabel("Growth of $1", fontsize=11, color=colors["text"])
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.3,
               facecolor=colors["bg"], edgecolor=colors["grid"],
               labelcolor=colors["text"])
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))

    # Add metrics annotation
    m = result["metrics"]
    b = result["bench_metrics"]
    stats_text = (
        f"Strategy   — Sharpe: {m['sharpe']:.2f}  |  "
        f"Return: {m['total_return']:+.1%}  |  "
        f"MaxDD: {m['max_drawdown']:.1%}\n"
        f"Benchmark — Sharpe: {b['sharpe']:.2f}  |  "
        f"Return: {b['total_return']:+.1%}  |  "
        f"MaxDD: {b['max_drawdown']:.1%}"
    )
    ax1.text(0.02, 0.04, stats_text, transform=ax1.transAxes,
             fontsize=9, color=colors["text"], family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a24",
                       edgecolor=colors["grid"], alpha=0.9))

    # ── Panel 2: Drawdown ──────────────────────────────────────
    ax2 = axes[1]
    cum = result["strategy_cum"]
    dd = (cum - cum.cummax()) / cum.cummax()
    ax2.fill_between(dd.index, 0, dd.values, color=colors["drawdown"],
                     alpha=0.5, linewidth=0)
    ax2.plot(dd, color=colors["drawdown"], linewidth=0.8, alpha=0.8)
    ax2.set_ylabel("Drawdown", fontsize=11, color=colors["text"])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    # ── Panel 3: Daily turnover ────────────────────────────────
    ax3 = axes[2]
    daily_turnover = result["turnover"].sum(axis=1) * WEIGHT
    ax3.bar(daily_turnover.index, daily_turnover.values,
            color=colors["turnover"], alpha=0.6, width=2)
    ax3.set_ylabel("Turnover", fontsize=11, color=colors["text"])
    ax3.set_xlabel("Date", fontsize=11, color=colors["text"])

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=colors["bg"], edgecolor="none")
    plt.close()
    print(f"📊 Plot saved → {output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reporting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(result: dict):
    """Print a formatted performance report to stdout."""

    def fmt(m: dict):
        lines = [
            f"  {'Name:':<22} {m['name']}",
            f"  {'Total Return:':<22} {m['total_return']:+.2%}",
            f"  {'Annual Return:':<22} {m['annual_return']:+.2%}",
            f"  {'Annual Volatility:':<22} {m['annual_vol']:.2%}",
            f"  {'Sharpe Ratio:':<22} {m['sharpe']:.3f}",
            f"  {'Max Drawdown:':<22} {m['max_drawdown']:.2%}",
            f"  {'Calmar Ratio:':<22} {m['calmar']:.3f}",
            f"  {'Win Rate:':<22} {m['win_rate']:.1%}",
            f"  {'Trading Days:':<22} {m['n_days']}",
        ]
        return "\n".join(lines)

    total_turnover = result["turnover"].sum().sum() * WEIGHT
    total_tc = total_turnover * TC_PER_TURN
    n_trades = (result["turnover"].sum(axis=1) > 0).sum()

    print("=" * 64)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 64)
    print()
    print("── Strategy ──")
    print(fmt(result["metrics"]))
    print()
    print("── Benchmark (Equal-Weight Buy & Hold) ──")
    print(fmt(result["bench_metrics"]))
    print()
    print("── Trading Activity ──")
    print(f"  {'Signal change days:':<22} {n_trades}")
    print(f"  {'Total turnover:':<22} {total_turnover:.2f}")
    print(f"  {'Total txn cost:':<22} {total_tc:.4f} ({total_tc:.2%} of capital)")
    print(f"  {'Cost per trade day:':<22} {total_tc/max(n_trades,1):.5f}")
    print()

    # Per-asset contribution
    positions = result["positions"]
    print("── Per-Asset Signal Breakdown ──")
    for t in TICKERS:
        longs  = (positions[t] ==  1).sum()
        shorts = (positions[t] == -1).sum()
        flat   = (positions[t] ==  0).sum()
        total  = len(positions[t])
        print(f"  {t}:  long={longs:3d} ({longs/total:.0%})  "
              f"short={shorts:3d} ({shorts/total:.0%})  "
              f"flat={flat:3d} ({flat/total:.0%})")
    print("=" * 64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Load
    returns, signals = load_data()
    print(f"Loaded {len(returns)} common trading days\n")

    # Backtest
    result = backtest(returns, signals)

    # Report
    print_report(result)

    # Plot
    plot_results(result, OUTPUT_PLOT)

    # Save daily returns
    output = pd.DataFrame({
        "strategy_return":  result["strategy_daily"],
        "benchmark_return": result["benchmark_daily"],
        "strategy_cumulative":  result["strategy_cum"],
        "benchmark_cumulative": result["benchmark_cum"],
    })
    output.to_csv("backtest_daily.csv")
    print(f"💾 Daily returns saved → backtest_daily.csv")
