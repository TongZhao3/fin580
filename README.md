# Narrative-Driven Macro Trading System

## Overview
This project builds a multi-agent trading system that converts macroeconomic news into investment decisions.

Pipeline:
**News → Signals → Macro Interpretation → Asset Allocation → Portfolio**

---

## System Design
- Narrative Agent: extracts inflation, growth, and risk signals
- Macro Agent: interprets market environment
- Asset Mapping: generates positions for SPY, TLT, GLD, UUP
- Portfolio: equal-weight allocation (25% each)

---

## Backtesting
- Daily data with 1-day signal lag (no look-ahead bias)
- Transaction cost: 30 bps per turnover
- Benchmark: equal-weight buy-and-hold

---

## Results
- More stable performance with lower drawdowns
- Lower returns in strong trending markets
- Macro reasoning and risk control improve robustness

---

## Run
```bash
python backtester.py
