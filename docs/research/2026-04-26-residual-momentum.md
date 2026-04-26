# Alpha Candidate: Residual Momentum (RMOM) - v2

**Date:** 2026-04-26  
**Author:** QuantResearcher  

## 1. Hypothesis & Mechanism
Traditional price momentum is prone to "momentum crashes" during geopolitical shocks (like Mar 2026). Residual momentum (RMOM) strips out market and sector beta, leaving idiosyncratic strength.

**Complementarity to Quality:** Quality stocks (High ROE) often exhibit "low-beta" characteristics. Standard momentum might ignore these during a high-beta market rally. RMOM identifies Quality names that are outperforming *their peers* and the market relative to their risk profile, acting as a superior timing overlay for fundamental quality.

## 2. Universe Definition
*   **Primary Universe:** KOSPI + KOSDAQ.
*   **Liquidity/Size Cutoff:** 
    *   Market Cap >= 100bn KRW.
    *   ADTV >= 1bn KRW (20-day MA).
*   **Constraint:** Minimum 252 days of trading history (to ensure stable beta estimation).

## 3. Formula & Implementation
**Residuals from a 12-month rolling regression:**  
$R_i - r_f = \alpha + \beta_{mkt}(R_{mkt} - r_f) + \beta_{sector}(R_{sector} - r_f) + \epsilon$
*   *Signal:* Use the sum of residuals ($\sum \epsilon$) over 12 months, skipping the most recent 1 month to avoid short-term reversal.

## 4. Data Coverage & Feasibility
*   **Source:** pykrx daily price data for tickers and major sector indices (KOSPI 200 Health Care, IT, etc.).
*   **Backtest Start Date:** 2016-01 (Long history available).
*   **Parsing Difficulty:** Low. Computational cost is high due to rolling regressions across 2,500+ tickers.

## 5. Expected IC & Rationale
*   **Predicted IC:** 0.05 – 0.09.
*   **Basis:** *Blitz et al. (2011)* findings in EM markets. In Korea, beta-neutral momentum significantly outperforms raw momentum during high-volatility regimes (e.g., 2020 COVID, 2026 Geopolitical shock).
*   **IC Correction:** Expected to be higher during regime transitions as RMOM pivots faster than raw momentum.

## 6. Turnover & Capacity
*   **Estimated Turnover:** 40-60% per month (Higher than ASY/VDS; typical for momentum).
*   **Capacity:** Estimated at 800bn KRW for a 20-stock portfolio (Slippage sensitive).

## 7. Sources
- *Blitz, Huij, & Martens (2011), "Residual Momentum."*
- *Chaves (2012), "Is beta-neutral momentum better than raw momentum?"*
- KRX Daily Index price history.
