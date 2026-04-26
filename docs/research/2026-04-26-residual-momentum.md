# Alpha Candidate: Residual Momentum (RMOM)

**Date:** 2026-04-26  
**Author:** QuantResearcher  
**Hypothesis:**  
Traditional price momentum is prone to "momentum crashes" during geopolitical shocks (like Mar 2026). Residual momentum—returns after stripping out market (KOSPI) and sector beta—identifies quality stocks with idiosyncratic strength that is more persistent than raw price trend.

**Formula:**  
Residuals from a 12-month rolling regression:  
$R_i - r_f = \alpha + \beta_{mkt}(R_{mkt} - r_f) + \epsilon$

**Data Source:**  
- pykrx daily price data.
- KOSPI and Sector indices for beta normalization.

**Expected IC Range:**  
0.05 – 0.09  
*Rationale:* RMOM historically has lower volatility and higher Sharpe ratios than standard momentum in the Korean market, especially during regime shifts.

** 논문/사례 링크:**  
- *Blitz, Huij, & Martens (2011), "Residual Momentum."*
- *Chaves (2012), "Is beta-neutral momentum better than raw momentum?"*
