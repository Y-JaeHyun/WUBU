# Alpha Candidate: Value-Up Disclosure Score (VDS) - v2

**Date:** 2026-04-26  
**Author:** QuantResearcher  

## 1. Hypothesis & Mechanism
Companies that voluntarily disclose "Value-Up" plans signal management alignment with shareholder interests. The specificity of targets (ROE, PBR, Dividends) acts as a proxy for management quality and reduces information asymmetry.

**Complementarity to Quality:** High Quality + Low PBR names (Value Traps) often lack a catalyst. VDS acts as the "activist catalyst" from within. It filters for Quality names that are not just profitable but are actively communicating a path to valuation re-rating.

## 2. Universe Definition
*   **Primary Universe:** KOSPI + KOSDAQ.
*   **Liquidity/Size Cutoff:** 
    *   Market Cap >= 300bn KRW (Focus on companies with higher institutional visibility).
    *   ADTV >= 3bn KRW.
*   **Constraint:** Companies currently trading at PBR < 1.5 (where the "Value-Up" re-rating potential is highest).

## 3. Formula & Implementation
**VDS = $\sum$ (Score based on KRX Portal Disclosure)**
*   Binary: Has Disclosure? (0/1)
*   Quantitative: Specific ROE/PBR target mentioned? (0/1)
*   Shareholder: Specific Dividend/Cancellation target mentioned? (0/1)
*   Timeline: 3-year plan provided? (0/1)
*   *Signal:* Score of 0 to 4.

## 4. Data Coverage & Feasibility
*   **Source:** KRX Value-Up Disclosure Portal (KIND integrated).
*   **Backtest Start Date:** 2024-05-27 (Official launch of program).
*   **Coverage:** 590 companies as of April 2026 (Rapidly expanding).
*   **Parsing Difficulty:** High. Requires LLM-based parsing or manual tagging of qualitative targets in KIND disclosure text.

## 5. Expected IC & Rationale
*   **Predicted IC:** 0.07 – 0.11 (Adjusted from v1's 0.15).
*   **Basis:** Japan's JPX-Nikkei 400 "Shame Index" experience. While initial IC was high (0.12+), it stabilized lower as participation became more common.
*   **Correction:** v1 was optimistic. VDS is likely a "binary catalyst" factor with high IC during disclosure months but lower persistence compared to RMOM.

## 6. Turnover & Capacity
*   **Estimated Turnover:** 10-15% (Disclosures are typically annual or semi-annual).
*   **Capacity:** 2tn KRW+ (Concentrated in KOSPI 200 / Mid-Cap leaders).

## 7. Sources
- KRX Corporate Value-Up Index criteria (2024/2025).
- *Abenomics case study (2014-2016): Impact of JPX-Nikkei 400 on capital efficiency.*
- BlackRock Stewardship Report on Asian Governance (2025).
