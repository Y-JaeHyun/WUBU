# Alpha Candidate: Advanced Shareholder Yield (ASY) - v2

**Date:** 2026-04-26  
**Author:** QuantResearcher  

## 1. Hypothesis & Mechanism
In the 2026 "Value-Up" regime, the market distinguishes between simple buybacks and **share cancellations**. While buybacks can be perfunctory (often re-issued for ESOPs or M&A), cancellations permanently reduce the share count and eliminate the "Treasury Share Magic" used for control enhancement.

**Complementarity to Quality:** Quality (ROE/GPA) focuses on capital *generation*. ASY focuses on capital *allocation*. High Quality + High ASY identifies "Compounders" that generate high returns and have the discipline to return excess cash rather than hoarding it (lowering ROE).

## 2. Universe Definition
*   **Primary Universe:** KOSPI + KOSDAQ (excluding Financials and Utilities for more stable GPA/ROE comparisons).
*   **Liquidity/Size Cutoff:** 
    *   Market Cap >= 200bn KRW (to ensure capacity for institutional inflows).
    *   Average Daily Traded Value (ADTV) >= 2bn KRW (20-day MA).
*   **Constraint:** Companies must have a positive Net Income (to avoid yield-traps from companies liquidating assets while in deficit).

## 3. Formula & Implementation
**ASY = (Cash Dividends + Treasury Share Cancellations) / Market Capitalization**
*   *Note:* Buybacks are only included if accompanied by a formal cancellation disclosure within the same fiscal year.

## 4. Data Coverage & Feasibility
*   **Source:** DART (Electronic Disclosure) reports specifically for "Treasury Share Retirement" (주식소각) and "Dividend Announcement."
*   **Backtest Start Date:** 2024-05 (Effective start of Value-Up voluntary disclosures).
*   **Parsing Difficulty:** Medium. DART API provides standardized tables for treasury share changes from late 2024. Historical data (pre-2024) requires semi-structured text parsing of "Report on Major Decisions."

## 5. Expected IC & Rationale
*   **Predicted IC:** 0.08 – 0.12 (Universe: KOSPI Top 500).
*   **Basis:** Japan's TSE reform (2023-2025) where share cancellation announcements delivered an average abnormal return of 4.2% in the T+20 window.
*   **IC Correction:** Expect higher IC in 2026 due to the "Mandatory Cancellation Rule" (Commercial Act Amendment), making disclosures higher-conviction.

## 6. Turnover & Capacity
*   **Estimated Turnover:** 15-25% per month (low frequency; aligned with dividend/disclosure cycles).
*   **Capacity:** Estimated at 1.5tn KRW for a 20-stock portfolio before slippage exceeds 30bps (assuming 15% of ADTV limit).

## 7. Sources
- *Fama & French (2001)* on Dividends/Buybacks.
- FSC (Korea) "Treasury Share Disclosure Reform" (2024/2025).
- Tokyo Stock Exchange (TSE) Governance Reform Case Study (2023-2024).
