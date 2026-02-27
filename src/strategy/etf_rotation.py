"""ETF 로테이션 전략 모듈.

ETF 기반 섹터/테마/글로벌 로테이션 전략이다.
모멘텀 기반으로 상위 ETF를 선별하고, 절대모멘텀이 음수이면 안전자산으로 전환한다.

데이터 소스:
- pykrx ETF 가격 데이터 (etf_collector.get_etf_price)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 기본 ETF 유니버스
DEFAULT_ETF_UNIVERSE: dict[str, str] = {
    "069500": "KODEX 200",
    "371460": "TIGER 미국S&P500",
    "133690": "TIGER 미국나스닥100",
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
    "117700": "KODEX 건설",
    "132030": "KODEX 골드선물(H)",
    "464310": "TIGER 글로벌AI&로보틱스INDXX",
    "469150": "ACE AI반도체포커스",
    "439870": "KODEX 단기채권",
}

# ETF 섹터 카테고리 매핑 (섹터 집중도 제한용)
ETF_SECTOR_MAP: dict[str, str] = {
    "069500": "국내지수",
    "371460": "미국지수",
    "133690": "미국지수",
    "091160": "반도체",
    "091170": "금융",
    "117700": "건설",
    "132030": "원자재",
    "464310": "AI/로보틱스",
    "469150": "반도체",
    "439870": "채권",
}

# 기본 안전자산 (단기채권 ETF)
DEFAULT_SAFE_ASSET = "439870"


class ETFRotationStrategy(Strategy):
    """ETF 기반 섹터/테마 로테이션 전략.

    모멘텀 기반으로 상위 ETF를 선별하고,
    절대모멘텀이 음수이면 안전자산(단기채권)으로 전환한다.

    Args:
        lookback: 모멘텀 계산 기간 (기본 252거래일 = 12개월)
        num_etfs: 선정할 ETF 수 (기본 3)
        safe_asset: 안전자산 ETF 종목코드 (기본 '439870' = KODEX 단기채권)
        etf_universe: ETF 유니버스 딕셔너리 {종목코드: 이름}
        weighting: 가중 방식 ('equal' 또는 'inverse_vol')
        abs_momentum: 절대모멘텀 적용 여부 (기본 True)
        max_same_sector: 같은 섹터 ETF 최대 수 (기본 1, 0이면 제한 없음)
        momentum_cap: 모멘텀 절대 상한 (기본 3.0 = 300%).
            극단적 모멘텀(예: +400%)이 z-score를 왜곡하는 것을 방지.
            0 이하이면 캡 비활성화.
    """

    def __init__(
        self,
        lookback: int = 252,
        num_etfs: int = 3,
        safe_asset: str = DEFAULT_SAFE_ASSET,
        etf_universe: Optional[dict[str, str]] = None,
        weighting: str = "equal",
        abs_momentum: bool = True,
        max_same_sector: int = 1,
        momentum_cap: float = 3.0,
    ):
        self.lookback = lookback
        self.num_etfs = num_etfs
        self.safe_asset = safe_asset
        self.etf_universe = etf_universe or DEFAULT_ETF_UNIVERSE.copy()
        self.weighting = weighting
        self.abs_momentum = abs_momentum
        self.max_same_sector = max_same_sector
        self.momentum_cap = momentum_cap

        if weighting not in ("equal", "inverse_vol"):
            raise ValueError(
                f"지원하지 않는 가중 방식: {weighting}. "
                "'equal' 또는 'inverse_vol' 중 선택하세요."
            )

        # 단계적 fallback lookback 목록
        self._fallback_lookbacks = self._build_fallback_lookbacks(lookback)

        # 최근 실행 진단 정보
        self.last_diagnostics: dict = {}

        logger.info(
            f"ETFRotationStrategy 초기화: lookback={lookback}, "
            f"num_etfs={num_etfs}, safe_asset={safe_asset}, "
            f"weighting={weighting}, max_same_sector={max_same_sector}, "
            f"momentum_cap={momentum_cap}, "
            f"universe={len(self.etf_universe)}개"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"ETFRotation(top{self.num_etfs}, {self.lookback}d)"

    @staticmethod
    def _build_fallback_lookbacks(primary: int) -> list[int]:
        """단계적 fallback lookback 목록을 생성한다.

        기본 lookback부터 시작해서 126, 63까지 내림차순으로 구성.
        중복 제거 + 최소 63일 보장.
        """
        candidates = sorted(
            {primary, 126, 63}, reverse=True,
        )
        return [c for c in candidates if c <= primary]

    def _calculate_momentum(
        self,
        etf_prices: dict[str, pd.DataFrame],
        lookback_override: Optional[int] = None,
    ) -> pd.Series:
        """ETF별 모멘텀을 계산한다.

        모멘텀 = (현재가 / lookback일 전 가격) - 1

        Args:
            etf_prices: {ticker: OHLCV DataFrame} 딕셔너리
            lookback_override: None이면 self.lookback 사용.

        Returns:
            pd.Series (index=ticker, values=momentum)
        """
        lb = lookback_override if lookback_override is not None else self.lookback
        momentums = {}
        diagnostics: dict[str, dict] = {}

        for ticker in self.etf_universe:
            if ticker == self.safe_asset:
                continue

            if ticker not in etf_prices:
                diagnostics[ticker] = {
                    "status": "DATA_MISSING",
                    "available_days": 0,
                    "required_days": lb,
                }
                continue

            price_df = etf_prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                diagnostics[ticker] = {
                    "status": "DATA_MISSING",
                    "available_days": 0,
                    "required_days": lb,
                }
                continue

            closes = price_df["close"]
            available = len(closes)

            if available < lb:
                diagnostics[ticker] = {
                    "status": "DATA_SHORT",
                    "available_days": available,
                    "required_days": lb,
                }
                continue

            price_current = closes.iloc[-1]
            price_past = closes.iloc[-lb]

            if price_past <= 0 or np.isnan(price_past) or np.isnan(price_current):
                diagnostics[ticker] = {
                    "status": "DATA_INVALID",
                    "available_days": available,
                    "required_days": lb,
                }
                continue

            mom = float(price_current / price_past - 1)

            # 모멘텀 캡 적용: 극단적 상승 모멘텀 제한
            capped = False
            if self.momentum_cap > 0 and mom > self.momentum_cap:
                logger.info(
                    f"모멘텀 캡 적용: {ticker} {mom:.1%} → {self.momentum_cap:.1%}"
                )
                mom = self.momentum_cap
                capped = True

            momentums[ticker] = mom
            diagnostics[ticker] = {
                "status": "OK",
                "available_days": available,
                "required_days": lb,
                "momentum": mom,
                "capped": capped,
            }

        # per_ticker 진단 저장 (generate_signals에서 합산)
        self._last_per_ticker = diagnostics

        if not momentums:
            return pd.Series(dtype=float)

        return pd.Series(momentums, name="momentum")

    def _apply_sector_filter(
        self, momentum_ranked: pd.Series, num_etfs: int,
    ) -> pd.Series:
        """섹터 집중도를 제한하며 상위 ETF를 선정한다.

        모멘텀 순서대로 순회하면서 같은 섹터가 max_same_sector를
        초과하면 스킵하고 다음 순위 ETF로 대체한다.

        Args:
            momentum_ranked: 모멘텀 내림차순 정렬된 Series.
            num_etfs: 선정할 ETF 수.

        Returns:
            선정된 ETF의 모멘텀 Series.
        """
        if self.max_same_sector <= 0:
            return momentum_ranked.head(num_etfs)

        selected = []
        sector_counts: dict[str, int] = {}
        skipped: list[dict] = []

        for ticker in momentum_ranked.index:
            if len(selected) >= num_etfs:
                break

            sector = ETF_SECTOR_MAP.get(ticker, "기타")
            current_count = sector_counts.get(sector, 0)

            if current_count >= self.max_same_sector:
                skipped.append({
                    "ticker": ticker,
                    "sector": sector,
                    "momentum": float(momentum_ranked[ticker]),
                    "reason": f"섹터 '{sector}' 초과 ({current_count}/{self.max_same_sector})",
                })
                continue

            selected.append(ticker)
            sector_counts[sector] = current_count + 1

        # 스킵 정보를 진단에 기록
        if skipped:
            self._sector_skipped = skipped
            logger.info(
                f"섹터 필터: {len(skipped)}개 ETF 스킵 "
                f"({', '.join(s['ticker'] for s in skipped)})"
            )
        else:
            self._sector_skipped = []

        return momentum_ranked[selected]

    def _calculate_inverse_vol_weights(
        self, etf_prices: dict[str, pd.DataFrame], tickers: list[str]
    ) -> dict[str, float]:
        """역변동성 가중 비중을 계산한다.

        Args:
            etf_prices: {ticker: OHLCV DataFrame} 딕셔너리
            tickers: 대상 종목코드 리스트

        Returns:
            {ticker: weight} 딕셔너리
        """
        volatilities = {}

        for ticker in tickers:
            if ticker not in etf_prices:
                continue

            price_df = etf_prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                continue

            closes = price_df["close"].tail(self.lookback)
            returns = closes.pct_change().dropna()

            if len(returns) < 10:
                continue

            vol = float(returns.std())
            if vol > 0:
                volatilities[ticker] = vol

        if not volatilities:
            # fallback to equal weight
            weight = 1.0 / len(tickers) if tickers else 0.0
            return {t: weight for t in tickers}

        # 역변동성 = 1 / vol, 정규화
        inv_vols = {t: 1.0 / v for t, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol <= 0:
            weight = 1.0 / len(tickers)
            return {t: weight for t in tickers}

        return {t: iv / total_inv_vol for t, iv in inv_vols.items()}

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        1. 단계적 fallback으로 모멘텀 계산 (252 → 126 → 63)
        2. 상위 num_etfs개 선정
        3. 절대모멘텀 음수 ETF는 안전자산으로 대체
        4. 가중 방식에 따라 비중 할당
        5. self.last_diagnostics에 진단 정보 저장

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'etf_prices': dict[ticker, DataFrame],
                또는 'prices': dict[ticker, DataFrame] (호환용),
            }

        Returns:
            종목코드: 비중 딕셔너리
        """
        etf_prices = data.get("etf_prices", data.get("prices", {}))
        self._last_per_ticker = {}

        if not etf_prices:
            logger.warning(f"ETF 가격 데이터 없음 ({date}), 빈 시그널 반환")
            self.last_diagnostics = {
                "status": "DATA_UNAVAILABLE",
                "lookback_used": self.lookback,
                "reason": "ETF 가격 데이터 수집 실패",
                "per_ticker": {},
            }
            return {}

        # 단계적 fallback 모멘텀 계산
        momentum = pd.Series(dtype=float)
        lookback_used = self.lookback
        min_eligible = self.num_etfs + 1

        for lb in self._fallback_lookbacks:
            momentum = self._calculate_momentum(etf_prices, lookback_override=lb)
            if len(momentum) >= min_eligible:
                lookback_used = lb
                break
            lookback_used = lb

        if momentum.empty:
            logger.warning(f"모멘텀 계산 실패 ({date}), 모든 fallback 소진")
            self.last_diagnostics = {
                "status": "DATA_INSUFFICIENT",
                "lookback_used": lookback_used,
                "reason": f"최소 lookback {lookback_used}일에서도 "
                          f"모멘텀 계산 가능 ETF 없음",
                "per_ticker": dict(self._last_per_ticker),
            }
            return {}

        status = "OK" if lookback_used == self.lookback else "DEGRADED"
        if status == "DEGRADED":
            logger.info(
                f"ETF 모멘텀 fallback: {self.lookback}→{lookback_used}일 "
                f"({len(momentum)}개 ETF)"
            )

        # 모멘텀 상위 ETF 선정 (섹터 집중도 제한 적용)
        self._sector_skipped = []
        momentum_ranked = momentum.sort_values(ascending=False)
        top_momentum = self._apply_sector_filter(momentum_ranked, self.num_etfs)

        # 절대모멘텀 적용: 음수 모멘텀 ETF는 안전자산으로 대체
        selected_tickers = []
        safe_asset_count = 0

        for ticker in top_momentum.index:
            if self.abs_momentum and top_momentum[ticker] <= 0:
                safe_asset_count += 1
            else:
                selected_tickers.append(ticker)

        # 비중 할당
        total_slots = len(selected_tickers) + safe_asset_count

        if total_slots == 0:
            self.last_diagnostics = {
                "status": "DATA_INSUFFICIENT",
                "lookback_used": lookback_used,
                "reason": "선정 가능 ETF 없음",
                "per_ticker": dict(self._last_per_ticker),
            }
            return {}

        if self.weighting == "equal":
            slot_weight = 1.0 / total_slots
            signals = {t: slot_weight for t in selected_tickers}
            if safe_asset_count > 0:
                signals[self.safe_asset] = slot_weight * safe_asset_count
        elif self.weighting == "inverse_vol":
            if selected_tickers:
                risk_weights = self._calculate_inverse_vol_weights(
                    etf_prices, selected_tickers
                )
                risk_portion = len(selected_tickers) / total_slots
                signals = {t: w * risk_portion for t, w in risk_weights.items()}
                if safe_asset_count > 0:
                    safe_portion = safe_asset_count / total_slots
                    signals[self.safe_asset] = safe_portion
            else:
                signals = {self.safe_asset: 1.0}
        else:
            signals = {}

        # 진단 정보 저장
        self.last_diagnostics = {
            "status": status,
            "lookback_used": lookback_used,
            "lookback_requested": self.lookback,
            "eligible_count": len(momentum),
            "per_ticker": dict(self._last_per_ticker),
            "sector_skipped": list(self._sector_skipped),
        }

        logger.info(
            f"ETF 로테이션 시그널 생성 ({date}): {len(signals)}개 ETF, "
            f"안전자산 배분={safe_asset_count}/{total_slots}, "
            f"lookback={lookback_used}일"
        )

        return signals
