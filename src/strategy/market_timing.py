"""마켓 타이밍 오버레이 모듈.

시장 지수의 이동평균선을 활용한 마켓 타이밍 전략을 제공한다.
지수가 이동평균선 위에 있으면 RISK_ON, 아래면 RISK_OFF 신호를 생성한다.
독립 전략이 아닌 오버레이로서 다른 전략의 비중을 조절하는 데 사용한다.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketTimingOverlay:
    """마켓 타이밍 오버레이.

    시장 지수의 이동평균선을 기준으로 위험 선호/회피 신호를 생성하고,
    이를 기반으로 전략의 포지션 비중을 조절한다.

    Args:
        ma_period: 이동평균 기간 (기본 200일)
        ma_type: 이동평균 유형 - "SMA" 또는 "EMA" (기본 "SMA")
        switch_mode: 전환 모드 - "binary" 또는 "gradual" (기본 "binary")
        whipsaw_filter: 휩소 필터 - "monthly_check" 또는 None (기본 "monthly_check")
        band_pct: 이동평균 밴드 폭 (기본 0.02 = 2%)
        cash_return_annual: 현금 보유 시 연간 수익률 (기본 2.5%)
        reference_index: 참조 지수명 (기본 "KOSPI")
    """

    def __init__(
        self,
        ma_period: int = 200,
        ma_type: str = "SMA",
        switch_mode: str = "binary",
        whipsaw_filter: Optional[str] = "monthly_check",
        band_pct: float = 0.02,
        cash_return_annual: float = 0.025,
        reference_index: str = "KOSPI",
    ):
        self.ma_period = ma_period
        self.ma_type = ma_type.upper()
        self.switch_mode = switch_mode.lower()
        self.whipsaw_filter = whipsaw_filter
        self.band_pct = band_pct
        self.cash_return_annual = cash_return_annual
        self.reference_index = reference_index

        # 이전 신호 저장 (휩소 필터용)
        self._last_signal: Optional[str] = None
        self._last_signal_date: Optional[pd.Timestamp] = None

        if self.ma_type not in ("SMA", "EMA"):
            raise ValueError(
                f"지원하지 않는 이동평균 유형: {ma_type}. 'SMA' 또는 'EMA' 중 선택하세요."
            )

        if self.switch_mode not in ("binary", "gradual"):
            raise ValueError(
                f"지원하지 않는 전환 모드: {switch_mode}. 'binary' 또는 'gradual' 중 선택하세요."
            )

        logger.info(
            f"MarketTimingOverlay 초기화: ma_period={ma_period}, "
            f"ma_type={self.ma_type}, switch_mode={self.switch_mode}, "
            f"band_pct={band_pct}, reference_index={reference_index}"
        )

    def _calculate_ma(self, prices: pd.Series) -> pd.Series:
        """이동평균을 계산한다.

        Args:
            prices: 종가 시계열

        Returns:
            이동평균 시계열
        """
        if self.ma_type == "SMA":
            return prices.rolling(window=self.ma_period, min_periods=self.ma_period).mean()
        elif self.ma_type == "EMA":
            return prices.ewm(span=self.ma_period, min_periods=self.ma_period, adjust=False).mean()
        else:
            raise ValueError(f"지원하지 않는 이동평균 유형: {self.ma_type}")

    def _apply_whipsaw_filter(
        self,
        raw_signal: str,
        current_date: pd.Timestamp,
    ) -> str:
        """휩소 필터를 적용한다.

        월말에만 신호 전환을 허용하여 잦은 매매를 방지한다.

        Args:
            raw_signal: 원시 신호 ("RISK_ON" 또는 "RISK_OFF")
            current_date: 현재 날짜

        Returns:
            필터링된 신호
        """
        if self.whipsaw_filter != "monthly_check":
            return raw_signal

        # 첫 신호인 경우 그대로 적용
        if self._last_signal is None:
            self._last_signal = raw_signal
            self._last_signal_date = current_date
            return raw_signal

        # 같은 신호면 유지
        if raw_signal == self._last_signal:
            return raw_signal

        # 신호가 바뀌었을 때, 월이 바뀌었는지 확인
        if self._last_signal_date is not None:
            if current_date.month != self._last_signal_date.month or \
               current_date.year != self._last_signal_date.year:
                # 새로운 달이므로 신호 전환 허용
                self._last_signal = raw_signal
                self._last_signal_date = current_date
                return raw_signal

        # 같은 달 내에서는 기존 신호 유지
        return self._last_signal

    def get_signal(self, index_prices: pd.Series) -> str:
        """지수 가격 기반 마켓 타이밍 신호를 생성한다.

        Args:
            index_prices: 지수 종가 시계열 (DatetimeIndex)

        Returns:
            "RISK_ON" 또는 "RISK_OFF"
        """
        if index_prices.empty or len(index_prices) < self.ma_period:
            logger.warning(
                f"지수 데이터 부족 (필요: {self.ma_period}일, "
                f"보유: {len(index_prices)}일). 기본 RISK_ON 반환."
            )
            return "RISK_ON"

        # 이동평균 계산
        ma = self._calculate_ma(index_prices)
        current_price = index_prices.iloc[-1]
        current_ma = ma.iloc[-1]
        current_date = index_prices.index[-1]

        if np.isnan(current_ma):
            logger.warning("이동평균 계산 불가. 기본 RISK_ON 반환.")
            return "RISK_ON"

        # 밴드를 적용한 비교
        upper_band = current_ma * (1 + self.band_pct)
        lower_band = current_ma * (1 - self.band_pct)

        if current_price > upper_band:
            raw_signal = "RISK_ON"
        elif current_price < lower_band:
            raw_signal = "RISK_OFF"
        else:
            # 밴드 내부: 기존 신호 유지 (중립 구간)
            raw_signal = self._last_signal if self._last_signal is not None else "RISK_ON"

        # 휩소 필터 적용
        if isinstance(current_date, pd.Timestamp):
            signal = self._apply_whipsaw_filter(raw_signal, current_date)
        else:
            signal = raw_signal

        logger.info(
            f"마켓 타이밍 신호: price={current_price:.2f}, "
            f"MA{self.ma_period}={current_ma:.2f}, "
            f"band=[{lower_band:.2f}, {upper_band:.2f}], "
            f"signal={signal}"
        )

        return signal

    def get_exposure_ratio(self, index_prices: pd.Series) -> float:
        """gradual 모드에서 지수와 이동평균의 이격도에 따른 노출 비율을 계산한다.

        이격도에 따라 25/50/75/100% 비중을 반환한다.

        Args:
            index_prices: 지수 종가 시계열

        Returns:
            노출 비율 (0.0 ~ 1.0)
        """
        if self.switch_mode != "gradual":
            # binary 모드에서는 0 또는 1
            signal = self.get_signal(index_prices)
            return 1.0 if signal == "RISK_ON" else 0.0

        if index_prices.empty or len(index_prices) < self.ma_period:
            return 1.0

        ma = self._calculate_ma(index_prices)
        current_price = index_prices.iloc[-1]
        current_ma = ma.iloc[-1]

        if np.isnan(current_ma) or current_ma <= 0:
            return 1.0

        # 이격도 계산: (현재가 - MA) / MA
        deviation = (current_price - current_ma) / current_ma

        # 이격도에 따른 비중 결정
        if deviation > self.band_pct * 2:
            # MA 위 크게 이탈: 100% 노출
            exposure = 1.0
        elif deviation > self.band_pct:
            # MA 위 소폭 이탈: 75% 노출
            exposure = 0.75
        elif deviation > -self.band_pct:
            # MA 근처 (밴드 내): 50% 노출
            exposure = 0.50
        elif deviation > -self.band_pct * 2:
            # MA 아래 소폭 이탈: 25% 노출
            exposure = 0.25
        else:
            # MA 아래 크게 이탈: 0% 노출
            exposure = 0.0

        logger.info(
            f"Gradual 노출 비율: deviation={deviation:.4f}, exposure={exposure:.0%}"
        )
        return exposure

    def apply_overlay(self, weights: dict, signal: str) -> dict:
        """마켓 타이밍 신호에 따라 포트폴리오 비중을 조절한다.

        binary 모드:
        - RISK_ON: 원본 비중 그대로
        - RISK_OFF: 빈 딕셔너리 (전량 현금)

        Args:
            weights: {ticker: weight} 포트폴리오 비중 딕셔너리
            signal: "RISK_ON" 또는 "RISK_OFF"

        Returns:
            조절된 {ticker: weight} 딕셔너리
        """
        if signal == "RISK_OFF":
            logger.info("RISK_OFF 신호: 전량 현금 전환")
            return {}
        else:
            return weights

    def apply_overlay_gradual(
        self,
        weights: dict,
        index_prices: pd.Series,
    ) -> dict:
        """gradual 모드에서 이격도 기반으로 포트폴리오 비중을 조절한다.

        Args:
            weights: {ticker: weight} 포트폴리오 비중 딕셔너리
            index_prices: 지수 종가 시계열

        Returns:
            조절된 {ticker: weight} 딕셔너리
        """
        exposure = self.get_exposure_ratio(index_prices)

        if exposure <= 0:
            logger.info("Gradual 모드: 전량 현금 전환")
            return {}

        if exposure >= 1.0:
            return weights

        adjusted = {ticker: weight * exposure for ticker, weight in weights.items()}
        logger.info(
            f"Gradual 모드: 비중 조절 (exposure={exposure:.0%}, "
            f"종목 수={len(adjusted)})"
        )
        return adjusted

    def reset(self) -> None:
        """내부 상태를 초기화한다. 새로운 백테스트 시작 시 호출."""
        self._last_signal = None
        self._last_signal_date = None
        logger.info("MarketTimingOverlay 상태 초기화")
