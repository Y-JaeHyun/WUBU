"""듀얼 모멘텀 ETF 자산배분 전략 모듈.

상대 모멘텀(Relative Momentum)과 절대 모멘텀(Absolute Momentum)을
결합한 듀얼 모멘텀 전략으로 ETF 기반 자산배분을 수행한다.

Strategy ABC는 상속하지 않는다 (ETF 전용 전략이므로 별도 인터페이스).

전략 흐름:
1. 상대 모멘텀: 위험자산 중 lookback_months 수익률이 가장 높은 자산 선택
2. 절대 모멘텀: 선택된 자산의 수익률이 양수인지 확인
3. 절대 모멘텀 미통과 시 안전자산(단기채권 등)으로 배분
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 기본 위험자산 구성
DEFAULT_RISKY_ASSETS: dict[str, str] = {
    "domestic": "069500",  # KODEX 200
    "us": "360750",        # TIGER 미국S&P500
}

# 기본 안전자산
DEFAULT_SAFE_ASSET: str = "214980"  # KODEX 단기채권PLUS


class DualMomentumStrategy:
    """듀얼 모멘텀 ETF 자산배분 전략.

    상대 모멘텀과 절대 모멘텀을 결합하여 위험자산과 안전자산 간
    동적 자산배분을 수행한다.

    Args:
        risky_assets: 위험자산 딕셔너리 {이름: 종목코드}
            기본값: {"domestic": "069500", "us": "360750"}
        safe_asset: 안전자산 종목코드 (기본 "214980")
        lookback_months: 모멘텀 산출 기간 (월, 기본 12)
        n_select: 상대 모멘텀에서 선택할 자산 수 (기본 1)
        volatility_target: 목표 변동성 (기본 0.0, 비활성).
            0보다 큰 값 설정 시 목표 변동성 대비 실현 변동성으로
            포지션 비중 조절 (target_vol / realized_vol).
            예: 0.15 → 연환산 15% 변동성 타깃.
        etf_universe: 추가 ETF 유니버스 딕셔너리 {이름: 종목코드}.
            설정 시 기존 risky_assets에 병합. 기본 None.
    """

    def __init__(
        self,
        risky_assets: Optional[dict[str, str]] = None,
        safe_asset: str = DEFAULT_SAFE_ASSET,
        lookback_months: int = 12,
        n_select: int = 1,
        volatility_target: float = 0.0,
        etf_universe: Optional[dict[str, str]] = None,
    ):
        self.risky_assets = risky_assets or DEFAULT_RISKY_ASSETS.copy()
        self.safe_asset = safe_asset
        self.lookback_months = lookback_months
        self.n_select = n_select
        self.volatility_target = volatility_target

        # ETF 유니버스 확대
        if etf_universe:
            self.risky_assets.update(etf_universe)

        if self.n_select > len(self.risky_assets):
            logger.warning(
                f"n_select({n_select})이 위험자산 수({len(self.risky_assets)})보다 큼. "
                f"위험자산 수로 조정합니다."
            )
            self.n_select = len(self.risky_assets)

        logger.info(
            f"DualMomentumStrategy 초기화: "
            f"risky_assets={list(self.risky_assets.keys())}, "
            f"safe_asset={safe_asset}, "
            f"lookback_months={lookback_months}, n_select={n_select}, "
            f"volatility_target={volatility_target}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        assets_str = "+".join(self.risky_assets.keys())
        return f"DualMomentum({assets_str}, {self.lookback_months}M, top{self.n_select})"

    def calculate_momentum(
        self,
        prices: dict[str, pd.Series],
        lookback_months: Optional[int] = None,
    ) -> dict[str, float]:
        """각 자산의 모멘텀(lookback 수익률)을 계산한다.

        Args:
            prices: {자산 식별자: 종가 시계열(DatetimeIndex)} 딕셔너리
            lookback_months: 룩백 기간 (월)

        Returns:
            {자산 식별자: 수익률} 딕셔너리.
            데이터 부족 자산은 NaN으로 반환.
        """
        if lookback_months is None:
            lookback_months = self.lookback_months
        lookback_days = lookback_months * 21  # 월 → 거래일 환산
        momentum: dict[str, float] = {}

        for asset_id, price_series in prices.items():
            if price_series is None or len(price_series) == 0:
                logger.warning(f"가격 데이터 없음: {asset_id}")
                momentum[asset_id] = float("nan")
                continue

            if len(price_series) < lookback_days:
                logger.warning(
                    f"데이터 부족: {asset_id} "
                    f"(필요: {lookback_days}일, 보유: {len(price_series)}일)"
                )
                momentum[asset_id] = float("nan")
                continue

            # 현재가 / lookback_days 전 가격 - 1
            current_price = price_series.iloc[-1]
            past_price = price_series.iloc[-lookback_days]

            if past_price <= 0 or np.isnan(past_price) or np.isnan(current_price):
                momentum[asset_id] = float("nan")
                continue

            ret = float(current_price / past_price - 1)
            momentum[asset_id] = ret

        logger.info(
            f"모멘텀 계산 완료 ({lookback_months}M): "
            + ", ".join(f"{k}={v:.2%}" if not np.isnan(v) else f"{k}=NaN"
                        for k, v in momentum.items())
        )

        return momentum

    def get_relative_signal(
        self,
        momentum: dict[str, float],
        n: Optional[int] = None,
    ) -> list[str]:
        """상대 모멘텀: 수익률 상위 n개 자산을 선택한다.

        NaN 수익률의 자산은 제외하고, 유효한 자산 중 상위를 선택한다.

        Args:
            momentum: {자산 식별자: 수익률} 딕셔너리
            n: 선택할 자산 수

        Returns:
            선택된 자산 식별자 리스트 (수익률 내림차순)
        """
        if n is None:
            n = self.n_select
        # NaN 제외
        valid = {k: v for k, v in momentum.items() if not np.isnan(v)}

        if not valid:
            logger.warning("유효한 모멘텀 자산 없음: 빈 리스트 반환")
            return []

        # 수익률 내림차순 정렬
        sorted_assets = sorted(valid.items(), key=lambda x: x[1], reverse=True)

        # 상위 n개 선택
        selected = [asset_id for asset_id, _ in sorted_assets[:n]]

        logger.info(
            f"상대 모멘텀 선택: {selected} "
            f"(전체: {[f'{k}:{v:.2%}' for k, v in sorted_assets]})"
        )

        return selected

    def get_absolute_signal(
        self,
        momentum: dict[str, float],
        threshold: float = 0.0,
    ) -> dict[str, bool]:
        """절대 모멘텀: 각 자산의 수익률이 threshold 초과인지 판단한다.

        Args:
            momentum: {자산 식별자: 수익률} 딕셔너리
            threshold: 절대 모멘텀 기준 수익률 (기본 0.0, 양수이면 통과)

        Returns:
            {자산 식별자: 통과 여부(bool)} 딕셔너리.
            NaN 수익률은 False로 처리.
        """
        signals: dict[str, bool] = {}

        for asset_id, ret in momentum.items():
            if np.isnan(ret):
                signals[asset_id] = False
            else:
                signals[asset_id] = ret > threshold

        logger.info(
            f"절대 모멘텀 신호 (threshold={threshold:.2%}): "
            + ", ".join(f"{k}={'통과' if v else '미통과'}" for k, v in signals.items())
        )

        return signals

    def generate_allocation(
        self,
        prices: dict[str, pd.Series],
    ) -> dict[str, float]:
        """최종 자산배분 비중을 생성한다.

        듀얼 모멘텀 로직:
        1. 위험자산의 모멘텀 계산
        2. 상대 모멘텀: 상위 n개 자산 선택
        3. 절대 모멘텀: 선택된 자산의 수익률이 양수인지 확인
        4. 절대 모멘텀 통과 → 해당 자산에 투자
        5. 절대 모멘텀 미통과 → 안전자산으로 배분

        Args:
            prices: {자산 식별자 또는 종목코드: 종가 시계열} 딕셔너리
                위험자산과 안전자산의 가격 데이터를 모두 포함해야 한다.
                자산 식별자는 risky_assets의 key 또는 종목코드를 사용한다.

        Returns:
            {종목코드: 비중} 딕셔너리. 비중 합 = 1.0
        """
        # 위험자산의 모멘텀 계산
        risky_prices: dict[str, pd.Series] = {}
        for asset_name, ticker in self.risky_assets.items():
            # 자산명 또는 종목코드로 가격 데이터 탐색
            if asset_name in prices:
                risky_prices[asset_name] = prices[asset_name]
            elif ticker in prices:
                risky_prices[asset_name] = prices[ticker]
            else:
                logger.warning(f"위험자산 가격 데이터 없음: {asset_name} ({ticker})")

        if not risky_prices:
            logger.warning("위험자산 가격 데이터 전부 없음: 안전자산 100% 배분")
            return {self.safe_asset: 1.0}

        # 1. 모멘텀 계산
        momentum = self.calculate_momentum(risky_prices, self.lookback_months)

        # 2. 상대 모멘텀: 상위 n개 선택
        selected = self.get_relative_signal(momentum, self.n_select)

        if not selected:
            logger.info("상대 모멘텀 선택 자산 없음: 안전자산 100% 배분")
            return {self.safe_asset: 1.0}

        # 3. 절대 모멘텀 필터
        absolute_signals = self.get_absolute_signal(momentum)

        # 4. 최종 배분
        allocation: dict[str, float] = {}
        weight_per_asset = 1.0 / self.n_select

        safe_weight = 0.0

        for asset_name in selected:
            ticker = self.risky_assets.get(asset_name, asset_name)
            if absolute_signals.get(asset_name, False):
                # 절대 모멘텀 통과: 위험자산에 투자
                allocation[ticker] = allocation.get(ticker, 0.0) + weight_per_asset
            else:
                # 절대 모멘텀 미통과: 안전자산으로 전환
                safe_weight += weight_per_asset
                logger.info(
                    f"절대 모멘텀 미통과: {asset_name} ({ticker}) → 안전자산 전환"
                )

        if safe_weight > 0:
            allocation[self.safe_asset] = allocation.get(self.safe_asset, 0.0) + safe_weight

        # 변동성 타깃 적용
        if self.volatility_target > 0:
            allocation = self._apply_volatility_target(allocation, prices)

        logger.info(
            f"듀얼 모멘텀 배분 결과: "
            + ", ".join(f"{k}={v:.0%}" for k, v in allocation.items())
        )

        return allocation

    def _apply_volatility_target(
        self,
        allocation: dict[str, float],
        prices: dict[str, pd.Series],
    ) -> dict[str, float]:
        """변동성 타깃에 따라 포지션 비중을 조절한다.

        target_vol / realized_vol 비율로 위험자산 비중을 스케일링하고,
        초과분은 안전자산으로 배분한다.

        Args:
            allocation: 현재 자산배분 딕셔너리
            prices: 가격 시계열 딕셔너리

        Returns:
            조정된 자산배분 딕셔너리
        """
        adjusted = {}
        excess_to_safe = 0.0

        for ticker, weight in allocation.items():
            if ticker == self.safe_asset:
                adjusted[ticker] = weight
                continue

            # 실현 변동성 계산 (최근 60일 기준, 연환산)
            price_series = None
            # ticker로 직접 찾기
            if ticker in prices:
                price_series = prices[ticker]
            else:
                # risky_assets에서 asset_name으로 찾기
                for asset_name, asset_ticker in self.risky_assets.items():
                    if asset_ticker == ticker and asset_name in prices:
                        price_series = prices[asset_name]
                        break

            if price_series is None or len(price_series) < 60:
                adjusted[ticker] = weight
                continue

            daily_returns = price_series.iloc[-60:].pct_change().dropna()
            if daily_returns.empty:
                adjusted[ticker] = weight
                continue

            realized_vol = float(daily_returns.std() * np.sqrt(252))

            if realized_vol <= 0:
                adjusted[ticker] = weight
                continue

            # 비중 조절: target_vol / realized_vol
            vol_scalar = min(self.volatility_target / realized_vol, 1.5)  # 최대 1.5배
            new_weight = weight * vol_scalar
            adjusted[ticker] = new_weight
            excess_to_safe += weight - new_weight

        # 초과분을 안전자산에 추가
        if excess_to_safe > 0:
            adjusted[self.safe_asset] = adjusted.get(self.safe_asset, 0.0) + excess_to_safe

        # 비중 합이 1.0이 되도록 정규화
        total = sum(adjusted.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def backtest(
        self,
        price_data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000_000,
        rebalance_freq: str = "monthly",
        buy_cost: float = 0.00015,
        sell_cost: float = 0.00015,
    ) -> pd.DataFrame:
        """듀얼 모멘텀 전략의 월별 리밸런싱 시뮬레이션을 수행한다.

        Args:
            price_data: {종목코드: DataFrame(OHLCV)} 딕셔너리.
                risky_assets와 safe_asset의 가격 데이터를 포함해야 한다.
            start_date: 시작일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
            end_date: 종료일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
            initial_capital: 초기 자본금 (기본 1억원)
            rebalance_freq: 리밸런싱 주기 - "monthly" (기본)
            buy_cost: 매수 거래비용 비율 (기본 0.015%)
            sell_cost: 매도 거래비용 비율 (기본 0.015%)

        Returns:
            DataFrame with columns:
                - portfolio_value: 포트폴리오 가치
                - allocation: 해당 시점의 자산배분 (dict as string)
            Index: DatetimeIndex (리밸런싱 날짜)
        """
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        logger.info(
            f"듀얼 모멘텀 백테스트 시작: {start} ~ {end}, "
            f"초기자본={initial_capital:,.0f}원"
        )

        # 영업일 생성
        all_dates = pd.date_range(start=start, end=end, freq="B")
        if all_dates.empty:
            logger.error("유효한 영업일이 없습니다.")
            return pd.DataFrame()

        # 리밸런싱 날짜 결정 (월초 첫 영업일)
        if rebalance_freq == "monthly":
            rebal_dates = all_dates.to_series().groupby(
                all_dates.to_period("M")
            ).first().values
        else:
            rebal_dates = all_dates.to_series().groupby(
                all_dates.to_period("M")
            ).first().values

        rebal_dates = pd.DatetimeIndex(rebal_dates)

        # 모든 종목의 close 가격 시계열 추출
        close_prices: dict[str, pd.Series] = {}
        all_tickers = set(self.risky_assets.values()) | {self.safe_asset}

        for ticker in all_tickers:
            if ticker in price_data and not price_data[ticker].empty:
                df = price_data[ticker]
                if "close" in df.columns:
                    close_prices[ticker] = df["close"]
                else:
                    logger.warning(f"close 컬럼 없음: {ticker}")
            else:
                logger.warning(f"가격 데이터 없음: {ticker}")

        if not close_prices:
            logger.error("유효한 가격 데이터가 없습니다.")
            return pd.DataFrame()

        # 시뮬레이션
        portfolio_value = float(initial_capital)
        holdings: dict[str, float] = {}  # ticker -> 보유 비중 (금액)
        history: list[dict] = []

        for i, rebal_date in enumerate(rebal_dates):
            rebal_str = rebal_date.strftime("%Y%m%d")

            # 현재까지의 가격 데이터로 모멘텀 계산용 시계열 준비
            current_prices: dict[str, pd.Series] = {}
            for asset_name, ticker in self.risky_assets.items():
                if ticker in close_prices:
                    # 현재 날짜까지의 가격만 사용
                    mask = close_prices[ticker].index <= rebal_date
                    available = close_prices[ticker][mask]
                    if not available.empty:
                        current_prices[asset_name] = available

            # 기존 포지션의 현재 가치 계산
            if holdings:
                portfolio_value = 0.0
                for ticker, amount in holdings.items():
                    if ticker in close_prices:
                        mask = close_prices[ticker].index <= rebal_date
                        if mask.any():
                            current_price = close_prices[ticker][mask].iloc[-1]
                            # 이전 리밸런싱 시점의 가격 대비 수익률 적용
                            prev_date = rebal_dates[i - 1] if i > 0 else rebal_dates[0]
                            prev_mask = close_prices[ticker].index <= prev_date
                            if prev_mask.any():
                                prev_price = close_prices[ticker][prev_mask].iloc[-1]
                                if prev_price > 0:
                                    ret = current_price / prev_price
                                    portfolio_value += amount * ret
                                else:
                                    portfolio_value += amount
                            else:
                                portfolio_value += amount
                        else:
                            portfolio_value += amount
                    else:
                        portfolio_value += amount

            # 듀얼 모멘텀 자산배분 결정
            allocation = self.generate_allocation(current_prices)

            # 거래비용 적용 (단순화: 전체 포트폴리오에 비용 적용)
            if holdings:
                # 매도 비용
                portfolio_value *= (1 - sell_cost)

            # 매수 비용
            portfolio_value *= (1 - buy_cost)

            # 새 포지션 설정
            holdings = {}
            for ticker, weight in allocation.items():
                holdings[ticker] = portfolio_value * weight

            history.append({
                "date": rebal_date,
                "portfolio_value": portfolio_value,
                "allocation": str(allocation),
            })

            logger.debug(
                f"리밸런싱 ({rebal_str}): "
                f"포트폴리오={portfolio_value:,.0f}원, "
                f"배분={allocation}"
            )

        # 최종 가치 계산 (마지막 리밸런싱 이후 ~ 종료일)
        if holdings and len(all_dates) > 0:
            final_date = all_dates[-1]
            last_rebal = rebal_dates[-1] if len(rebal_dates) > 0 else final_date

            if final_date > last_rebal:
                final_value = 0.0
                for ticker, amount in holdings.items():
                    if ticker in close_prices:
                        # 마지막 리밸런싱 → 종료일 수익률
                        rebal_mask = close_prices[ticker].index <= last_rebal
                        final_mask = close_prices[ticker].index <= final_date

                        if rebal_mask.any() and final_mask.any():
                            rebal_price = close_prices[ticker][rebal_mask].iloc[-1]
                            final_price = close_prices[ticker][final_mask].iloc[-1]
                            if rebal_price > 0:
                                ret = final_price / rebal_price
                                final_value += amount * ret
                            else:
                                final_value += amount
                        else:
                            final_value += amount
                    else:
                        final_value += amount

                history.append({
                    "date": final_date,
                    "portfolio_value": final_value,
                    "allocation": history[-1]["allocation"] if history else "",
                })

        if not history:
            logger.warning("백테스트 이력이 비어 있습니다.")
            return pd.DataFrame()

        result = pd.DataFrame(history)
        result["date"] = pd.to_datetime(result["date"])
        result = result.set_index("date")

        # 성과 요약
        if len(result) > 0:
            final_val = result["portfolio_value"].iloc[-1]
            total_return = (final_val / initial_capital - 1) * 100
            logger.info(
                f"듀얼 모멘텀 백테스트 완료: "
                f"최종가치={final_val:,.0f}원, "
                f"총수익률={total_return:.2f}%, "
                f"리밸런싱={len(rebal_dates)}회"
            )

        return result
