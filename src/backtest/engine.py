"""백테스팅 엔진 모듈.

간단하지만 확장 가능한 백테스팅 엔진을 제공한다.
리밸런싱 주기, 거래비용, 다양한 성과 지표를 지원한다.
마켓 타이밍 오버레이를 통한 비중 조절을 지원한다.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.collector import get_price_data, get_all_fundamentals
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.strategy.market_timing import MarketTimingOverlay
    from src.strategy.drawdown_overlay import DrawdownOverlay
    from src.strategy.vol_targeting import VolTargetingOverlay

logger = get_logger(__name__)


class Strategy(ABC):
    """트레이딩 전략의 추상 베이스 클래스.

    모든 전략은 이 클래스를 상속하여 generate_signals 메서드를 구현해야 한다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름."""
        ...

    @abstractmethod
    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: 전략이 참조할 수 있는 데이터 딕셔너리
                  (예: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame]})

        Returns:
            종목코드: 비중 딕셔너리 (예: {'005930': 0.05, '000660': 0.05, ...})
            비중의 합은 1.0 이하여야 한다 (나머지는 현금).
        """
        ...


class Backtest:
    """백테스팅 엔진.

    전략 객체를 받아 과거 데이터를 기반으로 시뮬레이션을 실행하고 성과를 측정한다.

    Args:
        strategy: Strategy를 상속한 전략 객체
        start_date: 백테스트 시작일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        end_date: 백테스트 종료일
        initial_capital: 초기 자본금 (기본 1억원)
        rebalance_freq: 리밸런싱 주기 ('weekly', 'biweekly', 'monthly', 'quarterly')
        buy_cost: 매수 거래비용 비율 (기본 0.015%)
        sell_cost: 매도 거래비용 비율 (기본 0.015% 수수료 + 0.23% 세금 = 0.245%)
        overlay: 마켓 타이밍 오버레이 객체 (선택, 기본 None)
        drawdown_overlay: 드로다운 디레버리징 오버레이 객체 (선택, 기본 None)
        vol_targeting: 변동성 타겟팅 오버레이 객체 (선택, 기본 None)
        min_rebalance_threshold: 비중 변화 임계값. 이 값 미만의 비중 변화는
            거래하지 않는다. (기본 0.0 = 모든 변화 거래)
        lookback_days: 가격 데이터 수집 시 start_date 이전 영업일 수 (기본 400)
        pool_strategies: 3-Pool 모드용 전략/비중 매핑 (선택).
            예: {"long_term": (MultiFactor(), 0.7), "etf_rotation": (ETFRotation(), 0.3)}
            설정하면 strategy 파라미터는 무시된다.
        etf_sell_cost: ETF 매도 거래비용 비율 (기본 0.015%, 거래세 면제)
    """

    def __init__(
        self,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000_000,
        rebalance_freq: str = "monthly",
        buy_cost: float = 0.00015,
        sell_cost: float = 0.00245,
        overlay: Optional["MarketTimingOverlay"] = None,
        drawdown_overlay: Optional["DrawdownOverlay"] = None,
        vol_targeting: Optional["VolTargetingOverlay"] = None,
        min_rebalance_threshold: float = 0.0,
        lookback_days: int = 400,
        force_rebalance_every: int = 4,
        pool_strategies: Optional[dict[str, tuple[Strategy, float]]] = None,
        etf_sell_cost: float = 0.00015,
    ):
        self.strategy = strategy
        self.start_date = start_date.replace("-", "")
        self.end_date = end_date.replace("-", "")
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.overlay = overlay
        self.drawdown_overlay = drawdown_overlay
        self.vol_targeting = vol_targeting
        self.min_rebalance_threshold = min_rebalance_threshold
        self.lookback_days = lookback_days
        self.force_rebalance_every = force_rebalance_every
        self.pool_strategies = pool_strategies
        self.etf_sell_cost = etf_sell_cost

        # 가격 데이터 수집 시작일: start_date에서 lookback_days만큼 이전
        # 전략이 모멘텀 스코어, 상장기간 필터 등에 충분한 과거 데이터를 사용할 수 있도록 함
        data_start = pd.Timestamp(self.start_date) - pd.tseries.offsets.BDay(lookback_days)
        self._data_start_date = data_start.strftime("%Y%m%d")

        # 결과 저장
        self._portfolio_history: list[dict] = []
        self._trades: list[dict] = []
        self._rebalance_dates: list[str] = []
        self._rebalance_count: int = 0
        self._is_run = False

    def _get_rebalance_dates(self) -> list[str]:
        """리밸런싱 날짜 리스트를 생성한다.

        지원 주기: weekly, biweekly, monthly, quarterly.
        각 기간의 첫 영업일을 리밸런싱 날짜로 사용한다.
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="B")
        if dates.empty:
            return []

        if self.rebalance_freq == "weekly":
            # 각 주의 첫 영업일
            rebal = dates.to_series().groupby(dates.to_period("W")).first()
        elif self.rebalance_freq == "biweekly":
            # 각 주의 첫 영업일 중 짝수 주만 선택
            weekly = dates.to_series().groupby(dates.to_period("W")).first()
            rebal = weekly.iloc[::2]
        elif self.rebalance_freq == "monthly":
            # 각 월의 첫 영업일
            rebal = dates.to_series().groupby(dates.to_period("M")).first()
        elif self.rebalance_freq == "quarterly":
            # 각 분기의 첫 영업일
            rebal = dates.to_series().groupby(dates.to_period("Q")).first()
        else:
            raise ValueError(f"지원하지 않는 리밸런싱 주기: {self.rebalance_freq}")

        return [d.strftime("%Y%m%d") for d in rebal]

    def _get_business_dates(self) -> list[str]:
        """시작일~종료일 사이의 영업일 리스트."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="B")
        return [d.strftime("%Y%m%d") for d in dates]

    def _calc_portfolio_state(
        self,
        holdings: dict[str, int],
        price_cache: dict[str, pd.DataFrame],
        cash: float,
        date: str,
    ) -> tuple[float, dict[str, float]]:
        """현재 포트폴리오 가치와 종목별 비중을 계산한다."""
        portfolio_value = cash
        for ticker, qty in holdings.items():
            if ticker in price_cache and not price_cache[ticker].empty:
                price = self._get_price_on_date(price_cache[ticker], date)
                if price is not None:
                    portfolio_value += qty * price

        current_weights: dict[str, float] = {}
        if portfolio_value > 0:
            for ticker, qty in holdings.items():
                if ticker in price_cache and not price_cache[ticker].empty:
                    price = self._get_price_on_date(price_cache[ticker], date)
                    if price is not None:
                        current_weights[ticker] = qty * price / portfolio_value

        return portfolio_value, current_weights

    def _fetch_price(self, ticker: str) -> pd.DataFrame:
        """종목의 전체 기간 가격 데이터를 가져온다.

        lookback_days 만큼 start_date 이전부터 데이터를 수집하여,
        모멘텀·상장기간 필터 등이 첫 리밸런싱부터 충분한 과거 데이터를 갖도록 한다.
        주식 API에서 데이터가 빈 경우 ETF API로 fallback한다.
        """
        df = get_price_data(ticker, self._data_start_date, self.end_date)
        if df.empty:
            try:
                from src.data.etf_collector import get_etf_price
                df = get_etf_price(ticker, self._data_start_date, self.end_date)
                if not df.empty:
                    logger.info(f"ETF fallback 성공: {ticker}")
            except Exception as e:
                logger.debug(f"ETF fallback 실패: {ticker} - {e}")
        return df

    def _get_strategy_name(self) -> str:
        """전략 이름을 반환한다. 3-Pool 모드이면 풀 구성을 반영한다."""
        if self.pool_strategies:
            parts = "+".join(
                f"{strat.name}({pct:.0%})"
                for _, (strat, pct) in self.pool_strategies.items()
            )
            return f"MultiPool[{parts}]"
        return self.strategy.name

    def _get_sell_cost(self, ticker: str) -> float:
        """종목의 매도 거래비용 비율을 반환한다.

        ETF는 거래세가 면제되므로 etf_sell_cost를, 일반 주식은 sell_cost를 적용한다.
        """
        from src.utils.market_utils import is_etf

        if is_etf(ticker):
            return self.etf_sell_cost
        return self.sell_cost

    def run(self) -> None:
        """백테스트를 실행한다.

        마켓 타이밍 오버레이가 설정된 경우, 리밸런싱 시점마다
        지수 데이터를 기반으로 오버레이 신호를 생성하고 비중을 조절한다.
        """
        if self.pool_strategies:
            pool_desc = ", ".join(
                f"{name}({strat.name}:{pct:.0%})"
                for name, (strat, pct) in self.pool_strategies.items()
            )
            strategy_name = f"3-Pool[{pool_desc}]"
        else:
            strategy_name = self.strategy.name

        logger.info(
            f"백테스트 시작: {strategy_name} "
            f"({self.start_date} ~ {self.end_date}, "
            f"자본금={self.initial_capital:,}원, "
            f"리밸런싱={self.rebalance_freq}, "
            f"오버레이={'있음' if self.overlay else '없음'})"
        )

        rebalance_dates = self._get_rebalance_dates()
        business_dates = self._get_business_dates()

        if not rebalance_dates or not business_dates:
            logger.error("유효한 영업일이 없습니다.")
            return

        self._rebalance_dates = rebalance_dates

        # 상태 초기화
        cash = float(self.initial_capital)
        holdings: dict[str, int] = {}  # ticker -> 보유 수량
        price_cache: dict[str, pd.DataFrame] = {}  # ticker -> 가격 DataFrame

        # 지수 데이터 로드 (마켓 타이밍 오버레이 + 잔차 모멘텀 등 전략용)
        index_prices: pd.Series = pd.Series(dtype=float)
        try:
            from src.data.index_collector import get_index_data
            reference_index = (
                self.overlay.reference_index if self.overlay is not None
                else "KOSPI"
            )
            index_df = get_index_data(
                reference_index,
                self._data_start_date,
                self.end_date,
            )
            if not index_df.empty and "close" in index_df.columns:
                index_prices = index_df["close"]
                logger.info(
                    f"지수 데이터 로드 완료: {reference_index}, "
                    f"{len(index_prices)}일"
                )
        except Exception as e:
            logger.warning(f"지수 데이터 로드 실패: {e}.")

        if self.overlay is not None:
            # 오버레이 상태 초기화
            self.overlay.reset()

        # 드로다운 오버레이 상태 초기화
        if self.drawdown_overlay is not None:
            self.drawdown_overlay.reset()

        logger.info(f"리밸런싱 날짜: {len(rebalance_dates)}회")

        for date in business_dates:
            # 리밸런싱 날짜인지 확인
            if date in rebalance_dates:
                logger.info(f"리밸런싱 실행: {date}")

                # 전략에 전달할 데이터 수집
                try:
                    fundamentals = get_all_fundamentals(date)
                except Exception as e:
                    logger.warning(f"펀더멘탈 데이터 조회 실패 ({date}): {e}")
                    fundamentals = pd.DataFrame()

                # 유니버스 종목의 가격 데이터를 사전 로드한다.
                # 모멘텀 등 가격 기반 필터/스코어링이 필요한 전략을 위해,
                # generate_signals 호출 전에 price_cache를 채운다.
                if not fundamentals.empty and "ticker" in fundamentals.columns:
                    universe_tickers = fundamentals["ticker"].tolist()
                    loaded_count = 0
                    for ticker in universe_tickers:
                        if ticker not in price_cache:
                            try:
                                price_cache[ticker] = self._fetch_price(ticker)
                                loaded_count += 1
                            except Exception as e:
                                logger.debug(
                                    f"유니버스 가격 사전 로드 실패: {ticker} - {e}"
                                )
                    if loaded_count > 0:
                        logger.info(
                            f"유니버스 가격 사전 로드: {loaded_count}개 신규 "
                            f"(캐시 총 {len(price_cache)}개)"
                        )

                data = {
                    "fundamentals": fundamentals,
                    "prices": price_cache,
                    "index_prices": index_prices,
                }

                # 현재 보유 종목 정보를 전략에 전달 (Turnover 감소용)
                if hasattr(self.strategy, "update_holdings"):
                    self.strategy.update_holdings(set(holdings.keys()))

                # 전략 시그널 생성
                try:
                    if self.pool_strategies:
                        # 3-Pool 모드: 각 풀 전략 시그널 → 풀 비중 스케일링 → 병합
                        merged: dict[str, float] = {}
                        for pool_name, (strat, pct) in self.pool_strategies.items():
                            if pct <= 0:
                                continue
                            pool_signals = strat.generate_signals(date, data)
                            for ticker, weight in pool_signals.items():
                                scaled = round(weight * pct, 6)
                                merged[ticker] = round(
                                    merged.get(ticker, 0.0) + scaled, 6
                                )
                        signals = merged
                    else:
                        signals = self.strategy.generate_signals(date, data)
                except Exception as e:
                    logger.error(f"시그널 생성 실패 ({date}): {e}")
                    signals = {}

                # C4: 마켓 타이밍 오버레이 적용 (전략 내부 MT가 있으면 스킵)
                strategy_has_mt = (
                    hasattr(self.strategy, "market_timing")
                    and self.strategy.market_timing is not None
                )
                if signals and self.overlay is not None and not index_prices.empty:
                    if strategy_has_mt:
                        logger.info(
                            f"전략 내부 MarketTiming 감지 → 엔진 overlay 스킵 ({date})"
                        )
                    else:
                        try:
                            target_date = pd.Timestamp(date)
                            available_prices = index_prices[
                                index_prices.index <= target_date
                            ]
                            if not available_prices.empty:
                                if self.overlay.switch_mode == "gradual":
                                    signals = self.overlay.apply_overlay_gradual(
                                        signals, available_prices
                                    )
                                else:
                                    signal = self.overlay.get_signal(available_prices)
                                    signals = self.overlay.apply_overlay(
                                        signals, signal
                                    )
                        except Exception as e:
                            logger.warning(f"오버레이 적용 실패 ({date}): {e}")

                # 드로다운 오버레이 적용
                if signals and self.drawdown_overlay is not None:
                    try:
                        portfolio_value_now = cash
                        for t, q in holdings.items():
                            if t in price_cache and not price_cache[t].empty:
                                p = self._get_price_on_date(price_cache[t], date)
                                if p is not None:
                                    portfolio_value_now += q * p
                        signals = self.drawdown_overlay.apply_overlay(
                            signals, portfolio_value_now
                        )
                    except Exception as e:
                        logger.warning(f"드로다운 오버레이 적용 실패 ({date}): {e}")

                # 변동성 타겟팅 적용
                if signals and self.vol_targeting is not None:
                    try:
                        if self._portfolio_history:
                            pv_series = pd.Series(
                                [h["portfolio_value"] for h in self._portfolio_history],
                                index=pd.to_datetime(
                                    [h["date"] for h in self._portfolio_history]
                                ),
                            )
                            signals = self.vol_targeting.apply(signals, pv_series)
                        # else: 첫 리밸런싱에는 변동성 데이터 없으므로
                        # 기본 exposure(1.0) 유지 — 의도적 스킵 (M6)
                    except Exception as e:
                        logger.warning(f"변동성 타겟팅 적용 실패 ({date}): {e}")

                # C9: 오버레이 스태킹 최소 노출 하한
                MIN_COMBINED_EXPOSURE = 0.10
                if signals:
                    total_weight = sum(signals.values())
                    if 0 < total_weight < MIN_COMBINED_EXPOSURE:
                        scale = MIN_COMBINED_EXPOSURE / total_weight
                        signals = {t: w * scale for t, w in signals.items()}
                        logger.info(
                            f"최소 노출 하한 적용: {total_weight:.2%} → "
                            f"{MIN_COMBINED_EXPOSURE:.0%}"
                        )

                if signals:
                    # M4: 시그널 종목 + 보유 종목 모두 가격 캐시에 로드
                    all_tickers = set(signals.keys()) | set(holdings.keys())
                    for ticker in all_tickers:
                        if ticker not in price_cache:
                            try:
                                price_cache[ticker] = self._fetch_price(ticker)
                            except Exception as e:
                                logger.warning(
                                    f"가격 데이터 로드 실패: {ticker} - {e}"
                                )

                    # C1: 포트폴리오 가치/비중 계산
                    portfolio_value, current_weights = (
                        self._calc_portfolio_state(
                            holdings, price_cache, cash, date
                        )
                    )

                    # M1: 누적 드리프트 보정 — N회마다 강제 리밸런싱
                    self._rebalance_count += 1
                    effective_threshold = (
                        0.0
                        if (
                            self.force_rebalance_every > 0
                            and self._rebalance_count
                            % self.force_rebalance_every
                            == 0
                        )
                        else self.min_rebalance_threshold
                    )

                    # ── 차등 리밸런싱 ──
                    # 1) 타겟에 없는 종목 전량 매도
                    for ticker in list(holdings.keys()):
                        if ticker not in signals:
                            qty = holdings[ticker]
                            sold = False
                            if qty > 0 and ticker in price_cache:
                                price = self._get_price_on_date(
                                    price_cache[ticker], date
                                )
                                if price is not None:
                                    sell_cost_rate = self._get_sell_cost(ticker)
                                    proceeds = qty * price * (1 - sell_cost_rate)
                                    cash += proceeds
                                    self._trades.append({
                                        "date": date,
                                        "ticker": ticker,
                                        "action": "sell",
                                        "quantity": qty,
                                        "price": price,
                                        "cost": qty * price * sell_cost_rate,
                                    })
                                    sold = True
                            # C2: 매도 성공 시에만 holdings에서 제거
                            if sold:
                                del holdings[ticker]

                    # C1: Step 1 완료 후 재계산
                    portfolio_value, current_weights = (
                        self._calc_portfolio_state(
                            holdings, price_cache, cash, date
                        )
                    )

                    # 2) 비중 줄여야 할 종목: 차액만 매도
                    for ticker in list(holdings.keys()):
                        if ticker not in signals:
                            continue
                        cur_w = current_weights.get(ticker, 0.0)
                        tgt_w = signals[ticker]
                        delta = cur_w - tgt_w
                        if delta > effective_threshold:
                            price = self._get_price_on_date(
                                price_cache[ticker], date
                            )
                            if price is not None and price > 0:
                                sell_cost_rate = self._get_sell_cost(ticker)
                                sell_amount = portfolio_value * delta
                                # M2: 매수측과 동일하게 거래비용 반영
                                sell_qty = int(
                                    sell_amount / (price * (1 - sell_cost_rate))
                                )
                                sell_qty = min(sell_qty, holdings[ticker])
                                if sell_qty > 0:
                                    proceeds = (
                                        sell_qty * price * (1 - sell_cost_rate)
                                    )
                                    cash += proceeds
                                    holdings[ticker] -= sell_qty
                                    if holdings[ticker] <= 0:
                                        del holdings[ticker]
                                    self._trades.append({
                                        "date": date,
                                        "ticker": ticker,
                                        "action": "sell",
                                        "quantity": sell_qty,
                                        "price": price,
                                        "cost": sell_qty * price * sell_cost_rate,
                                    })

                    # C1: Step 2 완료 후 재계산
                    portfolio_value, current_weights = (
                        self._calc_portfolio_state(
                            holdings, price_cache, cash, date
                        )
                    )

                    # 3) 비중 늘려야 할 종목 + 신규 종목: 차액만 매수
                    for ticker, tgt_w in signals.items():
                        if tgt_w <= 0 or ticker not in price_cache:
                            continue
                        price_df = price_cache[ticker]
                        if price_df.empty:
                            continue
                        price = self._get_price_on_date(price_df, date)
                        if price is None or price <= 0:
                            continue

                        cur_w = current_weights.get(ticker, 0.0)
                        delta = tgt_w - cur_w
                        if delta <= effective_threshold:
                            continue  # 변화 작으면 스킵

                        buy_amount = portfolio_value * delta
                        buy_qty = int(buy_amount / (price * (1 + self.buy_cost)))
                        if buy_qty <= 0:
                            continue

                        cost = buy_qty * price * (1 + self.buy_cost)
                        if cost > cash:
                            buy_qty = int(
                                cash / (price * (1 + self.buy_cost))
                            )
                            cost = buy_qty * price * (1 + self.buy_cost)

                        if buy_qty > 0:
                            cash -= cost
                            holdings[ticker] = holdings.get(ticker, 0) + buy_qty
                            self._trades.append({
                                "date": date,
                                "ticker": ticker,
                                "action": "buy",
                                "quantity": buy_qty,
                                "price": price,
                                "cost": buy_qty * price * self.buy_cost,
                            })

            # 일별 포트폴리오 가치 기록
            portfolio_value = cash
            for ticker, qty in holdings.items():
                if ticker in price_cache and not price_cache[ticker].empty:
                    price = self._get_price_on_date(price_cache[ticker], date)
                    if price is not None:
                        portfolio_value += qty * price

            self._portfolio_history.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "num_holdings": len([t for t, q in holdings.items() if q > 0]),
            })

        self._is_run = True
        logger.info(
            f"백테스트 완료: 최종 가치={portfolio_value:,.0f}원, "
            f"총 거래={len(self._trades)}건"
        )

    def _get_price_on_date(self, price_df: pd.DataFrame, date: str) -> Optional[float]:
        """특정 날짜의 종가를 반환한다. 해당일 데이터가 없으면 직전 데이터를 사용."""
        if price_df.empty:
            return None

        target = pd.Timestamp(date)
        if target in price_df.index:
            return float(price_df.loc[target, "close"])

        # 해당 날짜 이전의 가장 가까운 데이터
        mask = price_df.index <= target
        if mask.any():
            return float(price_df.loc[mask].iloc[-1]["close"])

        return None

    def get_portfolio_history(self) -> pd.DataFrame:
        """포트폴리오 가치 시계열을 반환한다.

        Returns:
            DataFrame with columns: ['date', 'portfolio_value', 'cash', 'num_holdings']
        """
        if not self._is_run:
            raise RuntimeError("백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요.")

        df = pd.DataFrame(self._portfolio_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def get_trades(self) -> pd.DataFrame:
        """거래 내역을 반환한다."""
        if not self._is_run:
            raise RuntimeError("백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요.")

        if not self._trades:
            return pd.DataFrame(columns=["date", "ticker", "action", "quantity", "price", "cost"])

        df = pd.DataFrame(self._trades)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_results(self) -> dict:
        """성과 지표를 반환한다.

        Returns:
            dict with keys:
            - total_return: 총수익률 (%)
            - cagr: 연평균 수익률 (%)
            - sharpe_ratio: 샤프비율 (무위험이자율 3% 가정)
            - mdd: 최대 낙폭 (%)
            - win_rate: 리밸런싱 구간 승률 (%)
            - total_trades: 총 거래 횟수
            - start_date: 시작일
            - end_date: 종료일
            - initial_capital: 초기 자본금
            - final_value: 최종 포트폴리오 가치
        """
        if not self._is_run:
            raise RuntimeError("백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요.")

        history = self.get_portfolio_history()

        if history.empty:
            return {"error": "포트폴리오 이력이 비어 있습니다."}

        initial = self.initial_capital
        final = history["portfolio_value"].iloc[-1]

        # 총수익률
        total_return = (final / initial - 1) * 100

        # CAGR
        days = (history.index[-1] - history.index[0]).days
        years = days / 365.25 if days > 0 else 1
        cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

        # 일별 수익률
        daily_returns = history["portfolio_value"].pct_change().dropna()

        # 샤프비율 (연율화, 무위험이자율 3%)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            risk_free_daily = 0.03 / 252
            excess_return = daily_returns.mean() - risk_free_daily
            sharpe_ratio = excess_return / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # MDD
        cummax = history["portfolio_value"].cummax()
        drawdown = (history["portfolio_value"] - cummax) / cummax
        mdd = drawdown.min() * 100

        # 리밸런싱 구간 승률
        rebal_values = []
        for rd in self._rebalance_dates:
            rd_ts = pd.Timestamp(rd)
            if rd_ts in history.index:
                rebal_values.append(history.loc[rd_ts, "portfolio_value"])

        if len(rebal_values) > 1:
            rebal_returns = pd.Series(rebal_values).pct_change().dropna()
            win_rate = (rebal_returns > 0).sum() / len(rebal_returns) * 100
        else:
            win_rate = 0.0

        results = {
            "strategy_name": self._get_strategy_name(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": initial,
            "final_value": round(final, 0),
            "total_return": round(total_return, 2),
            "cagr": round(cagr, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "mdd": round(mdd, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(self._trades),
            "rebalance_count": len(self._rebalance_dates),
        }

        logger.info(f"성과 지표: {results}")
        return results
