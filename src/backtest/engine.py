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
        rebalance_freq: 리밸런싱 주기 ('monthly' 또는 'quarterly')
        buy_cost: 매수 거래비용 비율 (기본 0.015%)
        sell_cost: 매도 거래비용 비율 (기본 0.015% 수수료 + 0.23% 세금 = 0.245%)
        overlay: 마켓 타이밍 오버레이 객체 (선택, 기본 None)
    """

    def __init__(
        self,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        initial_capital: int = 100_000_000,
        rebalance_freq: str = "monthly",
        buy_cost: float = 0.00015,
        sell_cost: float = 0.00245,
        overlay: Optional["MarketTimingOverlay"] = None,
    ):
        self.strategy = strategy
        self.start_date = start_date.replace("-", "")
        self.end_date = end_date.replace("-", "")
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.overlay = overlay

        # 결과 저장
        self._portfolio_history: list[dict] = []
        self._trades: list[dict] = []
        self._rebalance_dates: list[str] = []
        self._is_run = False

    def _get_rebalance_dates(self) -> list[str]:
        """리밸런싱 날짜 리스트를 생성한다.

        각 월(또는 분기) 첫 영업일을 리밸런싱 날짜로 사용한다.
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="B")
        if dates.empty:
            return []

        if self.rebalance_freq == "monthly":
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

    def _fetch_price(self, ticker: str) -> pd.DataFrame:
        """종목의 전체 기간 가격 데이터를 가져온다."""
        return get_price_data(ticker, self.start_date, self.end_date)

    def run(self) -> None:
        """백테스트를 실행한다.

        마켓 타이밍 오버레이가 설정된 경우, 리밸런싱 시점마다
        지수 데이터를 기반으로 오버레이 신호를 생성하고 비중을 조절한다.
        """
        logger.info(
            f"백테스트 시작: {self.strategy.name} "
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

        # 마켓 타이밍 오버레이용 지수 데이터 로드
        index_prices: pd.Series = pd.Series(dtype=float)
        if self.overlay is not None:
            try:
                from src.data.index_collector import get_index_data
                index_df = get_index_data(
                    self.overlay.reference_index,
                    self.start_date,
                    self.end_date,
                )
                if not index_df.empty and "close" in index_df.columns:
                    index_prices = index_df["close"]
                    logger.info(
                        f"지수 데이터 로드 완료: {self.overlay.reference_index}, "
                        f"{len(index_prices)}일"
                    )
            except Exception as e:
                logger.warning(f"지수 데이터 로드 실패: {e}. 오버레이 미적용.")
            # 오버레이 상태 초기화
            self.overlay.reset()

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

                data = {
                    "fundamentals": fundamentals,
                    "prices": price_cache,
                    "index_prices": index_prices,
                }

                # 전략 시그널 생성
                try:
                    signals = self.strategy.generate_signals(date, data)
                except Exception as e:
                    logger.error(f"시그널 생성 실패 ({date}): {e}")
                    signals = {}

                # 마켓 타이밍 오버레이 적용
                if signals and self.overlay is not None and not index_prices.empty:
                    try:
                        # 현재 날짜까지의 지수 데이터 슬라이싱
                        target_date = pd.Timestamp(date)
                        available_prices = index_prices[index_prices.index <= target_date]

                        if not available_prices.empty:
                            if self.overlay.switch_mode == "gradual":
                                signals = self.overlay.apply_overlay_gradual(
                                    signals, available_prices
                                )
                            else:
                                signal = self.overlay.get_signal(available_prices)
                                signals = self.overlay.apply_overlay(signals, signal)
                    except Exception as e:
                        logger.warning(f"오버레이 적용 실패 ({date}): {e}")

                if signals:
                    # 필요한 가격 데이터를 캐시에 로드
                    for ticker in signals:
                        if ticker not in price_cache:
                            try:
                                price_cache[ticker] = self._fetch_price(ticker)
                            except Exception as e:
                                logger.warning(f"가격 데이터 로드 실패: {ticker} - {e}")

                    # 현재 포트폴리오 가치 계산
                    portfolio_value = cash
                    for ticker, qty in holdings.items():
                        if ticker in price_cache and not price_cache[ticker].empty:
                            price_df = price_cache[ticker]
                            price_on_date = self._get_price_on_date(price_df, date)
                            if price_on_date is not None:
                                portfolio_value += qty * price_on_date

                    # 기존 보유 종목 매도
                    for ticker, qty in list(holdings.items()):
                        if qty > 0:
                            if ticker in price_cache and not price_cache[ticker].empty:
                                price = self._get_price_on_date(price_cache[ticker], date)
                                if price is not None:
                                    proceeds = qty * price * (1 - self.sell_cost)
                                    cash += proceeds
                                    self._trades.append({
                                        "date": date,
                                        "ticker": ticker,
                                        "action": "sell",
                                        "quantity": qty,
                                        "price": price,
                                        "cost": qty * price * self.sell_cost,
                                    })
                    holdings.clear()

                    # 신규 매수
                    for ticker, weight in signals.items():
                        if weight <= 0 or ticker not in price_cache:
                            continue
                        price_df = price_cache[ticker]
                        if price_df.empty:
                            continue
                        price = self._get_price_on_date(price_df, date)
                        if price is None or price <= 0:
                            continue

                        target_amount = portfolio_value * weight
                        buy_qty = int(target_amount / (price * (1 + self.buy_cost)))
                        if buy_qty <= 0:
                            continue

                        cost = buy_qty * price * (1 + self.buy_cost)
                        if cost > cash:
                            buy_qty = int(cash / (price * (1 + self.buy_cost)))
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
            "strategy_name": self.strategy.name,
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
