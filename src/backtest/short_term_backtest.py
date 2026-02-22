"""단기 트레이딩 백테스트 엔진.

시그널 기반 단기 매매(스윙) 전략의 백테스트를 지원한다.
기존 engine.py가 월간 리밸런싱 전략을 다루는 반면,
이 엔진은 일별 포지션 추적, 손절/익절, T+1 진입 등
단기 매매 특유의 로직을 처리한다.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.short_term_base import ShortTermStrategy, ShortTermSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestPosition:
    """백테스트 포지션 추적.

    Attributes:
        ticker: 종목코드
        entry_date: 진입일 (YYYYMMDD)
        entry_price: 진입가 (T+1 시가)
        quantity: 보유 수량
        stop_loss_price: 손절가
        take_profit_price: 익절가
        strategy_name: 전략 이름
        mode: 매매 모드 (swing/daytrading)
        peak_price: 트레일링 스탑용 최고가
        metadata: 추가 정보
    """

    ticker: str
    entry_date: str
    entry_price: float
    quantity: int
    stop_loss_price: float
    take_profit_price: float
    strategy_name: str
    mode: str = "swing"
    peak_price: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price

    def to_dict(self) -> dict:
        """포지션 정보를 딕셔너리로 변환 (전략의 check_exit에 전달용)."""
        return {
            "ticker": self.ticker,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "strategy_name": self.strategy_name,
            "mode": self.mode,
            "peak_price": self.peak_price,
            "metadata": self.metadata,
        }


@dataclass
class BacktestTrade:
    """완료된 거래 기록.

    Attributes:
        ticker: 종목코드
        entry_date: 진입일
        exit_date: 청산일
        entry_price: 진입가
        exit_price: 청산가
        quantity: 수량
        pnl: 손익 (수수료 차감 후)
        pnl_pct: 손익률 (%)
        commission: 총 수수료 (매수+매도)
        reason: 청산 사유 (stop_loss/take_profit/strategy_exit)
        strategy_name: 전략 이름
    """

    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    commission: float
    reason: str
    strategy_name: str

    def to_dict(self) -> dict:
        """딕셔너리로 변환."""
        return {
            "ticker": self.ticker,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "reason": self.reason,
            "strategy_name": self.strategy_name,
        }


class ShortTermBacktest:
    """시그널 기반 단기 트레이딩 백테스트 엔진.

    Args:
        strategy: ShortTermStrategy를 상속한 전략 객체
        start_date: 백테스트 시작일 ('YYYYMMDD')
        end_date: 백테스트 종료일 ('YYYYMMDD')
        initial_capital: 초기 자본금 (기본 145,000원)
        max_positions: 최대 동시 보유 포지션 수 (기본 3)
        buy_cost: 매수 수수료율 (기본 0.015%)
        sell_cost_kospi: 매도 비용률 KOSPI (수수료+세금, 기본 0.245%)
        sell_cost_kosdaq: 매도 비용률 KOSDAQ (수수료+세금, 기본 0.165%)
    """

    def __init__(
        self,
        strategy: ShortTermStrategy,
        start_date: str,
        end_date: str,
        initial_capital: int = 145_000,
        max_positions: int = 3,
        buy_cost: float = 0.00015,
        sell_cost_kospi: float = 0.00245,
        sell_cost_kosdaq: float = 0.00165,
    ):
        self.strategy = strategy
        self.start_date = start_date.replace("-", "")
        self.end_date = end_date.replace("-", "")
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.buy_cost = buy_cost
        self.sell_cost_kospi = sell_cost_kospi
        self.sell_cost_kosdaq = sell_cost_kosdaq

        # 내부 상태
        self._positions: list[BacktestPosition] = []
        self._trades: list[BacktestTrade] = []
        self._portfolio_history: list[dict] = []
        self._pending_signals: list[ShortTermSignal] = []
        self._price_data: dict[str, pd.DataFrame] = {}
        self._cash: float = float(initial_capital)
        self._is_run: bool = False

    def _get_sell_cost(self, ticker: str) -> float:
        """종목의 매도 비용률을 반환한다.

        KOSDAQ 종목 (코드가 '2', '3'으로 시작)은 KOSDAQ 요율,
        그 외는 KOSPI 요율을 적용한다.
        """
        if ticker.startswith(("2", "3")):
            return self.sell_cost_kosdaq
        return self.sell_cost_kospi

    def _get_business_dates(self) -> list[str]:
        """시작일~종료일 사이의 영업일 리스트."""
        dates = pd.bdate_range(start=self.start_date, end=self.end_date, freq="B")
        return [d.strftime("%Y%m%d") for d in dates]

    def _get_price_on_date(
        self,
        price_df: pd.DataFrame,
        date: str,
        col: str = "close",
    ) -> Optional[float]:
        """특정 날짜의 가격을 반환한다. 없으면 직전 영업일 데이터를 사용."""
        if price_df is None or price_df.empty:
            return None

        target = pd.Timestamp(date)
        if target in price_df.index and col in price_df.columns:
            return float(price_df.loc[target, col])

        # 해당 날짜 이전의 가장 가까운 데이터
        if col not in price_df.columns:
            return None
        mask = price_df.index <= target
        if mask.any():
            return float(price_df.loc[mask].iloc[-1][col])

        return None

    def _calculate_position_value(self, position: BacktestPosition, date: str) -> float:
        """포지션의 현재 시가 평가액을 계산한다."""
        price_df = self._price_data.get(position.ticker)
        close = self._get_price_on_date(price_df, date, "close")
        if close is None:
            return position.entry_price * position.quantity
        return close * position.quantity

    def _execute_exit(
        self,
        position: BacktestPosition,
        exit_price: float,
        exit_date: str,
        reason: str,
    ) -> BacktestTrade:
        """포지션을 청산하고 거래 기록을 생성한다."""
        sell_cost_rate = self._get_sell_cost(position.ticker)

        buy_commission = position.entry_price * position.quantity * self.buy_cost
        sell_commission = exit_price * position.quantity * sell_cost_rate
        total_commission = buy_commission + sell_commission

        gross_pnl = (exit_price - position.entry_price) * position.quantity
        net_pnl = gross_pnl - total_commission

        pnl_pct = 0.0
        cost_basis = position.entry_price * position.quantity
        if cost_basis > 0:
            pnl_pct = (net_pnl / cost_basis) * 100

        # 현금 반영: 매도 금액 - 매도 수수료
        sell_proceeds = exit_price * position.quantity * (1 - sell_cost_rate)
        self._cash += sell_proceeds

        trade = BacktestTrade(
            ticker=position.ticker,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=round(net_pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            commission=round(total_commission, 2),
            reason=reason,
            strategy_name=position.strategy_name,
        )
        self._trades.append(trade)
        return trade

    def _execute_entry(
        self,
        signal: ShortTermSignal,
        entry_price: float,
        entry_date: str,
    ) -> Optional[BacktestPosition]:
        """시그널에 따라 포지션을 진입한다.

        포지션 사이즈: capital / max_positions (동일 비중).
        """
        if len(self._positions) >= self.max_positions:
            return None

        if entry_price <= 0:
            return None

        # 포지션당 할당 자금
        position_budget = self._cash / (self.max_positions - len(self._positions))
        # 매수 수수료를 고려한 최대 수량
        max_quantity = int(position_budget / (entry_price * (1 + self.buy_cost)))

        if max_quantity <= 0:
            return None

        # 실제 매수 금액 (수수료 포함)
        total_cost = max_quantity * entry_price * (1 + self.buy_cost)
        if total_cost > self._cash:
            max_quantity = int(self._cash / (entry_price * (1 + self.buy_cost)))
            if max_quantity <= 0:
                return None
            total_cost = max_quantity * entry_price * (1 + self.buy_cost)

        self._cash -= total_cost

        position = BacktestPosition(
            ticker=signal.ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=max_quantity,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
            strategy_name=signal.strategy,
            mode=signal.mode,
            peak_price=entry_price,
            metadata=signal.metadata.copy() if signal.metadata else {},
        )
        self._positions.append(position)
        return position

    def _check_exit_conditions(self, position: BacktestPosition, date: str) -> Optional[str]:
        """포지션의 청산 조건을 확인한다.

        Returns:
            청산 사유 문자열 또는 None (유지).
        """
        price_df = self._price_data.get(position.ticker)
        close = self._get_price_on_date(price_df, date, "close")
        if close is None:
            return None

        # 1. 손절 확인
        if position.stop_loss_price > 0 and close <= position.stop_loss_price:
            return "stop_loss"

        # 2. 익절 확인
        if position.take_profit_price > 0 and close >= position.take_profit_price:
            return "take_profit"

        # 3. 전략 커스텀 청산 조건
        market_data = self._build_market_data(date)
        exit_signal = self.strategy.check_exit(position.to_dict(), market_data)
        if exit_signal is not None:
            return "strategy_exit"

        # 4. peak_price 업데이트 (트레일링 스탑용)
        if close > position.peak_price:
            position.peak_price = close

        return None

    def _build_market_data(self, date: str) -> dict:
        """전략에 전달할 market_data를 구성한다.

        마지막 500 영업일 데이터를 슬라이싱하여 전달한다.
        (HighBreakout: 253일, BBSqueeze: 205일 등 전략별 최소 요구량 충족)
        """
        target = pd.Timestamp(date)
        daily_data = {}
        for ticker, df in self._price_data.items():
            if df is not None and not df.empty:
                mask = df.index <= target
                sliced = df.loc[mask].tail(500)
                if not sliced.empty:
                    daily_data[ticker] = sliced
        return {
            "daily_data": daily_data,
            "date": date,
        }

    def run(self, preloaded_data: Optional[dict[str, pd.DataFrame]] = None) -> None:
        """백테스트를 실행한다.

        Args:
            preloaded_data: 사전 로드된 가격 데이터 {ticker: DataFrame}.
                            None이면 빈 데이터로 실행 (테스트용).

        매일의 처리 순서:
        1. 보류 중인 시그널 → T+1 시가로 진입 실행
        2. 기존 포지션의 청산 조건 확인 (종가 기준)
        3. 빈 슬롯이 있으면 신규 시그널 스캔 (다음 날 실행 예약)
        4. 일별 포트폴리오 가치 기록
        """
        if preloaded_data is not None:
            self._price_data = preloaded_data

        business_dates = self._get_business_dates()
        if not business_dates:
            logger.warning("유효한 영업일이 없습니다.")
            self._is_run = True
            return

        logger.info(
            f"단기 백테스트 시작: {self.strategy.name} "
            f"({self.start_date} ~ {self.end_date}, "
            f"자본금={self.initial_capital:,}원, "
            f"최대포지션={self.max_positions})"
        )

        self._cash = float(self.initial_capital)
        self._positions = []
        self._trades = []
        self._portfolio_history = []
        self._pending_signals = []

        for i, date in enumerate(business_dates):
            # Step 1: 보류 시그널 → T+1 시가 진입
            if self._pending_signals:
                remaining_signals = []
                for signal in self._pending_signals:
                    if len(self._positions) >= self.max_positions:
                        break
                    # 이미 같은 종목 보유 중이면 스킵
                    held_tickers = {p.ticker for p in self._positions}
                    if signal.ticker in held_tickers:
                        continue
                    # T+1 시가로 진입
                    price_df = self._price_data.get(signal.ticker)
                    open_price = self._get_price_on_date(price_df, date, "open")
                    if open_price is not None and open_price > 0:
                        self._execute_entry(signal, open_price, date)
                    # 미진입 시그널은 버림 (1일 유효)
                self._pending_signals = []

            # Step 2: 청산 조건 확인 (종가 기준)
            positions_to_close = []
            for position in self._positions:
                exit_reason = self._check_exit_conditions(position, date)
                if exit_reason is not None:
                    positions_to_close.append((position, exit_reason))

            for position, reason in positions_to_close:
                price_df = self._price_data.get(position.ticker)
                exit_price = self._get_price_on_date(price_df, date, "close")
                if exit_price is not None:
                    self._execute_exit(position, exit_price, date, reason)
                    self._positions.remove(position)

            # Step 3: 빈 슬롯이 있으면 시그널 스캔
            if len(self._positions) < self.max_positions:
                market_data = self._build_market_data(date)
                try:
                    signals = self.strategy.scan_signals(market_data)
                except Exception as e:
                    logger.warning(f"시그널 스캔 실패 ({date}): {e}")
                    signals = []

                # 이미 보유 중인 종목 제외
                held_tickers = {p.ticker for p in self._positions}
                pending_tickers = {s.ticker for s in self._pending_signals}
                new_signals = [
                    s for s in signals
                    if s.ticker not in held_tickers
                    and s.ticker not in pending_tickers
                    and s.side == "buy"
                ]
                self._pending_signals.extend(new_signals)

            # Step 4: 일별 포트폴리오 가치 기록
            position_value = sum(
                self._calculate_position_value(p, date) for p in self._positions
            )
            portfolio_value = self._cash + position_value

            self._portfolio_history.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": self._cash,
                "num_positions": len(self._positions),
            })

        # 백테스트 종료: 잔여 포지션 강제 청산 (마지막 종가)
        if self._positions and business_dates:
            last_date = business_dates[-1]
            for position in list(self._positions):
                price_df = self._price_data.get(position.ticker)
                exit_price = self._get_price_on_date(price_df, last_date, "close")
                if exit_price is not None:
                    self._execute_exit(position, exit_price, last_date, "backtest_end")
            self._positions = []

            # 마지막 날 포트폴리오 가치 재계산
            if self._portfolio_history:
                self._portfolio_history[-1]["portfolio_value"] = self._cash
                self._portfolio_history[-1]["cash"] = self._cash
                self._portfolio_history[-1]["num_positions"] = 0

        self._is_run = True
        logger.info(
            f"단기 백테스트 완료: 최종 가치={self._cash:,.0f}원, "
            f"총 거래={len(self._trades)}건"
        )

    def get_results(self) -> dict:
        """성과 지표를 반환한다.

        Returns:
            dict with keys:
            - total_return: 총수익률 (소수, 예: 0.15 = 15%)
            - cagr: 연평균 수익률 (소수)
            - sharpe_ratio: 샤프비율 (연율화)
            - max_drawdown: 최대 낙폭 (음수, 예: -0.10 = -10%)
            - win_rate: 승률 (소수, 예: 0.6 = 60%)
            - profit_factor: 이익/손실 비율
            - avg_holding_days: 평균 보유일
            - total_trades: 총 거래수
            - commission_total: 총 수수료
            - avg_win_pct: 평균 수익 거래 수익률 (%)
            - avg_loss_pct: 평균 손실 거래 손익률 (%)
            - max_win_pct: 최대 수익 거래 수익률 (%)
            - max_loss_pct: 최대 손실 거래 손익률 (%)
        """
        if not self._is_run:
            raise RuntimeError("백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요.")

        history = self.get_portfolio_history()
        if history.empty:
            return self._empty_results()

        initial = float(self.initial_capital)
        final = float(history["portfolio_value"].iloc[-1])

        # 총수익률
        total_return = (final - initial) / initial if initial > 0 else 0.0

        # CAGR
        days = (history.index[-1] - history.index[0]).days
        years = days / 365.25 if days > 0 else 0.0
        if years > 0 and final > 0 and initial > 0:
            cagr = (final / initial) ** (1 / years) - 1
        else:
            cagr = 0.0

        # 일별 수익률
        daily_returns = history["portfolio_value"].pct_change().dropna()

        # 샤프비율 (연율화, 무위험이자율 0)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # MDD
        cummax = history["portfolio_value"].cummax()
        drawdown = (history["portfolio_value"] - cummax) / cummax
        max_drawdown = float(drawdown.min())

        # 거래 기반 지표
        trades = self._trades
        total_trades = len(trades)
        if total_trades > 0:
            winning = [t for t in trades if t.pnl > 0]
            losing = [t for t in trades if t.pnl <= 0]
            win_rate = len(winning) / total_trades

            gross_profit = sum(t.pnl for t in winning)
            gross_loss = abs(sum(t.pnl for t in losing))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

            # 보유일 계산
            holding_days_list = []
            for t in trades:
                try:
                    entry_dt = pd.Timestamp(t.entry_date)
                    exit_dt = pd.Timestamp(t.exit_date)
                    holding_days_list.append((exit_dt - entry_dt).days)
                except Exception:
                    holding_days_list.append(0)
            avg_holding_days = np.mean(holding_days_list) if holding_days_list else 0.0

            commission_total = sum(t.commission for t in trades)

            win_pcts = [t.pnl_pct for t in winning]
            loss_pcts = [t.pnl_pct for t in losing]
            avg_win_pct = np.mean(win_pcts) if win_pcts else 0.0
            avg_loss_pct = np.mean(loss_pcts) if loss_pcts else 0.0
            max_win_pct = max(win_pcts) if win_pcts else 0.0
            max_loss_pct = min(loss_pcts) if loss_pcts else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_holding_days = 0.0
            commission_total = 0.0
            avg_win_pct = 0.0
            avg_loss_pct = 0.0
            max_win_pct = 0.0
            max_loss_pct = 0.0

        return {
            "strategy_name": self.strategy.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": round(final, 2),
            "total_return": round(total_return, 6),
            "cagr": round(cagr, 6),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else float("inf"),
            "avg_holding_days": round(avg_holding_days, 1),
            "total_trades": total_trades,
            "commission_total": round(commission_total, 2),
            "avg_win_pct": round(avg_win_pct, 4),
            "avg_loss_pct": round(avg_loss_pct, 4),
            "max_win_pct": round(max_win_pct, 4),
            "max_loss_pct": round(max_loss_pct, 4),
        }

    def _empty_results(self) -> dict:
        """빈 결과를 반환한다."""
        return {
            "strategy_name": self.strategy.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": self.initial_capital,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "total_trades": 0,
            "commission_total": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "max_win_pct": 0.0,
            "max_loss_pct": 0.0,
        }

    def get_trades(self) -> pd.DataFrame:
        """거래 내역을 DataFrame으로 반환한다."""
        if not self._is_run:
            raise RuntimeError("백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요.")

        if not self._trades:
            return pd.DataFrame(columns=[
                "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
                "quantity", "pnl", "pnl_pct", "commission", "reason", "strategy_name",
            ])

        df = pd.DataFrame([t.to_dict() for t in self._trades])
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        return df

    def get_portfolio_history(self) -> pd.DataFrame:
        """일별 포트폴리오 가치 히스토리를 DataFrame으로 반환한다."""
        if not self._is_run:
            raise RuntimeError("백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요.")

        if not self._portfolio_history:
            return pd.DataFrame(columns=["portfolio_value", "cash", "num_positions"])

        df = pd.DataFrame(self._portfolio_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
