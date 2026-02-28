"""Walk-Forward 백테스팅 모듈.

In-Sample(학습) 구간에서 전략을 학습/최적화하고,
Out-of-Sample(검증) 구간에서 성과를 측정하는 Walk-Forward 분석을 구현한다.

롤링 윈도우 방식으로 여러 구간을 순회하며,
각 구간의 OOS 결과를 연결하여 전체 기간의 순수 OOS 성과를 산출한다.
"""

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.backtest.engine import Backtest, Strategy
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.strategy.market_timing import MarketTimingOverlay
    from src.strategy.drawdown_overlay import DrawdownOverlay
    from src.strategy.vol_targeting import VolTargetingOverlay

logger = get_logger(__name__)


class WalkForwardBacktest:
    """Walk-Forward 백테스팅 엔진.

    학습/검증 윈도우를 롤링하며 OOS(Out-of-Sample) 성과를 측정한다.

    Args:
        strategy_factory: 학습 구간 시작일/종료일을 받아 Strategy 객체를 반환하는 콜러블.
            시그니처: (train_start: str, train_end: str) -> Strategy
        full_start_date: 전체 분석 시작일 (YYYYMMDD 형식)
        full_end_date: 전체 분석 종료일 (YYYYMMDD 형식)
        train_years: 학습 구간 길이 (년, 기본 5)
        test_years: 검증 구간 길이 (년, 기본 1)
        step_months: 윈도우 이동 간격 (개월, 기본 12)
        initial_capital: 초기 자본금 (기본 1억원)
        rebalance_freq: 리밸런싱 주기 ('monthly' 또는 'quarterly')
        buy_cost: 매수 거래비용 비율
        sell_cost: 매도 거래비용 비율
        overlay: 마켓 타이밍 오버레이 객체 (선택)
        drawdown_overlay: 드로다운 디레버리징 오버레이 객체 (선택)
        vol_targeting: 변동성 타겟팅 오버레이 객체 (선택)
    """

    def __init__(
        self,
        strategy_factory: Callable[[str, str], Strategy],
        full_start_date: str,
        full_end_date: str,
        train_years: int = 5,
        test_years: int = 1,
        step_months: int = 12,
        initial_capital: int = 100_000_000,
        rebalance_freq: str = "monthly",
        buy_cost: float = 0.00015,
        sell_cost: float = 0.00245,
        overlay: Optional["MarketTimingOverlay"] = None,
        drawdown_overlay: Optional["DrawdownOverlay"] = None,
        vol_targeting: Optional["VolTargetingOverlay"] = None,
        min_rebalance_threshold: float = 0.0,
    ):
        self.strategy_factory = strategy_factory
        self.full_start_date = full_start_date.replace("-", "")
        self.full_end_date = full_end_date.replace("-", "")
        self.train_years = train_years
        self.test_years = test_years
        self.step_months = step_months
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.overlay = overlay
        self.drawdown_overlay = drawdown_overlay
        self.vol_targeting = vol_targeting
        self.min_rebalance_threshold = min_rebalance_threshold

        # 결과 저장
        self._windows: list[dict] = []
        self._window_results: list[dict] = []
        self._window_histories: list[pd.DataFrame] = []
        self._is_run = False

    def _generate_windows(self) -> list[dict]:
        """롤링 학습/검증 윈도우를 생성한다.

        전체 기간 내에서 train_years + test_years 크기의 윈도우를
        step_months 간격으로 이동시키며 생성한다.

        Returns:
            윈도우 딕셔너리 리스트. 각 항목은:
            - train_start: 학습 시작일 (YYYYMMDD)
            - train_end: 학습 종료일 (YYYYMMDD)
            - test_start: 검증 시작일 (YYYYMMDD)
            - test_end: 검증 종료일 (YYYYMMDD)
        """
        windows = []

        full_start = pd.Timestamp(self.full_start_date)
        full_end = pd.Timestamp(self.full_end_date)

        train_start = full_start

        while True:
            train_end_dt = train_start + relativedelta(years=self.train_years) - relativedelta(days=1)
            test_start_dt = train_end_dt + relativedelta(days=1)
            test_end_dt = test_start_dt + relativedelta(years=self.test_years) - relativedelta(days=1)

            # 검증 구간 종료일이 전체 종료일을 초과하면 중단
            if test_end_dt > full_end:
                break

            window = {
                "train_start": train_start.strftime("%Y%m%d"),
                "train_end": train_end_dt.strftime("%Y%m%d"),
                "test_start": test_start_dt.strftime("%Y%m%d"),
                "test_end": test_end_dt.strftime("%Y%m%d"),
            }
            windows.append(window)

            # 다음 윈도우로 이동
            train_start = train_start + relativedelta(months=self.step_months)

        return windows

    def run(self) -> None:
        """Walk-Forward 백테스트를 실행한다.

        1. 윈도우를 생성한다.
        2. 각 윈도우에 대해:
           a. strategy_factory로 학습 구간 기반 전략을 생성한다.
           b. 검증 구간에서 Backtest를 실행한다.
           c. 이전 윈도우의 최종 자본금을 다음 윈도우의 초기 자본금으로 이어간다.
        3. 결과를 저장한다.
        """
        self._windows = self._generate_windows()

        if not self._windows:
            logger.warning(
                f"생성된 윈도우가 없습니다. "
                f"전체 기간({self.full_start_date}~{self.full_end_date})이 "
                f"학습({self.train_years}년)+검증({self.test_years}년) 기간보다 짧습니다."
            )
            self._is_run = True
            return

        logger.info(
            f"Walk-Forward 백테스트 시작: "
            f"{self.full_start_date}~{self.full_end_date}, "
            f"윈도우 {len(self._windows)}개 "
            f"(학습={self.train_years}년, 검증={self.test_years}년, "
            f"스텝={self.step_months}개월)"
        )

        current_capital = self.initial_capital
        self._window_results = []
        self._window_histories = []

        for i, window in enumerate(self._windows):
            logger.info(
                f"[윈도우 {i + 1}/{len(self._windows)}] "
                f"학습: {window['train_start']}~{window['train_end']}, "
                f"검증: {window['test_start']}~{window['test_end']}, "
                f"자본금: {current_capital:,.0f}원"
            )

            # 1. 학습 구간으로 전략 생성
            try:
                strategy = self.strategy_factory(
                    window["train_start"], window["train_end"]
                )
            except Exception as e:
                logger.error(
                    f"[윈도우 {i + 1}] 전략 생성 실패: {e}. 이 윈도우를 건너뜁니다."
                )
                continue

            # 2. 검증 구간에서 백테스트 실행
            bt = Backtest(
                strategy=strategy,
                start_date=window["test_start"],
                end_date=window["test_end"],
                initial_capital=int(current_capital),
                rebalance_freq=self.rebalance_freq,
                buy_cost=self.buy_cost,
                sell_cost=self.sell_cost,
                overlay=self.overlay,
                drawdown_overlay=self.drawdown_overlay,
                vol_targeting=self.vol_targeting,
                min_rebalance_threshold=self.min_rebalance_threshold,
            )

            try:
                bt.run()
            except Exception as e:
                logger.error(
                    f"[윈도우 {i + 1}] 백테스트 실행 실패: {e}. 이 윈도우를 건너뜁니다."
                )
                continue

            # 3. 결과 수집
            try:
                result = bt.get_results()
                history = bt.get_portfolio_history()
            except Exception as e:
                logger.error(
                    f"[윈도우 {i + 1}] 결과 수집 실패: {e}. 이 윈도우를 건너뜁니다."
                )
                continue

            result["window_index"] = i + 1
            result["train_start"] = window["train_start"]
            result["train_end"] = window["train_end"]

            self._window_results.append(result)
            self._window_histories.append(history)

            # 4. 자본금 이어가기 (capital chaining)
            if not history.empty:
                current_capital = float(history["portfolio_value"].iloc[-1])

            logger.info(
                f"[윈도우 {i + 1}] 완료: "
                f"수익률={result.get('total_return', 'N/A')}%, "
                f"최종 자본금={current_capital:,.0f}원"
            )

        self._is_run = True
        logger.info(
            f"Walk-Forward 백테스트 완료: "
            f"총 {len(self._window_results)}/{len(self._windows)} 윈도우 실행 성공"
        )

    def get_oos_results(self) -> dict:
        """전체 OOS(Out-of-Sample) 성과 지표를 반환한다.

        개별 윈도우의 포트폴리오 히스토리를 연결하여
        전체 기간의 OOS 성과를 계산한다.

        Returns:
            dict with keys:
            - sharpe_ratio: 전체 OOS 샤프비율
            - cagr: 전체 OOS 연평균 수익률 (%)
            - mdd: 전체 OOS 최대 낙폭 (%)
            - total_return: 전체 OOS 총수익률 (%)
            - num_windows: 실행된 윈도우 수
            - window_results: 각 윈도우별 결과 리스트
        """
        if not self._is_run:
            raise RuntimeError(
                "Walk-Forward 백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요."
            )

        if not self._window_results:
            return {
                "sharpe_ratio": 0.0,
                "cagr": 0.0,
                "mdd": 0.0,
                "total_return": 0.0,
                "num_windows": 0,
                "window_results": [],
            }

        # 전체 OOS 포트폴리오 히스토리 연결
        oos_history = self.get_oos_portfolio_history()

        if oos_history.empty:
            return {
                "sharpe_ratio": 0.0,
                "cagr": 0.0,
                "mdd": 0.0,
                "total_return": 0.0,
                "num_windows": len(self._window_results),
                "window_results": self._window_results,
            }

        portfolio_values = oos_history["portfolio_value"]

        # 총수익률
        initial = portfolio_values.iloc[0]
        final = portfolio_values.iloc[-1]
        total_return = (final / initial - 1) * 100

        # CAGR
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        years = days / 365.25 if days > 0 else 1
        cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        # 일별 수익률
        daily_returns = portfolio_values.pct_change().dropna()

        # 샤프비율 (연율화, 무위험이자율 3%)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            risk_free_daily = 0.03 / 252
            excess_return = daily_returns.mean() - risk_free_daily
            sharpe_ratio = excess_return / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # MDD
        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax
        mdd = drawdown.min() * 100

        results = {
            "sharpe_ratio": round(float(sharpe_ratio), 2),
            "cagr": round(float(cagr), 2),
            "mdd": round(float(mdd), 2),
            "total_return": round(float(total_return), 2),
            "num_windows": len(self._window_results),
            "window_results": self._window_results,
        }

        logger.info(f"OOS 성과 지표: Sharpe={results['sharpe_ratio']}, "
                     f"CAGR={results['cagr']}%, MDD={results['mdd']}%")

        return results

    def get_oos_portfolio_history(self) -> pd.DataFrame:
        """전체 OOS 포트폴리오 가치 시계열을 반환한다.

        각 윈도우의 포트폴리오 히스토리를 시간순으로 연결한다.
        윈도우 간 겹치는 날짜가 있으면 나중 윈도우의 값을 사용한다.

        Returns:
            DataFrame with index=date, columns=[portfolio_value, cash, num_holdings]
        """
        if not self._is_run:
            raise RuntimeError(
                "Walk-Forward 백테스트가 아직 실행되지 않았습니다. run()을 먼저 호출하세요."
            )

        if not self._window_histories:
            return pd.DataFrame(columns=["portfolio_value", "cash", "num_holdings"])

        # 모든 윈도우 히스토리를 연결
        combined = pd.concat(self._window_histories)

        # 중복 날짜 제거 (나중 윈도우 우선)
        combined = combined[~combined.index.duplicated(keep="last")]

        # 시간순 정렬
        combined = combined.sort_index()

        return combined
