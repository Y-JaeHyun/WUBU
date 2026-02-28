"""자동 백테스트 실행 모듈.

스케줄에 따라 여러 전략의 백테스트를 실행하고 결과를 비교한다.
Feature Flag 'auto_backtest'로 제어.
"""

from datetime import datetime, timedelta
from typing import Optional

from src.backtest.engine import Backtest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AutoBacktester:
    """자동 백테스트 실행기.

    Args:
        lookback_months: 백테스트 기간 (개월).
        strategies: 실행할 전략 이름 리스트.
    """

    def __init__(
        self,
        lookback_months: int = 6,
        strategies: Optional[list[str]] = None,
    ) -> None:
        self.lookback_months = lookback_months
        self.strategies = strategies or [
            "value", "momentum", "multi_factor", "three_factor", "quality",
        ]

    def run_all(self) -> str:
        """등록된 모든 전략에 대해 백테스트를 실행한다.

        Returns:
            결과 텍스트.
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (
            datetime.now() - timedelta(days=self.lookback_months * 30)
        ).strftime("%Y%m%d")

        lines = [
            f"[자동 백테스트] {start_date} ~ {end_date}",
            "=" * 50,
        ]

        for name in self.strategies:
            try:
                strategy = self._create_strategy(name)
                if strategy is None:
                    lines.append(f"\n{name}: 전략 생성 실패")
                    continue

                bt = Backtest(
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=100_000_000,
                    rebalance_freq="monthly",
                )
                bt.run()
                results = bt.get_results()

                strategy_name = results.get("strategy_name", name)
                lines.append(f"\n[{strategy_name}]")
                lines.append(
                    f"  총수익률: {results['total_return']:+.2f}%"
                )
                lines.append(f"  CAGR: {results['cagr']:+.2f}%")
                lines.append(f"  Sharpe: {results['sharpe_ratio']:.2f}")
                lines.append(f"  MDD: {results['mdd']:.2f}%")
                lines.append(f"  승률: {results['win_rate']:.1f}%")

            except Exception as e:
                logger.error("백테스트 실패 (%s): %s", name, e)
                lines.append(f"\n{name}: 오류 - {e}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    @staticmethod
    def _create_strategy(name: str) -> Optional[object]:
        """전략명으로 전략 객체를 생성한다.

        Args:
            name: 전략 이름.

        Returns:
            Strategy 객체. 알 수 없는 이름이면 None.
        """
        try:
            if name == "value":
                from src.strategy.value import ValueStrategy

                return ValueStrategy()
            elif name == "momentum":
                from src.strategy.momentum import MomentumStrategy

                return MomentumStrategy()
            elif name == "multi_factor":
                from src.strategy.multi_factor import MultiFactorStrategy

                return MultiFactorStrategy(
                    factors=["value", "momentum"],
                    weights=[0.5, 0.5],
                    combine_method="zscore",
                    num_stocks=10,
                    apply_market_timing=True,
                )
            elif name == "quality":
                from src.strategy.quality import QualityStrategy

                return QualityStrategy()
            elif name == "three_factor":
                from src.strategy.three_factor import ThreeFactorStrategy

                return ThreeFactorStrategy(num_stocks=10)
            elif name == "pead":
                from src.strategy.pead import PEADStrategy

                return PEADStrategy(num_stocks=10)
            elif name == "shareholder_yield":
                from src.strategy.shareholder_yield import (
                    ShareholderYieldStrategy,
                )

                return ShareholderYieldStrategy(num_stocks=10)
            elif name == "low_vol_quality":
                from src.strategy.low_vol_quality import (
                    LowVolQualityStrategy,
                )

                return LowVolQualityStrategy(num_stocks=10)
            elif name == "accrual":
                from src.strategy.accrual import AccrualStrategy

                return AccrualStrategy(num_stocks=10)
            elif name == "dual_momentum":
                from src.strategy.dual_momentum import DualMomentumStrategy

                return DualMomentumStrategy()
            elif name == "etf_rotation":
                from src.strategy.etf_rotation import ETFRotationStrategy

                return ETFRotationStrategy()
            elif name == "enhanced_etf_rotation":
                from src.strategy.enhanced_etf_rotation import (
                    EnhancedETFRotationStrategy,
                )

                return EnhancedETFRotationStrategy()
            elif name == "cross_asset_momentum":
                from src.strategy.cross_asset_momentum import (
                    CrossAssetMomentumStrategy,
                )

                return CrossAssetMomentumStrategy()
            else:
                logger.warning("알 수 없는 전략: %s", name)
        except Exception as e:
            logger.error("전략 생성 실패 (%s): %s", name, e)
        return None
