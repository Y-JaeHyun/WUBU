"""드로다운 기반 디레버리징 오버레이 모듈.

포트폴리오 가치의 고점 대비 낙폭(drawdown)을 모니터링하여
단계별로 포지션 노출을 줄이는 오버레이를 제공한다.
독립 전략이 아닌 오버레이로서 다른 전략의 비중을 조절하는 데 사용한다.

동작 예시 (기본 설정):
  - drawdown >= 0%:   exposure = 1.00 (정상)
  - drawdown <= -10%: exposure = 0.75
  - drawdown <= -15%: exposure = 0.50
  - drawdown <= -20%: exposure = 0.25

회복 버퍼(recovery_buffer)를 통해 임계값 근처에서의 빈번한 전환을 방지한다.
예: -10% 임계값에서 0.75로 전환 후, drawdown이 -8%(-10% + 2%) 이상으로
회복되어야 1.0으로 복귀한다.
"""

from typing import List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DrawdownOverlay:
    """드로다운 기반 디레버리징 오버레이.

    포트폴리오 가치의 고점 대비 낙폭에 따라 포지션 노출 비율을
    단계적으로 줄인다. 회복 시에는 recovery_buffer만큼 추가 회복해야
    노출 비율을 복원하여, 임계값 근처에서의 빈번한 전환(whipsaw)을 방지한다.

    Args:
        thresholds: (drawdown_pct, exposure_ratio) 튜플 리스트.
            drawdown_pct는 음수(예: -0.10), exposure_ratio는 0~1 사이 값.
            drawdown_pct 기준 내림차순(가장 작은 값 먼저)으로 정렬된다.
        recovery_buffer: 회복 시 요구되는 추가 마진 (기본 0.02 = 2%).
            현재 적용 중인 임계값의 drawdown_pct + recovery_buffer를
            초과해야 한 단계 위로 복귀한다.
    """

    def __init__(
        self,
        thresholds: Optional[List[Tuple[float, float]]] = None,
        recovery_buffer: float = 0.02,
    ):
        if thresholds is None:
            thresholds = [(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)]

        # drawdown_pct 기준 내림차순 정렬 (가장 작은 = 가장 심한 낙폭 먼저)
        self.thresholds: List[Tuple[float, float]] = sorted(
            thresholds, key=lambda t: t[0]
        )
        self.recovery_buffer = recovery_buffer

        # 내부 상태
        self._peak: float = 0.0
        self._current_exposure: float = 1.0

        logger.info(
            f"DrawdownOverlay 초기화: thresholds={self.thresholds}, "
            f"recovery_buffer={self.recovery_buffer}"
        )

    def reset(self) -> None:
        """내부 상태를 초기화한다. 새로운 백테스트 시작 시 호출."""
        self._peak = 0.0
        self._current_exposure = 1.0
        logger.info("DrawdownOverlay 상태 초기화")

    def update(self, portfolio_value: float) -> float:
        """포트폴리오 가치를 갱신하고 현재 노출 비율을 반환한다.

        1. 고점(peak)을 갱신한다.
        2. drawdown = (현재가치 - 고점) / 고점 을 계산한다.
        3. 임계값 테이블을 순회하며 해당 노출 비율을 결정한다.
        4. 회복 시에는 recovery_buffer만큼 추가 회복이 필요하다.

        Args:
            portfolio_value: 현재 포트폴리오 가치

        Returns:
            노출 비율 (0.0 ~ 1.0)
        """
        # 고점 갱신
        if portfolio_value > self._peak:
            self._peak = portfolio_value

        # 고점이 0 이하면 계산 불가 -> 기본값 반환
        if self._peak <= 0:
            return 1.0

        # 드로다운 계산
        drawdown = (portfolio_value - self._peak) / self._peak

        # 드로다운에 해당하는 목표 노출 비율 결정
        # thresholds는 drawdown_pct 오름차순(가장 심한 낙폭 먼저)으로 정렬되어 있다.
        # 예: [(-0.20, 0.25), (-0.15, 0.50), (-0.10, 0.75)]
        # 가장 심한 임계값부터 확인하여, 첫 번째로 해당하는 단계를 적용한다.
        # dd=-0.15일 때: -0.20에는 미달, -0.15에 해당 -> 0.50 적용
        target_exposure = 1.0
        for dd_threshold, exposure_ratio in self.thresholds:
            if drawdown <= dd_threshold:
                target_exposure = exposure_ratio
                break

        # 현재 노출이 1.0(정상)이면 즉시 디레버리징 적용
        if self._current_exposure >= 1.0:
            self._current_exposure = target_exposure
            if target_exposure < 1.0:
                logger.info(
                    f"디레버리징 진입: drawdown={drawdown:.4f}, "
                    f"exposure={target_exposure:.2f}"
                )
            return self._current_exposure

        # 현재 디레버리징 상태에서의 로직
        if target_exposure < self._current_exposure:
            # 더 심한 낙폭 -> 추가 디레버리징 (즉시 적용)
            self._current_exposure = target_exposure
            logger.info(
                f"추가 디레버리징: drawdown={drawdown:.4f}, "
                f"exposure={target_exposure:.2f}"
            )
        elif target_exposure > self._current_exposure:
            # 회복 방향 -> recovery_buffer 확인
            # 현재 노출에 해당하는 임계값을 찾는다
            current_threshold = self._find_threshold_for_exposure(
                self._current_exposure
            )
            if current_threshold is not None:
                recovery_level = current_threshold + self.recovery_buffer
                if drawdown > recovery_level:
                    # 충분히 회복 -> 한 단계 위로
                    self._current_exposure = self._find_next_exposure_up(
                        self._current_exposure
                    )
                    logger.info(
                        f"회복 단계 상승: drawdown={drawdown:.4f}, "
                        f"recovery_level={recovery_level:.4f}, "
                        f"exposure={self._current_exposure:.2f}"
                    )
            # recovery_buffer를 충족하지 못하면 현재 상태 유지

        return self._current_exposure

    def _find_threshold_for_exposure(self, exposure: float) -> Optional[float]:
        """주어진 노출 비율에 해당하는 drawdown 임계값을 찾는다.

        Args:
            exposure: 노출 비율

        Returns:
            해당 drawdown 임계값, 없으면 None
        """
        for dd_threshold, exposure_ratio in self.thresholds:
            if abs(exposure_ratio - exposure) < 1e-9:
                return dd_threshold
        return None

    def _find_next_exposure_up(self, current_exposure: float) -> float:
        """현재 노출 비율보다 한 단계 위의 노출 비율을 반환한다.

        thresholds에 정의된 단계를 기준으로, 현재보다 한 단계 높은
        노출 비율을 찾는다. 가장 높은 단계를 초과하면 1.0을 반환한다.

        Args:
            current_exposure: 현재 노출 비율

        Returns:
            한 단계 위의 노출 비율
        """
        # thresholds는 drawdown 오름차순(가장 심한 먼저)으로 정렬
        # exposure는 가장 낮은 것부터 올라감
        # 현재 exposure 위치를 찾고 한 단계 위를 반환
        exposures = [exp for _, exp in self.thresholds]

        for i, exp in enumerate(exposures):
            if abs(exp - current_exposure) < 1e-9:
                # 다음 단계가 있으면 반환, 없으면 1.0
                if i + 1 < len(exposures):
                    return exposures[i + 1]
                else:
                    return 1.0

        # 매칭 안 되면 1.0 반환
        return 1.0

    def apply_overlay(self, weights: dict, portfolio_value: float) -> dict:
        """포트폴리오 비중에 드로다운 오버레이를 적용한다.

        update()를 호출하여 현재 노출 비율을 갱신하고,
        해당 비율만큼 포트폴리오 비중을 조절한다.

        Args:
            weights: {ticker: weight} 포트폴리오 비중 딕셔너리
            portfolio_value: 현재 포트폴리오 가치

        Returns:
            조절된 {ticker: weight} 딕셔너리.
            노출 비율이 1.0이면 원본 그대로, 0.0이면 빈 딕셔너리를 반환한다.
        """
        exposure = self.update(portfolio_value)

        if exposure >= 1.0:
            return weights

        if exposure <= 0.0:
            logger.info("드로다운 오버레이: 전량 현금 전환")
            return {}

        adjusted = {ticker: weight * exposure for ticker, weight in weights.items()}
        logger.info(
            f"드로다운 오버레이 비중 조절: exposure={exposure:.0%}, "
            f"종목 수={len(adjusted)}"
        )
        return adjusted
