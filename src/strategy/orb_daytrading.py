"""Opening Range Breakout (ORB) 데이트레이딩 전략.

장 시작 후 30분(09:00~09:30) 고가/저가 범위를 설정하고,
해당 범위를 돌파할 때 거래량 확인 후 진입하는 장중 단기 전략.

진입: 시가범위 상단 돌파(매수) 또는 하단 돌파(매도) + 거래량 확인
청산: 익절 1.5x 범위폭, 손절 반대쪽 범위, 15:20 강제청산
유효시간: 09:30~14:30 (신규 진입), 15:20까지 보유 가능
"""

from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ORBDaytradingStrategy(ShortTermStrategy):
    """Opening Range Breakout 데이트레이딩 전략."""

    name = "orb_daytrading"
    mode = "daytrading"

    DEFAULT_PARAMS = {
        "opening_range_minutes": 30,          # 09:00~09:30
        "volume_confirm_ratio": 1.5,          # 돌파 봉 거래량 vs 평균 거래량
        "profit_target_ratio": 1.5,           # 목표가 = 1.5x 범위폭
        "stop_loss_buffer_pct": 0.002,        # 손절 버퍼 0.2%
        "min_range_pct": 0.005,               # 최소 범위 0.5%
        "max_range_pct": 0.03,                # 최대 범위 3%
        "max_signals": 2,                     # 동시 최대 시그널 수
        "no_entry_after": "14:30",            # 신규 진입 마감 시각
        "force_close_time": "15:20",          # 강제 청산 시각
        "min_market_cap": 500_000_000_000,    # 최소 시가총액 5000억
    }

    # 장 시작 시각 (KST)
    _MARKET_OPEN = time(9, 0)

    def __init__(self, params: dict = None):
        """ORBDaytradingStrategy 초기화.

        Args:
            params: 파라미터 오버라이드 딕셔너리.
        """
        self._params = dict(self.DEFAULT_PARAMS)
        if params:
            self._params.update(params)

    @property
    def params(self) -> dict:
        return dict(self._params)

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        """ORB 시그널 스캔.

        market_data에 필요한 키:
        - "intraday_data": {ticker: DataFrame} -- 분봉 데이터
            columns: open, high, low, close, volume, timestamp
        - "daily_data": {ticker: DataFrame} -- 일봉 데이터 (시가총액 필터용)
        - "current_time": datetime (선택, 기본값=현재 시각)

        Returns:
            상위 max_signals개의 ShortTermSignal 리스트.
        """
        intraday_data = market_data.get("intraday_data", {})
        daily_data = market_data.get("daily_data", {})
        current_time = market_data.get("current_time", datetime.now())

        if not intraday_data:
            logger.warning("분봉 데이터 없음 -- ORB 스캔 스킵")
            return []

        # 시간 검증: 09:30 이전이면 아직 시가범위 미확정
        no_entry_after = self._parse_time(self._params["no_entry_after"])
        current_t = current_time.time() if isinstance(current_time, datetime) else current_time

        opening_range_end = self._get_opening_range_end()

        if current_t < opening_range_end:
            logger.info("ORB: 시가범위 확정 전 (현재 %s, 확정 %s) -- 스캔 대기",
                        current_t.strftime("%H:%M"), opening_range_end.strftime("%H:%M"))
            return []

        if current_t > no_entry_after:
            logger.info("ORB: 신규 진입 마감 시각 경과 (%s) -- 스캔 스킵",
                        self._params["no_entry_after"])
            return []

        candidates = []

        for ticker, df in intraday_data.items():
            if df is None or df.empty:
                continue

            try:
                result = self._evaluate_ticker(
                    ticker, df, daily_data.get(ticker), current_time
                )
                if result is not None:
                    candidates.append(result)
            except Exception as e:
                logger.debug("종목 %s ORB 평가 실패: %s", ticker, e)

        # 점수 정렬 후 상위 N개
        candidates.sort(key=lambda x: x["score"], reverse=True)
        max_signals = self._params["max_signals"]
        top = candidates[:max_signals]

        signals = []
        now_iso = current_time.isoformat() if isinstance(current_time, datetime) else ""
        force_close = self._params["force_close_time"]
        # 만료 시각: 당일 force_close_time
        if isinstance(current_time, datetime):
            fc_time = self._parse_time(force_close)
            expires_dt = current_time.replace(
                hour=fc_time.hour, minute=fc_time.minute, second=0, microsecond=0
            )
            expires_iso = expires_dt.isoformat()
        else:
            expires_iso = ""

        for c in top:
            sig = ShortTermSignal(
                id="",  # ShortTermTrader가 자동 생성
                ticker=c["ticker"],
                strategy=self.name,
                side=c["side"],
                mode=self.mode,
                confidence=c["score"],
                target_price=c["breakout_price"],
                stop_loss_price=c["stop_loss_price"],
                take_profit_price=c["take_profit_price"],
                reason=c["reason"],
                created_at=now_iso,
                expires_at=expires_iso,
                metadata={
                    "opening_range_high": c["or_high"],
                    "opening_range_low": c["or_low"],
                    "range_pct": c["range_pct"],
                    "volume_ratio": c["volume_ratio"],
                    "breakout_type": c["side"],
                    "avg_volume": c["avg_volume"],
                },
            )
            signals.append(sig)

        logger.info("ORB 스캔: %d 후보 중 %d 시그널 생성", len(candidates), len(signals))
        return signals

    def _evaluate_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        daily_df: Optional[pd.DataFrame],
        current_time: datetime,
    ) -> Optional[dict]:
        """개별 종목의 ORB 돌파를 평가한다.

        Returns:
            평가 결과 딕셔너리 또는 None (조건 미달).
        """
        # 시가총액 필터
        if daily_df is not None and not daily_df.empty:
            latest_daily = daily_df.iloc[-1]
            market_cap = float(
                latest_daily.get("시가총액", latest_daily.get("market_cap", float("inf")))
            )
            if market_cap < self._params["min_market_cap"]:
                return None

        # 1. 시가범위 계산
        opening_minutes = self._params["opening_range_minutes"]
        or_high, or_low, avg_volume = self._get_opening_range(df, opening_minutes)

        if or_high is None or or_low is None:
            return None

        if or_low <= 0:
            return None

        # 2. 범위 비율 검증
        range_pct = (or_high - or_low) / or_low
        if range_pct < self._params["min_range_pct"]:
            return None
        if range_pct > self._params["max_range_pct"]:
            return None

        # 3. 시가범위 이후 데이터에서 현재가 및 거래량 확인
        current_price, current_volume = self._get_current_bar(df, opening_minutes)
        if current_price is None or current_volume is None:
            return None
        if current_price <= 0 or current_volume <= 0:
            return None

        # 4. 돌파 방향 확인
        side = None
        breakout_price = current_price
        range_width = or_high - or_low

        if current_price > or_high:
            side = "buy"
        elif current_price < or_low:
            side = "sell"
        else:
            return None  # 범위 내 — 돌파 없음

        # 5. 거래량 확인
        if avg_volume <= 0:
            return None
        volume_ratio = current_volume / avg_volume
        if volume_ratio < self._params["volume_confirm_ratio"]:
            return None

        # 6. 목표가/손절가 계산
        buffer = self._params["stop_loss_buffer_pct"]

        if side == "buy":
            take_profit_price = current_price + range_width * self._params["profit_target_ratio"]
            stop_loss_price = or_low * (1 - buffer)
        else:  # sell
            take_profit_price = current_price - range_width * self._params["profit_target_ratio"]
            stop_loss_price = or_high * (1 + buffer)

        # 7. 종합 점수 계산 (0.0 ~ 1.0)
        score = self._calculate_breakout_score(
            side=side,
            current_price=current_price,
            or_high=or_high,
            or_low=or_low,
            volume_ratio=volume_ratio,
            range_pct=range_pct,
            current_time=current_time,
        )

        # 사유 문자열 생성
        breakout_dir = "상향" if side == "buy" else "하향"
        reason = (
            f"ORB {breakout_dir} 돌파: "
            f"범위={or_low:,.0f}~{or_high:,.0f} ({range_pct:.2%}), "
            f"현재가={current_price:,.0f}, "
            f"거래량={volume_ratio:.1f}배"
        )

        return {
            "ticker": ticker,
            "side": side,
            "score": score,
            "breakout_price": breakout_price,
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
            "or_high": or_high,
            "or_low": or_low,
            "range_pct": range_pct,
            "volume_ratio": volume_ratio,
            "avg_volume": avg_volume,
            "reason": reason,
        }

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        """보유 포지션 청산 여부 확인.

        Args:
            position: {
                "ticker": str,
                "entry_price": float,
                "current_price": float,
                "entry_date": str,
                "metadata": {
                    "opening_range_high": float,
                    "opening_range_low": float,
                }
            }
            market_data: {"current_time": datetime} (선택)

        Returns:
            청산 시그널 또는 None.
        """
        ticker = position.get("ticker", "")
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)
        metadata = position.get("metadata", {})

        if entry_price <= 0 or current_price <= 0:
            return None

        or_high = metadata.get("opening_range_high", 0)
        or_low = metadata.get("opening_range_low", 0)

        pnl_pct = (current_price - entry_price) / entry_price
        reasons = []
        exit_confidence = 0.0

        # 1. 손절 체크
        stop_loss_price = position.get("stop_loss_price", 0)
        if stop_loss_price > 0:
            # 매수 포지션: 현재가가 손절가 이하
            if entry_price <= current_price or True:  # 항상 체크
                if position.get("side", "buy") == "buy" and current_price <= stop_loss_price:
                    reasons.append(f"손절: {pnl_pct:.2%} (손절가 {stop_loss_price:,.0f})")
                    exit_confidence = 1.0
                elif position.get("side", "sell") == "sell" and current_price >= stop_loss_price:
                    reasons.append(f"손절: {pnl_pct:.2%} (손절가 {stop_loss_price:,.0f})")
                    exit_confidence = 1.0

        # 손절가가 position에 없으면 시가범위 기반으로 체크
        if not reasons and or_high > 0 and or_low > 0:
            buffer = self._params["stop_loss_buffer_pct"]
            side = position.get("side", metadata.get("breakout_type", "buy"))
            if side == "buy" and current_price <= or_low * (1 - buffer):
                reasons.append(f"손절: 시가범위 하단 이탈 ({current_price:,.0f} < {or_low:,.0f})")
                exit_confidence = 1.0
            elif side == "sell" and current_price >= or_high * (1 + buffer):
                reasons.append(f"손절: 시가범위 상단 이탈 ({current_price:,.0f} > {or_high:,.0f})")
                exit_confidence = 1.0

        # 2. 익절 체크
        take_profit_price = position.get("take_profit_price", 0)
        if take_profit_price > 0:
            side = position.get("side", metadata.get("breakout_type", "buy"))
            if side == "buy" and current_price >= take_profit_price:
                reasons.append(f"익절: {pnl_pct:.2%} (목표가 {take_profit_price:,.0f})")
                exit_confidence = max(exit_confidence, 0.95)
            elif side == "sell" and current_price <= take_profit_price:
                reasons.append(f"익절: {pnl_pct:.2%} (목표가 {take_profit_price:,.0f})")
                exit_confidence = max(exit_confidence, 0.95)

        # 3. 강제 청산 시각 체크
        current_time = market_data.get("current_time", datetime.now())
        force_close = self._parse_time(self._params["force_close_time"])
        current_t = current_time.time() if isinstance(current_time, datetime) else current_time

        if current_t >= force_close:
            reasons.append(f"강제청산: {self._params['force_close_time']} 도래 (수익률 {pnl_pct:.2%})")
            exit_confidence = max(exit_confidence, 1.0)

        if not reasons:
            return None

        return ShortTermSignal(
            id="",
            ticker=ticker,
            strategy=self.name,
            side="sell" if position.get("side", "buy") == "buy" else "buy",
            mode=self.mode,
            confidence=exit_confidence,
            target_price=current_price,
            reason=", ".join(reasons),
            metadata={
                "exit_pnl_pct": pnl_pct,
                "opening_range_high": or_high,
                "opening_range_low": or_low,
            },
        )

    # ──────────────────────────────────────────────
    # 헬퍼 메서드
    # ──────────────────────────────────────────────

    def _get_opening_range(
        self, df: pd.DataFrame, minutes: int
    ) -> tuple[Optional[float], Optional[float], float]:
        """시가범위(opening range)를 계산한다.

        장 시작 후 `minutes`분간의 고가/저가/평균거래량을 반환한다.
        DataFrame에 'timestamp' 컬럼이 있으면 시간 기반으로,
        없으면 인덱스 순서 기반으로 상위 `minutes`행을 사용한다.

        Args:
            df: 분봉 DataFrame (columns: open, high, low, close, volume, timestamp).
            minutes: 시가범위 산출 분 수 (기본 30).

        Returns:
            (opening_range_high, opening_range_low, avg_volume) 튜플.
            데이터 불충분 시 (None, None, 0.0).
        """
        if df.empty:
            return None, None, 0.0

        # timestamp 컬럼 기반 필터링 시도
        if "timestamp" in df.columns:
            or_bars = self._filter_opening_range_bars(df, minutes)
        else:
            # 인덱스가 datetime이면 활용
            if isinstance(df.index, pd.DatetimeIndex):
                or_bars = self._filter_opening_range_bars_by_index(df, minutes)
            else:
                # 단순 행 수 기반: 앞에서 minutes개 행
                or_bars = df.head(minutes)

        if or_bars.empty:
            return None, None, 0.0

        high_col = "high" if "high" in or_bars.columns else "고가"
        low_col = "low" if "low" in or_bars.columns else "저가"
        vol_col = "volume" if "volume" in or_bars.columns else "거래량"

        if high_col not in or_bars.columns or low_col not in or_bars.columns:
            return None, None, 0.0

        or_high = float(or_bars[high_col].max())
        or_low = float(or_bars[low_col].min())

        avg_volume = 0.0
        if vol_col in or_bars.columns:
            avg_volume = float(or_bars[vol_col].mean())

        return or_high, or_low, avg_volume

    def _filter_opening_range_bars(
        self, df: pd.DataFrame, minutes: int
    ) -> pd.DataFrame:
        """timestamp 컬럼 기반으로 시가범위 봉을 필터링한다."""
        ts = pd.to_datetime(df["timestamp"])
        market_open = self._MARKET_OPEN
        opening_end = (
            datetime.combine(datetime.today(), market_open) + timedelta(minutes=minutes)
        ).time()

        times = ts.dt.time
        mask = (times >= market_open) & (times < opening_end)
        return df.loc[mask]

    def _filter_opening_range_bars_by_index(
        self, df: pd.DataFrame, minutes: int
    ) -> pd.DataFrame:
        """DatetimeIndex 기반으로 시가범위 봉을 필터링한다."""
        market_open = self._MARKET_OPEN
        opening_end = (
            datetime.combine(datetime.today(), market_open) + timedelta(minutes=minutes)
        ).time()

        times = df.index.time
        mask = (times >= market_open) & (times < opening_end)
        return df.loc[mask]

    def _get_current_bar(
        self, df: pd.DataFrame, opening_minutes: int
    ) -> tuple[Optional[float], Optional[float]]:
        """시가범위 이후 최신 봉의 종가와 거래량을 반환한다.

        Args:
            df: 분봉 DataFrame.
            opening_minutes: 시가범위 분 수.

        Returns:
            (current_price, current_volume) 튜플. 데이터 없으면 (None, None).
        """
        if df.empty:
            return None, None

        close_col = "close" if "close" in df.columns else "종가"
        vol_col = "volume" if "volume" in df.columns else "거래량"

        if close_col not in df.columns:
            return None, None

        # 시가범위 이후 데이터
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            opening_end = (
                datetime.combine(datetime.today(), self._MARKET_OPEN)
                + timedelta(minutes=opening_minutes)
            ).time()
            times = ts.dt.time
            post_or = df.loc[times >= opening_end]
        elif isinstance(df.index, pd.DatetimeIndex):
            opening_end = (
                datetime.combine(datetime.today(), self._MARKET_OPEN)
                + timedelta(minutes=opening_minutes)
            ).time()
            times = df.index.time
            post_or = df.loc[times >= opening_end]
        else:
            # 행 수 기반: opening_minutes 이후 행
            post_or = df.iloc[opening_minutes:]

        if post_or.empty:
            return None, None

        latest = post_or.iloc[-1]
        current_price = float(latest[close_col])
        current_volume = float(latest.get(vol_col, 0)) if vol_col in post_or.columns else 0.0

        return current_price, current_volume

    def _calculate_breakout_score(
        self,
        side: str,
        current_price: float,
        or_high: float,
        or_low: float,
        volume_ratio: float,
        range_pct: float,
        current_time: datetime,
    ) -> float:
        """돌파 시그널의 종합 점수를 계산한다.

        점수 구성 (합계 1.0):
        - 돌파 강도 (25%): 시가범위 대비 돌파 거리
        - 거래량 비율 (35%): 평균 대비 거래량 배수
        - 범위 품질 (20%): 적절한 범위 폭일수록 높은 점수
        - 시간 요인 (20%): 이른 시간 돌파일수록 높은 점수

        Args:
            side: "buy" 또는 "sell".
            current_price: 현재가.
            or_high: 시가범위 고가.
            or_low: 시가범위 저가.
            volume_ratio: 현재 거래량 / 평균 거래량.
            range_pct: 시가범위 비율.
            current_time: 현재 시각.

        Returns:
            0.0 ~ 1.0 사이의 종합 점수.
        """
        range_width = or_high - or_low
        if range_width <= 0:
            return 0.0

        # 1. 돌파 강도 (25%) — 범위폭 대비 얼마나 멀리 돌파했는지
        if side == "buy":
            breakout_distance = current_price - or_high
        else:
            breakout_distance = or_low - current_price

        # 0~1로 정규화 (범위폭의 0~100% 돌파 = 0~1)
        breakout_strength = min(max(breakout_distance / range_width, 0.0), 1.0)
        score_breakout = breakout_strength * 0.25

        # 2. 거래량 비율 (35%) — 1.5배(최소) ~ 5배 이상 = 만점
        vol_confirm = self._params["volume_confirm_ratio"]
        vol_score_raw = min((volume_ratio - vol_confirm) / (5.0 - vol_confirm), 1.0)
        vol_score_raw = max(vol_score_raw, 0.0)
        score_volume = vol_score_raw * 0.35

        # 3. 범위 품질 (20%) — 1%~2% 범위가 최적
        min_r = self._params["min_range_pct"]
        max_r = self._params["max_range_pct"]
        optimal_low = 0.01
        optimal_high = 0.02

        if optimal_low <= range_pct <= optimal_high:
            range_quality = 1.0
        elif range_pct < optimal_low:
            range_quality = max((range_pct - min_r) / (optimal_low - min_r), 0.0)
        else:
            range_quality = max((max_r - range_pct) / (max_r - optimal_high), 0.0)
        score_range = range_quality * 0.20

        # 4. 시간 요인 (20%) — 09:30 = 1.0, 14:30 = 0.0 (이른 돌파일수록 유리)
        current_t = current_time.time() if isinstance(current_time, datetime) else current_time
        opening_end = self._get_opening_range_end()
        no_entry_after = self._parse_time(self._params["no_entry_after"])

        # 분 단위로 변환
        current_minutes = current_t.hour * 60 + current_t.minute
        start_minutes = opening_end.hour * 60 + opening_end.minute
        end_minutes = no_entry_after.hour * 60 + no_entry_after.minute

        total_window = end_minutes - start_minutes
        if total_window > 0:
            elapsed = current_minutes - start_minutes
            time_factor = max(1.0 - elapsed / total_window, 0.0)
        else:
            time_factor = 0.5
        score_time = time_factor * 0.20

        total_score = score_breakout + score_volume + score_range + score_time
        return round(min(max(total_score, 0.0), 1.0), 4)

    def _get_opening_range_end(self) -> time:
        """시가범위 종료 시각을 반환한다."""
        minutes = self._params["opening_range_minutes"]
        end_dt = datetime.combine(datetime.today(), self._MARKET_OPEN) + timedelta(minutes=minutes)
        return end_dt.time()

    @staticmethod
    def _parse_time(time_str: str) -> time:
        """'HH:MM' 문자열을 time 객체로 변환한다."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))
