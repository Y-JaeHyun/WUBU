"""퀀트 트레이딩 봇 메인 스케줄러.

APScheduler를 사용하여 장 전/장중/장 후 작업을 자동 실행한다.
systemd 서비스로 등록하여 무인 운영할 수 있다.

스케줄:
    07:00 - 모닝 브리핑 (마켓 요약 + 포트폴리오 현황)
    08:00 - 헬스 체크
    08:50 - 장 전 시그널 체크 (리밸런싱 대상일 판별)
    09:05 - 리밸런싱 실행 (해당일에만)
    매시 정각(09~15) - 포트폴리오 모니터링
    15:35 - 장 마감 후 일일 리뷰
    16:00 - 종목 리뷰 [stock_review]
    19:00 - 이브닝 종합 리포트
    22:00 - 야간 리서치 [night_research]
    매시(22~06) - 글로벌 시장 모니터 [global_monitor]
    일요일 10:00 - 자동 백테스트 [auto_backtest]
"""

from __future__ import annotations

import signal
import sys
import traceback
from datetime import datetime, timedelta
from typing import Optional

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from src.alert.alert_manager import AlertManager
from src.alert.conditions import (
    DailyMoveCondition,
    DisclosureAlertCondition,
    MarketCrashCondition,
    MddThresholdCondition,
    PriceShockCondition,
    RebalanceAlertCondition,
)
from src.alert.telegram_bot import TelegramNotifier
from src.alert.telegram_commander import TelegramCommander
from src.backtest.auto_runner import AutoBacktester
from src.data.cache import DataCache
from src.data.collector import get_all_fundamentals, get_price_data
from src.data.global_collector import get_global_snapshot, format_global_snapshot
from src.data.index_collector import get_index_data
from src.execution.executor import RebalanceExecutor
from src.execution.kis_client import KISClient
from src.execution.portfolio_allocator import PortfolioAllocator
from src.execution.risk_guard import RiskGuard
from src.execution.short_term_risk import ShortTermRiskConfig, ShortTermRiskManager
from src.execution.short_term_trader import ShortTermTrader
from src.report.night_research import NightResearcher
from src.strategy.swing_reversion import SwingReversionStrategy
from src.strategy.orb_daytrading import ORBDaytradingStrategy
from src.strategy.high_breakout import HighBreakoutStrategy
from src.strategy.bb_squeeze import BBSqueezeStrategy
from src.report.portfolio_tracker import PortfolioTracker
from src.report.stock_reviewer import StockReviewer
from src.scheduler.holidays import KRXHolidays
from src.utils.feature_flags import FeatureFlags
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

KST = pytz.timezone("Asia/Seoul")


class TradingBot:
    """트레이딩 봇 메인 클래스.

    APScheduler를 통해 정해진 시각에 작업을 실행하고,
    KRXHolidays로 비거래일을 자동 스킵한다.

    Attributes:
        scheduler: APScheduler BlockingScheduler.
        holidays: KRX 휴장일 관리 객체.
        kis_client: KIS OpenAPI 클라이언트.
        executor: 리밸런싱 실행기.
        alert_manager: 알림 관리기.
        notifier: 텔레그램 알림 발송기.
        rebalance_freq: 리밸런싱 주기 ("monthly" 또는 "quarterly").
    """

    def __init__(self, rebalance_freq: str = "monthly") -> None:
        """TradingBot을 초기화한다.

        Args:
            rebalance_freq: 리밸런싱 주기. 기본 "monthly".
        """
        self.scheduler: BlockingScheduler = BlockingScheduler(timezone=KST)
        self.holidays: KRXHolidays = KRXHolidays()
        self.rebalance_freq: str = rebalance_freq

        # KIS 클라이언트 및 실행기 (모드별 리스크 한도 자동 적용)
        self.kis_client: KISClient = KISClient()
        self.risk_guard: RiskGuard = RiskGuard(
            is_live=not self.kis_client.is_paper
        )
        self.executor: RebalanceExecutor = RebalanceExecutor(
            self.kis_client, self.risk_guard
        )

        # 포트폴리오 추적
        self.portfolio_tracker: PortfolioTracker = PortfolioTracker()

        # 알림
        self.notifier: TelegramNotifier = TelegramNotifier()
        self.alert_manager: AlertManager = AlertManager()
        self.alert_manager.add_notifier(self.notifier)

        # 알림 조건 등록
        self.alert_manager.add_condition(MddThresholdCondition(threshold=-0.15))
        self.alert_manager.add_condition(DailyMoveCondition(threshold=0.05))
        self.alert_manager.add_condition(RebalanceAlertCondition(days_before=3))
        self.alert_manager.add_condition(DisclosureAlertCondition())
        self.alert_manager.add_condition(PriceShockCondition())
        self.alert_manager.add_condition(MarketCrashCondition())

        # Phase 5-A: Feature Flags + 24시간 기능 확장
        self.feature_flags: FeatureFlags = FeatureFlags()
        cache_cfg = self.feature_flags.get_config("data_cache")
        self.data_cache: DataCache = DataCache(
            ttl_hours=cache_cfg.get("cache_ttl_hours", 24),
            max_size_mb=cache_cfg.get("max_cache_size_mb", 500),
        )
        self.commander: TelegramCommander = TelegramCommander(
            self.notifier, self.feature_flags
        )
        self.commander.register_command("/portfolio", self._cmd_portfolio)
        self.commander.register_command("/balance", self._cmd_balance)
        self.stock_reviewer: StockReviewer = StockReviewer()
        self.night_researcher: NightResearcher = NightResearcher()

        # 오버레이 (feature flag 기반 초기화)
        self._drawdown_overlay = None
        self._vol_targeting_overlay = None

        if self.feature_flags.is_enabled("drawdown_overlay"):
            from src.strategy.drawdown_overlay import DrawdownOverlay

            dd_cfg = self.feature_flags.get_config("drawdown_overlay")
            thresholds = [
                (t[0], t[1])
                for t in dd_cfg.get(
                    "thresholds",
                    [[-0.10, 0.75], [-0.15, 0.50], [-0.20, 0.25]],
                )
            ]
            self._drawdown_overlay = DrawdownOverlay(
                thresholds=thresholds,
                recovery_buffer=dd_cfg.get("recovery_buffer", 0.02),
            )

        if self.feature_flags.is_enabled("vol_targeting"):
            from src.strategy.vol_targeting import VolTargetingOverlay

            vt_cfg = self.feature_flags.get_config("vol_targeting")
            self._vol_targeting_overlay = VolTargetingOverlay(
                target_vol=vt_cfg.get("target_vol", 0.15),
                lookback_days=vt_cfg.get("lookback_days", 20),
                use_downside_only=vt_cfg.get("use_downside_only", True),
            )

        # 전략 (외부에서 주입 가능)
        self._strategy = None

        # 시그널 캐시 (EOD 뉴스에서 관심종목 참조용)
        self._last_long_signals: dict[str, float] = {}
        self._last_etf_signals: dict[str, float] = {}

        # Phase 5-B: 단기 트레이딩 모듈 초기화
        self._init_short_term_modules()

        logger.info(
            "TradingBot 초기화 완료 (리밸런싱 주기: %s)", rebalance_freq
        )

    def set_strategy(self, strategy: object) -> None:
        """전략 객체를 설정한다.

        Args:
            strategy: generate_signals(date, data) 메서드를 가진 전략 객체.
        """
        self._strategy = strategy
        strategy_name = getattr(strategy, "name", type(strategy).__name__)
        logger.info("전략 설정: %s", strategy_name)

    # ──────────────────────────────────────────────────────────
    # 전략 데이터 수집
    # ──────────────────────────────────────────────────────────

    def _collect_strategy_data(self, date_str: str) -> dict:
        """전략 시그널 생성에 필요한 데이터를 수집한다.

        백테스트 엔진(engine.py:178-193)과 동일한 구조를 반환한다.
        DataCache를 통해 동일 기준일 재호출 시 pykrx API를 생략한다.

        Args:
            date_str: 데이터 기준일 ('YYYYMMDD').

        Returns:
            {"fundamentals": DataFrame, "prices": dict, "index_prices": Series}
        """
        import pandas as pd

        data: dict = {
            "fundamentals": pd.DataFrame(),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }

        use_cache = self.feature_flags.is_enabled("data_cache")

        # 1. Fundamentals ─────────────────────────────────
        cache_key_fund = DataCache.make_key("fundamentals", date_str)
        fundamentals = None
        if use_cache:
            fundamentals = self.data_cache.get(cache_key_fund)

        if fundamentals is not None:
            logger.info("펀더멘탈 캐시 히트: %d종목 (%s)", len(fundamentals), date_str)
        else:
            try:
                fundamentals = get_all_fundamentals(date_str)
                logger.info(
                    "펀더멘탈 수집 완료: %d종목 (%s)", len(fundamentals), date_str
                )
                if use_cache and not fundamentals.empty:
                    self.data_cache.put(cache_key_fund, fundamentals)
            except Exception as e:
                logger.error("펀더멘탈 수집 실패 (%s): %s", date_str, e)
                fundamentals = pd.DataFrame()

        data["fundamentals"] = fundamentals

        # 2. Price data (시총 상위 종목) ───────────────────
        if not fundamentals.empty:
            top_tickers = self._get_price_universe(fundamentals)
            start_dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=400)
            start_date = start_dt.strftime("%Y%m%d")

            prices: dict = {}
            for ticker in top_tickers:
                cache_key_p = DataCache.make_key("price", ticker, start_date, date_str)
                price_df = None
                if use_cache:
                    price_df = self.data_cache.get(cache_key_p)

                if price_df is not None:
                    prices[ticker] = price_df
                    continue

                try:
                    price_df = get_price_data(ticker, start_date, date_str)
                    if not price_df.empty:
                        prices[ticker] = price_df
                        if use_cache:
                            self.data_cache.put(cache_key_p, price_df)
                except Exception:
                    pass  # 개별 종목 실패는 무시

            data["prices"] = prices
            logger.info("가격 데이터 수집: %d/%d종목", len(prices), len(top_tickers))

        # 3. KOSPI 지수 ────────────────────────────────────
        try:
            cache_key_idx = DataCache.make_key("kospi_index", date_str)
            index_series = None
            if use_cache:
                idx_df = self.data_cache.get(cache_key_idx)
                if idx_df is not None and "close" in idx_df.columns:
                    index_series = idx_df["close"]

            if index_series is None:
                start_dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=400)
                index_df = get_index_data(
                    "KOSPI", start_dt.strftime("%Y%m%d"), date_str
                )
                if not index_df.empty and "close" in index_df.columns:
                    index_series = index_df["close"]
                    if use_cache:
                        self.data_cache.put(cache_key_idx, index_df)
                else:
                    index_series = pd.Series(dtype=float)

            data["index_prices"] = index_series
            logger.info("KOSPI 지수 수집 완료: %d일", len(index_series))
        except Exception as e:
            logger.warning("KOSPI 지수 수집 실패: %s", e)

        return data

    @staticmethod
    def _get_price_universe(
        fundamentals, max_stocks: int = 200,
    ) -> list[str]:
        """시가총액 상위 종목을 가격 수집 대상으로 선정한다."""
        if fundamentals.empty or "market_cap" not in fundamentals.columns:
            return []
        df = fundamentals[fundamentals["market_cap"] > 0].copy()
        df = df.sort_values("market_cap", ascending=False).head(max_stocks)
        return df["ticker"].tolist()

    def _collect_short_term_data(self, date_str: str) -> dict:
        """단기 전략 시그널 스캔에 필요한 데이터를 수집한다.

        장기 전략용 _collect_strategy_data()와 분리된 경량 버전.
        시총 상위 100종목의 260일 OHLCV만 수집한다.

        Args:
            date_str: 데이터 기준일 ('YYYYMMDD').

        Returns:
            {"daily_data": {ticker: DataFrame}, "date": date_str}
        """
        import pandas as pd

        use_cache = self.feature_flags.is_enabled("data_cache")

        # 1. Fundamentals (시총 순위용)
        cache_key_fund = DataCache.make_key("fundamentals", date_str)
        fundamentals = None
        if use_cache:
            fundamentals = self.data_cache.get(cache_key_fund)

        if fundamentals is None:
            try:
                fundamentals = get_all_fundamentals(date_str)
                if use_cache and not fundamentals.empty:
                    self.data_cache.put(cache_key_fund, fundamentals)
            except Exception as e:
                logger.error("단기 펀더멘탈 수집 실패 (%s): %s", date_str, e)
                fundamentals = pd.DataFrame()

        if fundamentals.empty:
            return {"daily_data": {}, "date": date_str}

        # 2. 시총 상위 100종목 260일 OHLCV
        top_tickers = self._get_price_universe(fundamentals, max_stocks=100)
        start_dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=260)
        start_date = start_dt.strftime("%Y%m%d")

        daily_data: dict = {}
        for ticker in top_tickers:
            cache_key_p = DataCache.make_key("price", ticker, start_date, date_str)
            price_df = None
            if use_cache:
                price_df = self.data_cache.get(cache_key_p)

            if price_df is not None:
                daily_data[ticker] = price_df
                continue

            try:
                price_df = get_price_data(ticker, start_date, date_str)
                if not price_df.empty:
                    daily_data[ticker] = price_df
                    if use_cache:
                        self.data_cache.put(cache_key_p, price_df)
            except Exception:
                pass

        logger.info("단기 데이터 수집: %d/%d종목 (%s)", len(daily_data), len(top_tickers), date_str)
        return {"daily_data": daily_data, "date": date_str}

    def _init_short_term_modules(self) -> None:
        """포트폴리오 할당 및 단기 트레이딩 모듈을 초기화한다.

        etf_rotation 또는 short_term_trading 중 하나라도 활성화되면
        PortfolioAllocator를 생성하여 풀 비율을 관리한다.
        """
        self.allocator = None
        self.short_term_trader = None
        self.short_term_risk = None

        etf_enabled = self.feature_flags.is_enabled("etf_rotation")
        short_enabled = self.feature_flags.is_enabled("short_term_trading")

        if not etf_enabled and not short_enabled:
            logger.info("ETF 로테이션/단기 트레이딩 모두 비활성화")
            return

        try:
            # 풀 비율 동적 계산
            etf_pct = 0.0
            short_pct = 0.0

            if etf_enabled:
                etf_config = self.feature_flags.get_config("etf_rotation")
                etf_pct = etf_config.get("etf_rotation_pct", 0.30)

            if short_enabled:
                short_config = self.feature_flags.get_config("short_term_trading")
                short_pct = short_config.get("short_term_pct", 0.05)

            long_pct = 1.0 - etf_pct - short_pct

            if long_pct < 0:
                logger.error(
                    "풀 비율 합이 100%%를 초과합니다: "
                    "장기=%.0f%%, ETF=%.0f%%, 단기=%.0f%%",
                    long_pct * 100, etf_pct * 100, short_pct * 100,
                )
                return

            # PortfolioAllocator
            self.allocator = PortfolioAllocator(
                kis_client=self.kis_client,
                long_term_pct=long_pct,
                short_term_pct=short_pct,
                etf_rotation_pct=etf_pct,
            )

            # Executor에 allocator 주입
            self.executor = RebalanceExecutor(
                self.kis_client, self.risk_guard, allocator=self.allocator
            )

            logger.info(
                "포트폴리오 할당 초기화: 장기=%.0f%%, ETF=%.0f%%, 단기=%.0f%%",
                long_pct * 100, etf_pct * 100, short_pct * 100,
            )

            # 단기 트레이딩 모듈 (기존 로직 유지)
            if short_enabled:
                config = self.feature_flags.get_config("short_term_trading")

                risk_config = ShortTermRiskConfig(
                    stop_loss_pct=config.get("stop_loss_pct", -0.05),
                    take_profit_pct=config.get("take_profit_pct", 0.10),
                    max_concurrent_positions=config.get(
                        "max_concurrent_positions", 3
                    ),
                    max_daily_loss_pct=config.get("max_daily_loss_pct", -0.03),
                )
                self.short_term_risk = ShortTermRiskManager(config=risk_config)

                mode = config.get("mode", "swing")
                self.short_term_trader = ShortTermTrader(
                    allocator=self.allocator,
                    risk_manager=self.short_term_risk,
                    order_manager=self.executor.order_manager,
                    confirm_timeout_minutes=config.get(
                        "confirm_timeout_minutes", 30
                    ),
                    mode=mode,
                )

                strategy_name = config.get("strategy", "bb_squeeze")
                swing_strategy = self._create_swing_strategy(strategy_name)
                self.short_term_trader.register_strategy(swing_strategy)

                orb = ORBDaytradingStrategy()
                self.short_term_trader.register_strategy(orb)

                self.commander = TelegramCommander(
                    self.notifier, self.feature_flags,
                    short_term_trader=self.short_term_trader,
                    short_term_risk=self.short_term_risk,
                )
                self.commander.register_command("/portfolio", self._cmd_portfolio)
                self.commander.register_command("/balance", self._cmd_balance)

                logger.info("단기 트레이딩 모듈 초기화 완료 (모드=%s)", mode)

        except Exception as e:
            logger.error("포트폴리오 할당 초기화 실패: %s", e)
            self.allocator = None
            self.short_term_trader = None
            self.short_term_risk = None

    @staticmethod
    def _create_swing_strategy(name: str):
        """전략 이름으로 스윙 전략 인스턴스를 생성한다.

        Args:
            name: 전략 이름 (swing_reversion, swing_reversion_obv,
                  high_breakout, bb_squeeze).

        Returns:
            ShortTermStrategy 인스턴스.
        """
        strategies = {
            "swing_reversion": lambda: SwingReversionStrategy(),
            "swing_reversion_obv": lambda: SwingReversionStrategy(
                params={"use_obv_filter": True}
            ),
            "high_breakout": lambda: HighBreakoutStrategy(),
            "bb_squeeze": lambda: BBSqueezeStrategy(),
        }
        factory = strategies.get(name)
        if factory is None:
            logger.warning(
                "알 수 없는 전략: %s. 기본값(bb_squeeze) 사용.", name
            )
            return BBSqueezeStrategy()
        return factory()

    def _is_trading_day(self) -> bool:
        """오늘이 거래일인지 확인한다.

        Returns:
            거래일이면 True, 비거래일이면 False.
        """
        today = datetime.now(KST).date()
        is_trading = self.holidays.is_trading_day(today)

        if not is_trading:
            reason = "주말" if self.holidays.is_weekend(today) else "공휴일"
            logger.info("오늘(%s)은 %s입니다. 작업을 스킵합니다.", today, reason)

        return is_trading

    def _send_notification(self, message: str, level: str = "INFO") -> None:
        """알림을 발송한다.

        메시지 앞에 트레이딩 모드 태그([모의]/[실전])를 자동 추가한다.

        Args:
            message: 발송할 메시지.
            level: 알림 수준 ("INFO", "WARNING", "CRITICAL").
        """
        try:
            tagged_message = f"{self.kis_client.mode_tag} {message}"
            self.alert_manager.send(tagged_message, level=level)
        except Exception as e:
            logger.error("알림 발송 실패: %s", e)

    # ──────────────────────────────────────────────────────────
    # 스케줄 작업
    # ──────────────────────────────────────────────────────────

    def morning_briefing(self) -> None:
        """07:00 - 마켓 브리핑 생성 및 텔레그램 발송.

        오늘의 거래일 여부, 리밸런싱 예정 여부,
        전일 포트폴리오 현황을 요약하여 발송한다.
        """
        if not self._is_trading_day():
            return

        try:
            today = datetime.now(KST).date()
            today_str = today.strftime("%Y-%m-%d")

            is_rebal_day = self.holidays.is_rebalance_day(
                today, self.rebalance_freq
            )
            days_to_rebal = self.holidays.days_to_next_rebalance(
                today, self.rebalance_freq
            )

            lines = [
                f"[모닝 브리핑] {today_str}",
                "─" * 30,
            ]

            if is_rebal_day:
                lines.append("** 오늘은 리밸런싱 예정일입니다 **")
            elif days_to_rebal <= 3:
                lines.append(f"리밸런싱까지 {days_to_rebal}일 남았습니다.")

            # 포트폴리오 현황 (KIS 설정 시)
            if self.kis_client.is_configured():
                try:
                    balance = self.kis_client.get_balance()
                    total_eval = balance.get("total_eval", 0)
                    cash = balance.get("cash", 0)
                    profit_pct = balance.get("total_profit_pct", 0.0)
                    holdings = balance.get("holdings", [])

                    lines.append("")
                    lines.append("[포트폴리오 현황]")
                    lines.append(f"  총 평가: {total_eval:,}원")
                    lines.append(f"  현금: {cash:,}원")
                    lines.append(f"  수익률: {profit_pct:+.2f}%")
                    mdd = self.portfolio_tracker.get_mdd()
                    lines.append(f"  MDD: {mdd:.2%}")
                    lines.append(f"  종목 수: {len(holdings)}개")
                except Exception as e:
                    lines.append(f"\n[포트폴리오 조회 실패: {e}]")
            else:
                lines.append("\nKIS API 미설정 상태")

            message = "\n".join(lines)
            self._send_notification(message)
            logger.info("모닝 브리핑 발송 완료")

        except Exception as e:
            logger.error("모닝 브리핑 생성 실패: %s", e)
            logger.debug(traceback.format_exc())

    def premarket_check(self) -> None:
        """08:50 - 장 전 시그널 체크.

        리밸런싱 대상일이면 전략 시그널을 미리 생성하여 알림한다.
        """
        if not self._is_trading_day():
            return

        try:
            today = datetime.now(KST).date()

            if not self.holidays.is_rebalance_day(today, self.rebalance_freq):
                logger.info("오늘은 리밸런싱일이 아닙니다.")
                return

            if self._strategy is None:
                logger.warning("전략이 설정되지 않았습니다. 시그널을 생성할 수 없습니다.")
                self._send_notification(
                    "[장 전 체크] 리밸런싱일이나 전략이 미설정 상태입니다.",
                    level="WARNING",
                )
                return

            # 전략 시그널 미리보기 (dry run)
            logger.info("리밸런싱 사전 시그널 체크 시작")

            # T-1 기준 데이터 수집 (08:50에는 당일 데이터 미완성)
            prev_day = self.holidays.prev_trading_day(today)
            data_date = prev_day.strftime("%Y%m%d")
            logger.info("전략 데이터 수집 (기준일: %s)", data_date)
            strategy_data = self._collect_strategy_data(data_date)

            if strategy_data["fundamentals"].empty:
                self._send_notification(
                    f"[장 전 체크] {today.strftime('%Y-%m-%d')}\n"
                    "펀더멘탈 데이터 수집 실패. 시그널 생성 불가.",
                    level="WARNING",
                )
                return

            date_str = today.strftime("%Y%m%d")
            try:
                signals = self._strategy.generate_signals(date_str, strategy_data)
            except Exception as e:
                logger.warning("시그널 생성 중 오류: %s", e)
                signals = {}

            # EOD 뉴스 관심종목용 시그널 캐시
            if signals:
                self._last_long_signals = dict(signals)

            if signals:
                top_tickers = sorted(
                    signals.items(), key=lambda x: x[1], reverse=True
                )[:10]
                lines = [
                    f"[장 전 시그널 체크] {today.strftime('%Y-%m-%d')}",
                    "─" * 30,
                    f"장기 목표 종목 수: {len(signals)}개",
                    "",
                    "Top 10 비중:",
                ]
                for ticker, weight in top_tickers:
                    lines.append(f"  {ticker}: {weight:.1%}")

                # 통합 Dry run (allocator 활성 시)
                if self.allocator:
                    pool_signals: dict[str, dict[str, float]] = {
                        "long_term": signals,
                    }
                    etf_signals = self._generate_etf_signals(date_str)
                    if etf_signals:
                        pool_signals["etf_rotation"] = etf_signals
                        self._last_etf_signals = dict(etf_signals)
                        lines.append(f"\nETF 종목 수: {len(etf_signals)}개")

                    dry_result = self.executor.dry_run_integrated(
                        pool_signals
                    )
                else:
                    dry_result = self.executor.dry_run(
                        signals, pool="long_term"
                    )

                lines.append("")
                lines.append(
                    f"예상 매도: {len(dry_result.get('sell_orders', []))}건"
                )
                lines.append(
                    f"예상 매수: {len(dry_result.get('buy_orders', []))}건"
                )

                risk_check = dry_result.get("risk_check", {})
                if not risk_check.get("passed", True):
                    lines.append("")
                    lines.append("** 리스크 경고 **")
                    for w in risk_check.get("warnings", []):
                        lines.append(f"  - {w}")

                message = "\n".join(lines)
                self._send_notification(message)
            else:
                self._send_notification(
                    f"[장 전 시그널 체크] {today.strftime('%Y-%m-%d')}\n"
                    "시그널이 없습니다."
                )

            logger.info("장 전 시그널 체크 완료")

        except Exception as e:
            logger.error("장 전 체크 실패: %s", e)
            logger.debug(traceback.format_exc())
            self._send_notification(
                f"[장 전 체크 오류] {e}", level="CRITICAL"
            )

    def _fetch_etf_prices(
        self,
        etf_universe: dict[str, str],
        lookback_days: int = 252,
    ) -> dict:
        """ETF 유니버스의 가격 데이터를 수집한다.

        Args:
            etf_universe: {ticker: name} 딕셔너리.
            lookback_days: 모멘텀 계산에 필요한 거래일 수.

        Returns:
            {ticker: DataFrame} 딕셔너리.
        """
        from src.data.etf_collector import get_etf_price

        end_date = datetime.now(KST).strftime("%Y%m%d")
        start_dt = datetime.now(KST) - timedelta(days=int(lookback_days * 1.4))
        start_date = start_dt.strftime("%Y%m%d")

        etf_prices: dict = {}
        for ticker in etf_universe:
            try:
                df = get_etf_price(ticker, start_date, end_date)
                if not df.empty:
                    etf_prices[ticker] = df
            except Exception as e:
                logger.warning("ETF 가격 수집 실패: %s - %s", ticker, e)

        logger.info(
            "ETF 가격 수집 완료: %d/%d종목", len(etf_prices), len(etf_universe)
        )
        return etf_prices

    def execute_etf_rotation_rebalance(self) -> None:
        """09:10 - ETF 로테이션 모멘텀 순위 모니터링 (진단 전용).

        실제 ETF 매매는 09:05 execute_rebalance()에서 통합 수행된다.
        이 메서드는 모멘텀 순위를 로깅하여 운영 모니터링을 돕는다.
        리밸런싱일/비 리밸런싱일 공통으로 순위만 기록한다.
        """
        if not self.feature_flags.is_enabled("etf_rotation"):
            return
        if not self._is_trading_day():
            return

        try:
            today = datetime.now(KST).date()
            config = self.feature_flags.get_config("etf_rotation")
            rebalance_freq = config.get("rebalance_freq", "monthly")

            if self.holidays.is_rebalance_day(today, rebalance_freq):
                logger.info(
                    "ETF 로테이션: 리밸런싱일 → 09:05 통합 실행 완료. "
                    "모멘텀 순위만 로깅."
                )

            # 모멘텀 순위 로깅 (리밸런싱일/비 리밸런싱일 공통)
            strategy = self._create_etf_strategy()

            lookback_months = config.get("lookback_months", 12)
            lookback_days = lookback_months * 21

            etf_prices = self._fetch_etf_prices(
                strategy.etf_universe, lookback_days
            )
            if not etf_prices:
                logger.info("ETF 모멘텀 순위: 가격 데이터 없음")
                return

            date_str = today.strftime("%Y%m%d")
            strategy.generate_signals(date_str, {"etf_prices": etf_prices})

            diag = strategy.last_diagnostics

            # Enhanced 전략은 composite_scores, 기존 전략은 per_ticker
            composite_scores = diag.get("composite_scores", {})
            if composite_scores:
                ranked = sorted(
                    composite_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                rank_str = ", ".join(
                    f"{strategy.etf_universe.get(t, t)}={m:+.2f}"
                    for t, m in ranked[:5]
                )
                regime = diag.get("market_regime", "N/A")
                logger.info(
                    "Enhanced ETF 모멘텀 순위: %s (regime=%s)",
                    rank_str, regime,
                )
            else:
                per_ticker = diag.get("per_ticker", {})
                ok_tickers = {
                    t: info["momentum"]
                    for t, info in per_ticker.items()
                    if info.get("status") == "OK"
                }
                if ok_tickers:
                    ranked = sorted(
                        ok_tickers.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    rank_str = ", ".join(
                        f"{strategy.etf_universe.get(t, t)}={m:+.1%}"
                        for t, m in ranked[:5]
                    )
                    logger.info("ETF 모멘텀 순위: %s", rank_str)

        except Exception as e:
            logger.warning("ETF 모멘텀 순위 로깅 실패: %s", e)

    def _apply_live_overlays(
        self, signals: dict[str, float]
    ) -> dict[str, float]:
        """라이브 리밸런싱에 drawdown/vol_targeting 오버레이를 적용한다.

        Args:
            signals: {ticker: weight} 포트폴리오 비중.

        Returns:
            오버레이 적용 후 {ticker: weight}.
        """
        if not signals:
            return signals

        # 1. 드로다운 오버레이
        if self._drawdown_overlay is not None:
            try:
                balance = self.kis_client.get_balance()
                pv = balance.get("total_eval", 0)
                if pv > 0:
                    signals = self._drawdown_overlay.apply_overlay(
                        signals, pv
                    )
                    logger.info(
                        "드로다운 오버레이 적용: %d종목", len(signals)
                    )
            except Exception as e:
                logger.warning("드로다운 오버레이 실패: %s", e)

        # 2. 변동성 타겟팅
        if self._vol_targeting_overlay is not None:
            try:
                pv_series = self.portfolio_tracker.get_values_series()
                if len(pv_series) > 0:
                    signals = self._vol_targeting_overlay.apply(
                        signals, pv_series
                    )
                    logger.info(
                        "변동성 타겟팅 적용: %d종목", len(signals)
                    )
            except Exception as e:
                logger.warning("변동성 타겟팅 실패: %s", e)

        # 최소 노출 하한 (10%) — 백테스트 엔진과 동일
        if signals:
            total_w = sum(signals.values())
            if 0 < total_w < 0.10:
                scale = 0.10 / total_w
                signals = {t: w * scale for t, w in signals.items()}
                logger.info(
                    "최소 노출 하한 적용: %.1f%% → 10%%",
                    total_w * 100,
                )

        return signals

    def execute_rebalance(self) -> None:
        """09:05 - 통합 리밸런싱 실행 (해당일만).

        장기 + ETF 시그널을 병합하여 단일 diff 기반으로 실행한다.
        allocator가 없으면 장기 단독 실행으로 폴백한다.
        """
        if not self._is_trading_day():
            return

        try:
            today = datetime.now(KST).date()

            if not self.holidays.is_rebalance_day(today, self.rebalance_freq):
                logger.info("오늘은 리밸런싱일이 아닙니다. 실행을 스킵합니다.")
                return

            if self._strategy is None:
                logger.error("전략이 설정되지 않았습니다. 리밸런싱을 실행할 수 없습니다.")
                self._send_notification(
                    "[리밸런싱 오류] 전략이 미설정 상태입니다.",
                    level="CRITICAL",
                )
                return

            if not self.kis_client.is_configured():
                logger.error("KIS API가 설정되지 않았습니다.")
                self._send_notification(
                    "[리밸런싱 오류] KIS API 미설정 상태입니다.",
                    level="CRITICAL",
                )
                return

            # T-1 데이터 수집 (08:50 캐시 히트 기대)
            prev_day = self.holidays.prev_trading_day(today)
            data_date = prev_day.strftime("%Y%m%d")
            logger.info("리밸런싱 데이터 수집 (기준일: %s)", data_date)
            strategy_data = self._collect_strategy_data(data_date)

            if strategy_data["fundamentals"].empty:
                logger.error("펀더멘탈 데이터 수집 실패. 리밸런싱을 중단합니다.")
                self._send_notification(
                    "[리밸런싱 오류] 펀더멘탈 데이터 수집 실패.",
                    level="CRITICAL",
                )
                return

            # 장기 시그널 생성
            date_str = today.strftime("%Y%m%d")
            logger.info("리밸런싱 시그널 생성 중: %s", date_str)

            try:
                long_signals = self._strategy.generate_signals(
                    date_str, strategy_data
                )
            except Exception as e:
                logger.error("장기 시그널 생성 실패: %s", e)
                self._send_notification(
                    f"[리밸런싱 오류] 장기 시그널 생성 실패: {e}",
                    level="CRITICAL",
                )
                return

            # 오버레이 적용 (drawdown + vol_targeting)
            if long_signals:
                long_signals = self._apply_live_overlays(long_signals)

            # EOD 뉴스 관심종목용 시그널 캐시
            if long_signals:
                self._last_long_signals = dict(long_signals)
            else:
                logger.warning("장기 시그널이 없습니다.")

            # 통합 실행 (allocator 활성 시)
            if self.allocator:
                # 미태깅 보유 종목 자동 분류 (첫 실행 마이그레이션)
                self.allocator.backfill_untagged_positions(
                    etf_tickers=self._get_etf_universe_tickers(),
                    default_pool="long_term",
                )

                pool_signals: dict[str, dict[str, float]] = {}
                if long_signals:
                    pool_signals["long_term"] = long_signals

                # ETF 시그널 생성 (ETF 로테이션 활성 + 리밸런싱일)
                etf_signals = self._generate_etf_signals(date_str)
                if etf_signals:
                    pool_signals["etf_rotation"] = etf_signals
                    self._last_etf_signals = dict(etf_signals)

                if not pool_signals:
                    logger.warning("시그널이 없습니다. 리밸런싱을 스킵합니다.")
                    self._send_notification(
                        "[리밸런싱] 시그널이 없어 리밸런싱을 스킵합니다."
                    )
                    return

                self._send_notification(
                    f"[통합 리밸런싱 시작] {today.strftime('%Y-%m-%d')}\n"
                    f"풀: {', '.join(pool_signals.keys())}\n"
                    f"장기 {len(long_signals or {})}종목"
                    + (f", ETF {len(etf_signals or {})}종목"
                       if etf_signals else "")
                )

                result = self.executor.execute_integrated_rebalance(
                    pool_signals
                )
            else:
                # allocator 미설정: 기존 단독 실행 폴백
                if not long_signals:
                    self._send_notification(
                        "[리밸런싱] 시그널이 없어 스킵합니다."
                    )
                    return

                self._send_notification(
                    f"[리밸런싱 시작] {today.strftime('%Y-%m-%d')}\n"
                    f"종목 수: {len(long_signals)}개"
                )
                result = self.executor.execute_rebalance(long_signals)

            # 결과 알림
            lines = [
                f"[리밸런싱 완료] {today.strftime('%Y-%m-%d')}",
                "─" * 30,
                f"성공: {'예' if result.get('success') else '아니오'}",
                f"매도: {len(result.get('sells', []))}건 "
                f"({result.get('total_sell_amount', 0):,}원)",
                f"매수: {len(result.get('buys', []))}건 "
                f"({result.get('total_buy_amount', 0):,}원)",
            ]

            errors = result.get("errors", [])
            if errors:
                lines.append("")
                lines.append(f"오류 {len(errors)}건:")
                for err in errors[:5]:
                    lines.append(f"  - {err}")

            skipped = result.get("skipped", [])
            if skipped:
                lines.append(f"스킵 {len(skipped)}건")

            level = "INFO" if result.get("success") else "CRITICAL"
            self._send_notification("\n".join(lines), level=level)

            logger.info("리밸런싱 실행 완료")

        except Exception as e:
            logger.error("리밸런싱 실행 중 예외 발생: %s", e)
            logger.debug(traceback.format_exc())
            self._send_notification(
                f"[리밸런싱 실행 오류] {e}", level="CRITICAL"
            )

    def _create_etf_strategy(self):
        """Feature flag에 따라 적절한 ETF 로테이션 전략을 생성한다.

        enhanced_etf_rotation flag가 ON이면 EnhancedETFRotationStrategy,
        OFF이면 기존 ETFRotationStrategy를 생성한다.
        etf_rotation config에서 공통 파라미터(lookback, n_select 등)를 읽고,
        enhanced_etf_rotation config에서 추가 파라미터를 읽는다.

        Returns:
            ETF 전략 객체 (ETFRotationStrategy 또는 EnhancedETFRotationStrategy).
        """
        config = self.feature_flags.get_config("etf_rotation")
        lookback_months = config.get("lookback_months", 12)
        lookback_days = lookback_months * 21
        n_select = config.get("n_select", 2)
        max_same_sector = config.get("max_same_sector", 1)

        use_enhanced = self.feature_flags.is_enabled("enhanced_etf_rotation")

        if use_enhanced:
            from src.strategy.enhanced_etf_rotation import (
                EnhancedETFRotationStrategy,
            )

            enh_config = self.feature_flags.get_config("enhanced_etf_rotation")

            strategy = EnhancedETFRotationStrategy(
                num_etfs=n_select,
                max_same_sector=max_same_sector,
                use_vol_weight=enh_config.get("use_vol_weight", True),
                use_market_filter=enh_config.get("use_market_filter", True),
                use_trend_filter=enh_config.get("use_trend_filter", True),
                max_drawdown_filter=(
                    enh_config.get("max_drawdown_filter", 0.15)
                    if enh_config.get("use_max_drawdown_filter", True)
                    else 0.0
                ),
                cash_ratio_risk_off=enh_config.get(
                    "cash_ratio_risk_off", 0.7
                ),
                vol_lookback=enh_config.get("vol_lookback", 60),
                market_ma_period=enh_config.get("market_ma_period", 200),
                trend_short_ma=enh_config.get("trend_short_ma", 20),
                trend_long_ma=enh_config.get("trend_long_ma", 60),
            )
            logger.info(
                "Enhanced ETF 전략 생성: num_etfs=%d, "
                "vol_weight=%s, market_filter=%s, trend_filter=%s, "
                "cash_ratio_risk_off=%.1f",
                n_select,
                enh_config.get("use_vol_weight", True),
                enh_config.get("use_market_filter", True),
                enh_config.get("use_trend_filter", True),
                enh_config.get("cash_ratio_risk_off", 0.7),
            )
        else:
            from src.strategy.etf_rotation import ETFRotationStrategy

            strategy = ETFRotationStrategy(
                lookback=lookback_days,
                num_etfs=n_select,
                max_same_sector=max_same_sector,
            )

        return strategy

    def _generate_etf_signals(self, date_str: str) -> dict[str, float]:
        """ETF 로테이션 시그널을 생성한다.

        Args:
            date_str: 기준 날짜 ('YYYYMMDD').

        Returns:
            {ticker: weight} 시그널 딕셔너리. 비활성/실패 시 빈 딕셔너리.
        """
        if not self.feature_flags.is_enabled("etf_rotation"):
            return {}

        try:
            strategy = self._create_etf_strategy()

            config = self.feature_flags.get_config("etf_rotation")
            lookback_months = config.get("lookback_months", 12)
            lookback_days = lookback_months * 21

            etf_prices = self._fetch_etf_prices(
                strategy.etf_universe, lookback_days
            )
            if not etf_prices:
                logger.warning("ETF 가격 데이터 없음. ETF 시그널 스킵.")
                return {}

            signals = strategy.generate_signals(
                date_str, {"etf_prices": etf_prices}
            )
            return signals or {}

        except Exception as e:
            logger.warning("ETF 시그널 생성 실패: %s", e)
            return {}

    def _get_etf_universe_tickers(self) -> set[str]:
        """ETF 로테이션 유니버스의 ticker 집합을 반환한다."""
        try:
            strategy = self._create_etf_strategy()
            return set(strategy.etf_universe.keys())
        except Exception:
            return set()

    def hourly_monitor(self) -> None:
        """매시 정각 - 포트폴리오 모니터링.

        보유 종목의 이상 변동을 감지하고, MDD를 갱신하며,
        알림 조건을 검사하여 필요 시 알림을 발송한다.
        """
        if not self._is_trading_day():
            return

        try:
            if not self.kis_client.is_configured():
                return

            balance = self.kis_client.get_balance()
            total_eval = balance.get("total_eval", 0)
            holdings = balance.get("holdings", [])

            # MDD 갱신
            if total_eval > 0:
                self.portfolio_tracker.update(total_eval)

            if not holdings:
                return

            # 일일 변동률 수집
            alerts = []
            holdings_daily_returns: dict[str, float] = {}

            for h in holdings:
                ticker = h.get("ticker", "")
                name = h.get("name", ticker)

                try:
                    price_info = self.kis_client.get_current_price(ticker)
                    change_pct = price_info.get("change_pct", 0.0)
                    holdings_daily_returns[ticker] = change_pct / 100.0

                    if abs(change_pct) >= 5.0:
                        direction = "급등" if change_pct > 0 else "급락"
                        alerts.append(
                            f"  {name}({ticker}): {change_pct:+.2f}% ({direction})"
                        )
                except Exception:
                    pass

            # 알림 조건용 상태 딕셔너리 구성
            today = datetime.now(KST).date()
            days_to_rebal = self.holidays.days_to_next_rebalance(
                today, self.rebalance_freq
            )

            state = {
                "holdings_daily_returns": holdings_daily_returns,
                "current_mdd": self.portfolio_tracker.get_mdd(),
                "days_to_rebalance": days_to_rebal,
            }
            self.alert_manager.check_and_alert(state)

            # 매시 정각 상태 보고 (항상 발송)
            now = datetime.now(KST)
            profit_pct = balance.get("total_profit_pct", 0.0)
            mdd_pct = self.portfolio_tracker.get_mdd()

            lines = [
                f"[포트폴리오 모니터링] {now.strftime('%H:%M')}",
                f"  평가: {total_eval:,}원 ({profit_pct:+.2f}%)",
                f"  종목: {len(holdings)}개 | MDD: {mdd_pct:.2%}",
            ]

            if alerts:
                lines.append("")
                lines.append("이상 변동 감지:")
                lines.extend(alerts)

            level = "WARNING" if alerts else "INFO"
            self._send_notification("\n".join(lines), level=level)

            logger.info(
                "포트폴리오 모니터링 완료 (%s): %d개 보유, "
                "평가 %s원, MDD %.2f%%, %d건 이상 변동",
                now.strftime("%H:%M"),
                len(holdings),
                f"{total_eval:,}",
                mdd_pct * 100,
                len(alerts),
            )

        except Exception as e:
            logger.error("포트폴리오 모니터링 실패: %s", e)
            logger.debug(traceback.format_exc())

    def eod_review(self) -> None:
        """15:35 - 장 마감 후 일일 리뷰.

        오늘 실행된 거래 내역과 포트폴리오 최종 현황을 리뷰한다.
        """
        if not self._is_trading_day():
            return

        try:
            today = datetime.now(KST).date()
            today_str = today.strftime("%Y-%m-%d")

            lines = [
                f"[EOD 리뷰] {today_str}",
                "─" * 30,
            ]

            # 오늘의 주문 이력
            order_summary = self.executor.order_manager.get_summary()
            total_orders = order_summary.get("total_orders", 0)

            if total_orders > 0:
                by_status = order_summary.get("by_status", {})
                lines.append(f"오늘 주문: {total_orders}건")
                for status, count in by_status.items():
                    lines.append(f"  - {status}: {count}건")

                buy_amt = order_summary.get("total_buy_amount", 0)
                sell_amt = order_summary.get("total_sell_amount", 0)
                if buy_amt > 0 or sell_amt > 0:
                    lines.append(f"매수 금액: {buy_amt:,.0f}원")
                    lines.append(f"매도 금액: {sell_amt:,.0f}원")
            else:
                lines.append("오늘 실행된 주문 없음")

            # 최종 포트폴리오 현황
            if self.kis_client.is_configured():
                try:
                    balance = self.kis_client.get_balance()
                    total_eval = balance.get("total_eval", 0)
                    profit_pct = balance.get("total_profit_pct", 0.0)

                    lines.append("")
                    lines.append("[장 마감 포트폴리오]")
                    lines.append(f"  총 평가: {total_eval:,}원")
                    lines.append(f"  총 수익률: {profit_pct:+.2f}%")

                    holdings = balance.get("holdings", [])
                    if holdings:
                        lines.append(f"  종목 수: {len(holdings)}개")

                        # 수익률 상위/하위
                        sorted_h = sorted(
                            holdings,
                            key=lambda x: x.get("pnl_pct", 0),
                            reverse=True,
                        )
                        if len(sorted_h) >= 1:
                            top = sorted_h[0]
                            lines.append(
                                f"  최고: {top['name']}({top['ticker']}) "
                                f"{top['pnl_pct']:+.2f}%"
                            )
                        if len(sorted_h) >= 2:
                            bottom = sorted_h[-1]
                            lines.append(
                                f"  최저: {bottom['name']}({bottom['ticker']}) "
                                f"{bottom['pnl_pct']:+.2f}%"
                            )

                except Exception as e:
                    lines.append(f"\n[포트폴리오 조회 실패: {e}]")

            message = "\n".join(lines)
            self._send_notification(message)
            logger.info("EOD 리뷰 발송 완료")

        except Exception as e:
            logger.error("EOD 리뷰 실패: %s", e)
            logger.debug(traceback.format_exc())

    def evening_report(self) -> None:
        """19:00 - 이브닝 종합 리포트.

        일일 리포트를 생성하여 텔레그램으로 발송한다.
        DailyReport 모듈을 활용한다.
        """
        if not self._is_trading_day():
            return

        try:
            today = datetime.now(KST).date()
            today_str = today.strftime("%Y-%m-%d")

            # 리밸런싱 예정일 정보
            days_to_rebal = self.holidays.days_to_next_rebalance(
                today, self.rebalance_freq
            )

            lines = [
                f"[이브닝 리포트] {today_str}",
                "=" * 40,
            ]

            # 포트폴리오 현황
            if self.kis_client.is_configured():
                try:
                    balance = self.kis_client.get_balance()
                    total_eval = balance.get("total_eval", 0)
                    cash = balance.get("cash", 0)
                    profit_pct = balance.get("total_profit_pct", 0.0)
                    holdings = balance.get("holdings", [])

                    lines.append("")
                    lines.append("[포트폴리오 현황]")
                    lines.append(f"  총 평가: {total_eval:,}원")
                    lines.append(f"  현금: {cash:,}원")
                    cash_pct = (
                        (cash / total_eval * 100) if total_eval > 0 else 0
                    )
                    lines.append(f"  현금 비중: {cash_pct:.1f}%")
                    lines.append(f"  총 수익률: {profit_pct:+.2f}%")
                    lines.append(f"  종목 수: {len(holdings)}개")

                    # 개별 종목 현황
                    if holdings:
                        lines.append("")
                        lines.append("[보유 종목]")
                        sorted_h = sorted(
                            holdings,
                            key=lambda x: x.get("eval_amount", 0),
                            reverse=True,
                        )
                        for h in sorted_h[:15]:
                            lines.append(
                                f"  {h['name'][:8]:8s} "
                                f"{h['eval_amount']:>12,}원 "
                                f"{h['pnl_pct']:>+6.1f}%"
                            )

                except Exception as e:
                    lines.append(f"\n[포트폴리오 조회 실패: {e}]")

            # 매크로 요약 추가 (Phase 6)
            if self.feature_flags.is_enabled("macro_monitor"):
                try:
                    from src.data.macro_collector import MacroCollector
                    macro = MacroCollector()
                    macro_text = macro.format_macro_report()
                    if macro_text:
                        lines.append("")
                        lines.append(macro_text)
                except Exception as e:
                    logger.warning("이브닝 매크로 요약 실패: %s", e)

            # 스케줄 정보
            lines.append("")
            lines.append("[스케줄]")
            if days_to_rebal == 0:
                lines.append("  오늘 리밸런싱 완료")
            else:
                lines.append(f"  다음 리밸런싱까지 {days_to_rebal}일")

            next_td = self.holidays.next_trading_day(today)
            lines.append(f"  다음 거래일: {next_td.strftime('%Y-%m-%d')}")

            lines.append("")
            lines.append("=" * 40)

            message = "\n".join(lines)
            self._send_notification(message)
            logger.info("이브닝 리포트 발송 완료")

        except Exception as e:
            logger.error("이브닝 리포트 실패: %s", e)
            logger.debug(traceback.format_exc())

    def health_check(self) -> None:
        """08:00 - 시스템 헬스 체크.

        KIS API 연결, Telegram 연결을 확인하고 결과를 발송한다.
        거래일이 아니어도 실행한다 (시스템 점검용).
        """
        try:
            checks: list[str] = []
            all_ok = True

            # 1) KIS API 연결 확인
            if self.kis_client.is_configured():
                try:
                    price = self.kis_client.get_current_price("005930")
                    if price.get("price", 0) > 0:
                        checks.append("KIS API: OK")
                    else:
                        checks.append("KIS API: 응답 이상 (price=0)")
                        all_ok = False
                except Exception as e:
                    checks.append(f"KIS API: 실패 ({e})")
                    all_ok = False
            else:
                checks.append("KIS API: 미설정")

            # 2) Telegram 연결 확인 (이 메시지 자체가 테스트)
            checks.append("Telegram: OK")

            # 3) 봇 상태
            today = datetime.now(KST).date()
            is_trading = self.holidays.is_trading_day(today)
            checks.append(
                f"거래일: {'예' if is_trading else '아니오 (휴장)'}"
            )

            # 4) MDD 상태
            mdd = self.portfolio_tracker.get_mdd()
            checks.append(f"MDD: {mdd:.2%}")

            level = "INFO" if all_ok else "CRITICAL"
            message = (
                f"[헬스 체크] {today.strftime('%Y-%m-%d')} 08:00\n"
                + "\n".join(f"  {c}" for c in checks)
            )
            self._send_notification(message, level=level)
            logger.info("헬스 체크 완료: %s", "정상" if all_ok else "이상 감지")

        except Exception as e:
            logger.error("헬스 체크 실패: %s", e)
            logger.debug(traceback.format_exc())

    # ──────────────────────────────────────────────────────────
    # Phase 5-A: 피처 플래그 기반 확장 작업
    # ──────────────────────────────────────────────────────────

    def global_market_check(self) -> None:
        """매시(22~06) - 글로벌 시장 모니터링.

        yfinance를 통해 S&P500, NASDAQ, VIX, 환율 등을 수집하고
        요약을 텔레그램으로 발송한다. Feature Flag 'global_monitor'로 제어.
        """
        if not self.feature_flags.is_enabled("global_monitor"):
            return

        try:
            snapshot = get_global_snapshot()
            if snapshot.empty:
                logger.warning("글로벌 스냅샷 수집 실패 (빈 결과)")
                return

            text = format_global_snapshot(snapshot)
            self._send_notification(text)
            logger.info("글로벌 시장 모니터링 발송 완료")
        except Exception as e:
            logger.error("글로벌 시장 모니터링 실패: %s", e)
            logger.debug(traceback.format_exc())

    def stock_review_job(self) -> None:
        """16:00 평일 - 보유 종목 리뷰.

        52주 고저 비교, 시그널 분석을 수행하고 결과를 발송한다.
        Feature Flag 'stock_review'로 제어.
        """
        if not self.feature_flags.is_enabled("stock_review"):
            return
        if not self._is_trading_day():
            return

        try:
            if not self.kis_client.is_configured():
                return

            balance = self.kis_client.get_balance()
            holdings = balance.get("holdings", [])
            config = self.feature_flags.get_config("stock_review")
            self.stock_reviewer.max_stocks = config.get("max_stocks", 10)

            text = self.stock_reviewer.review_holdings(holdings)
            self._send_notification(text)
            logger.info("종목 리뷰 발송 완료")
        except Exception as e:
            logger.error("종목 리뷰 실패: %s", e)
            logger.debug(traceback.format_exc())

    def auto_backtest_job(self) -> None:
        """일요일 10:00 - 자동 백테스트.

        등록된 전략들의 백테스트를 실행하고 결과를 비교하여 발송한다.
        Feature Flag 'auto_backtest'로 제어.
        """
        if not self.feature_flags.is_enabled("auto_backtest"):
            return

        try:
            config = self.feature_flags.get_config("auto_backtest")
            backtester = AutoBacktester(
                lookback_months=config.get("lookback_months", 6),
                strategies=config.get("strategies"),
            )
            text = backtester.run_all()
            self._send_notification(text)
            logger.info("자동 백테스트 완료")
        except Exception as e:
            logger.error("자동 백테스트 실패: %s", e)
            logger.debug(traceback.format_exc())

    def night_research_job(self) -> None:
        """22:00 평일 - 야간 리서치 리포트.

        글로벌 시장 동향 + 포트폴리오 상태 기반 시사점을 생성하여 발송한다.
        Feature Flag 'night_research'로 제어.
        """
        if not self.feature_flags.is_enabled("night_research"):
            return

        try:
            config = self.feature_flags.get_config("night_research")
            self.night_researcher.include_global = config.get(
                "include_global", True
            )

            # 글로벌 데이터 수집
            global_snapshot = None
            if self.night_researcher.include_global:
                global_snapshot = get_global_snapshot()

            # 포트폴리오 상태
            portfolio_state = None
            if self.kis_client.is_configured():
                try:
                    balance = self.kis_client.get_balance()
                    total_eval = balance.get("total_eval", 0)
                    cash = balance.get("cash", 0)
                    cash_pct = (cash / total_eval * 100) if total_eval > 0 else 0
                    mdd = self.portfolio_tracker.get_mdd()
                    portfolio_state = {
                        "total_eval": total_eval,
                        "cash_pct": cash_pct,
                        "mdd": mdd,
                    }
                except Exception as e:
                    logger.warning("야간 리서치: 포트폴리오 조회 실패: %s", e)

            text = self.night_researcher.generate_report(
                global_snapshot=global_snapshot,
                portfolio_state=portfolio_state,
            )
            self._send_notification(text)
            logger.info("야간 리서치 발송 완료")
        except Exception as e:
            logger.error("야간 리서치 실패: %s", e)
            logger.debug(traceback.format_exc())

    # ──────────────────────────────────────────────────────────
    # Phase 5-B: 단기 트레이딩 스케줄 작업
    # ──────────────────────────────────────────────────────────

    def short_term_scan(self) -> None:
        """08:50, 13:00 - 단기 시그널 스캔.

        등록된 단기 전략으로 시그널을 스캔하고 텔레그램으로 알린다.
        Feature Flag 'short_term_trading'로 제어.
        """
        if not self.feature_flags.is_enabled("short_term_trading"):
            return
        if not self._is_trading_day():
            return
        if self.short_term_trader is None:
            return

        try:
            # 단기 전략용 데이터 수집 (시총 상위 100종목, 260일 OHLCV)
            today_str = datetime.now(KST).strftime("%Y%m%d")
            market_data = self._collect_short_term_data(today_str)

            if not market_data.get("daily_data"):
                logger.warning("단기 시그널 스캔: 데이터 수집 실패")
                return

            signals = self.short_term_trader.scan_for_signals(market_data)

            if not signals:
                logger.info("단기 시그널 스캔: 시그널 없음")
                return

            lines = [f"[단기 시그널] {len(signals)}개 발견"]
            for sig in signals:
                lines.append(
                    f"  {sig.side.upper()} {sig.ticker} "
                    f"({sig.strategy}, 신뢰도={sig.confidence:.0%})"
                )
                lines.append(f"  사유: {sig.reason}")
                lines.append(f"  ID: {sig.id}")
                lines.append("")
            lines.append("/confirm <id> 로 승인, /reject <id> 로 거절")

            self._send_notification("\n".join(lines))
            logger.info("단기 시그널 %d개 발송", len(signals))
        except Exception as e:
            logger.error("단기 시그널 스캔 실패: %s", e)

    def short_term_monitor(self) -> None:
        """장중 30분마다 - 단기 포지션 모니터링.

        보유 단기 포지션의 손절/익절/데이터헬스를 체크한다.
        Feature Flag 'short_term_trading'로 제어.
        """
        if not self.feature_flags.is_enabled("short_term_trading"):
            return
        if not self._is_trading_day():
            return
        if self.short_term_trader is None or self.allocator is None:
            return

        try:
            positions = self.allocator.get_positions_by_pool("short_term")
            if not positions:
                return

            alerts = []
            for pos in positions:
                entry_price = pos.get(
                    "entry_price",
                    pos.get("metadata", {}).get("entry_price", 0),
                )
                current_price = pos.get("current_price", 0)
                entry_date = pos.get("entry_date", "")
                mode = pos.get(
                    "mode", pos.get("metadata", {}).get("mode", "swing")
                )

                if self.short_term_risk:
                    result = self.short_term_risk.check_position(
                        entry_price=float(entry_price),
                        current_price=float(current_price),
                        entry_date=entry_date,
                        mode=mode,
                    )

                    if result["should_close"]:
                        alerts.append({
                            "ticker": pos.get("ticker", ""),
                            "reasons": result["reasons"],
                            "pnl_pct": result["pnl_pct"],
                        })

            if alerts:
                lines = ["[단기 포지션 알림]"]
                for alert in alerts:
                    lines.append(f"  {alert['ticker']}: {alert['pnl_pct']:.2%}")
                    for r in alert["reasons"]:
                        lines.append(f"    -> {r}")
                self._send_notification("\n".join(lines), level="WARNING")

            # 확인된 시그널 실행
            confirmed = self.short_term_trader.execute_confirmed_signals()
            if confirmed:
                logger.info("확인 시그널 %d개 실행", len(confirmed))

        except Exception as e:
            logger.error("단기 포지션 모니터링 실패: %s", e)

    def daytrading_close(self) -> None:
        """15:20 - 데이트레이딩 포지션 강제 청산.

        Feature Flag 'short_term_trading'으로 제어.
        daytrading 모드인 포지션만 청산한다.
        """
        if not self.feature_flags.is_enabled("short_term_trading"):
            return
        if not self._is_trading_day():
            return
        if self.short_term_trader is None or self.allocator is None:
            return

        config = self.feature_flags.get_config("short_term_trading")
        mode = config.get("mode", "swing")
        if mode not in ("daytrading", "multi"):
            return

        try:
            positions = self.allocator.get_positions_by_pool("short_term")
            daytrading_positions = [
                p for p in positions
                if p.get("mode", p.get("metadata", {}).get("mode", "")) == "daytrading"
            ]

            if not daytrading_positions:
                logger.info("데이트레이딩 청산: 대상 없음")
                return

            lines = [f"[데이트레이딩 청산] {len(daytrading_positions)}개"]
            for pos in daytrading_positions:
                ticker = pos.get("ticker", "")
                pnl_pct = pos.get("pnl_pct", 0)
                lines.append(f"  {ticker}: {pnl_pct:.2%}")

            self._send_notification("\n".join(lines))
            logger.info("데이트레이딩 %d개 청산 알림", len(daytrading_positions))

        except Exception as e:
            logger.error("데이트레이딩 청산 실패: %s", e)

    def _cmd_portfolio(self, args: str) -> str:
        """Telegram /portfolio 커맨드 핸들러."""
        if not self.kis_client.is_configured():
            return "KIS API 미설정 상태입니다."
        try:
            balance = self.kis_client.get_balance()
            total_eval = balance.get("total_eval", 0)
            cash = balance.get("cash", 0)
            profit_pct = balance.get("total_profit_pct", 0.0)
            holdings = balance.get("holdings", [])
            mdd = self.portfolio_tracker.get_mdd()

            lines = [
                "[포트폴리오 현황]",
                f"  총 평가: {total_eval:,}원",
                f"  현금: {cash:,}원",
                f"  수익률: {profit_pct:+.2f}%",
                f"  MDD: {mdd:.2%}",
                f"  종목 수: {len(holdings)}개",
            ]
            if holdings:
                lines.append("")
                sorted_h = sorted(
                    holdings,
                    key=lambda x: x.get("eval_amount", 0),
                    reverse=True,
                )
                for h in sorted_h[:10]:
                    lines.append(
                        f"  {h['name'][:8]:8s} "
                        f"{h['eval_amount']:>10,}원 "
                        f"{h['pnl_pct']:>+6.1f}%"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"포트폴리오 조회 오류: {e}"

    def _cmd_balance(self, args: str) -> str:
        """Telegram /balance 커맨드 핸들러.

        전체 포트폴리오 요약: 계좌 총괄 + 통합 보유종목 + 전략 프리뷰.
        """
        if not self.kis_client.is_configured():
            return "KIS API 미설정 상태입니다."

        try:
            # 1. 계좌 총괄
            balance = self.kis_client.get_balance()
            total_eval = balance.get("total_eval", 0)
            cash = balance.get("cash", 0)
            profit_pct = balance.get("total_profit_pct", 0.0)
            holdings = balance.get("holdings", [])
            holdings_map = {h["ticker"]: h for h in holdings}

            lines = [
                "[계좌 총괄]",
                f"  총 평가: {total_eval:,}원",
                f"  현금: {cash:,}원",
                f"  수익률: {profit_pct:+.2f}%",
                f"  종목 수: {len(holdings)}개",
            ]

            # 풀 배분 요약 (allocator 활성 시)
            if self.allocator:
                alloc = self.allocator.rebalance_allocation()
                long_pct = alloc.get("long_term_actual", 0)
                long_tgt = alloc.get("long_term_target", 0)
                etf_pct = alloc.get("etf_rotation_actual", 0)
                etf_tgt = alloc.get("etf_rotation_target", 0)
                lines.append(
                    f"  풀 배분: 장기 {long_pct:.0%}/{long_tgt:.0%}"
                    f" | ETF {etf_pct:.0%}/{etf_tgt:.0%}"
                )

            # 2. 현재 보유 종목 (통합 리스트 + 풀 태그)
            lines.append("")
            lines.append("[현재 보유 종목]")
            lines.extend(
                self._format_merged_holdings(holdings, holdings_map)
            )

            # 3. 장기 전략 프리뷰
            if self.feature_flags.is_enabled("daily_simulation"):
                lines.append("")
                try:
                    lines.extend(
                        self._generate_long_term_preview(holdings_map)
                    )
                except Exception as e:
                    lines.append(f"[장기 전략 프리뷰] 생성 실패: {e}")

            # 4. ETF 로테이션 프리뷰
            if self.feature_flags.is_enabled("etf_rotation"):
                lines.append("")
                lines.append("[ETF 리밸런싱 프리뷰 (오늘 기준)]")
                try:
                    lines.extend(self._generate_etf_preview())
                except Exception as e:
                    lines.append(f"  프리뷰 생성 실패: {e}")

            # 5. 통합 매수/매도 예상 (allocator 활성 시)
            if self.allocator and self.executor:
                try:
                    pool_signals = self._build_balance_pool_signals(
                        holdings_map
                    )
                    if pool_signals:
                        integrated = self.executor.dry_run_integrated(
                            pool_signals
                        )
                        sell_n = len(integrated.get("sell_orders", []))
                        buy_n = len(integrated.get("buy_orders", []))
                        sell_amt = integrated.get("total_sell_amount", 0)
                        buy_amt = integrated.get("total_buy_amount", 0)
                        lines.append("")
                        lines.append("[통합 매수/매도 예상]")
                        if sell_n or buy_n:
                            lines.append(
                                f"  매도 {sell_n}건 ({sell_amt:,}원)"
                                f" / 매수 {buy_n}건 ({buy_amt:,}원)"
                            )
                        else:
                            lines.append("  변경 없음")
                except Exception as e:
                    logger.debug("통합 프리뷰 실패: %s", e)

            result = "\n".join(lines)
            if len(result) > 4096:
                result = result[:4090] + "\n(...)"
            return result

        except Exception as e:
            return f"잔고 조회 오류: {e}"

    def _format_merged_holdings(
        self,
        holdings: list[dict],
        holdings_map: dict[str, dict],
    ) -> list[str]:
        """모든 보유 종목을 풀 태그와 함께 통합 리스트로 포맷한다."""
        # 풀별 ticker→label 매핑 구축
        pool_labels = {
            "long_term": "장기",
            "etf_rotation": "ETF",
            "short_term": "단기",
        }
        ticker_pool: dict[str, str] = {}

        if self.allocator:
            for pool_name, label in pool_labels.items():
                for p in self.allocator.get_positions_by_pool(pool_name):
                    ticker_pool[p["ticker"]] = label

        sorted_holdings = sorted(
            holdings,
            key=lambda x: x.get("eval_amount", 0),
            reverse=True,
        )

        lines: list[str] = []
        max_display = 15
        for h in sorted_holdings[:max_display]:
            label = ticker_pool.get(h["ticker"], "미분류") if self.allocator else ""
            prefix = f"{label}| " if label else "  "
            lines.append(
                f"  {prefix}{h['name'][:10]:10s} "
                f"{h.get('qty', 0):>5d}주 "
                f"{h.get('eval_amount', 0):>10,}원 "
                f"{h.get('pnl_pct', 0.0):>+6.1f}%"
            )

        remaining = len(sorted_holdings) - max_display
        if remaining > 0:
            lines.append(f"  ...외 {remaining}종목")

        if not holdings:
            lines.append("  (보유 없음)")

        return lines

    def _generate_long_term_preview(
        self, holdings_map: dict[str, dict],
    ) -> list[str]:
        """DailySimulator 캐시에서 장기 전략 프리뷰를 생성한다."""
        sim_config = self.feature_flags.get_config("daily_simulation")
        strategy_name = sim_config.get("primary_strategy", "multi_factor")

        sim_data = self._find_latest_simulation(strategy_name)
        if sim_data is None:
            return [
                f"[장기 전략 프리뷰 ({strategy_name})]",
                "  (시뮬레이션 데이터 없음 - 16:05 이후 갱신)",
            ]

        sim_date = sim_data.get("date", "?")
        selected = sim_data.get("selected", [])

        lines = [f"[장기 전략 프리뷰 ({sim_date} 기준)]"]

        max_display = 7
        for item in selected[:max_display]:
            rank = item.get("rank", "?")
            name = item.get("name", item.get("ticker", "?"))
            weight = item.get("weight", 0)
            ticker = item.get("ticker", "")
            held = "(보유중)" if ticker in holdings_map else "(신규)"
            lines.append(
                f"  {rank}. {name}: {weight:.1%} {held}"
            )

        remaining = len(selected) - max_display
        if remaining > 0:
            lines.append(f"  ...외 {remaining}종목")

        # dry_run으로 매수/매도 예상
        if self.executor and selected:
            try:
                signals = {
                    item["ticker"]: item["weight"]
                    for item in selected
                }
                dry = self.executor.dry_run(signals, pool="long_term")
                sell_count = len(dry.get("sell_orders", []))
                buy_count = len(dry.get("buy_orders", []))
                lines.append(
                    f"  예상 변경: 매도 {sell_count}건, 매수 {buy_count}건"
                )
            except Exception:
                pass

        return lines

    def _build_balance_pool_signals(
        self, holdings_map: dict[str, dict],
    ) -> dict[str, dict[str, float]]:
        """현재 시뮬레이션 캐시에서 풀별 시그널을 구성한다."""
        pool_signals: dict[str, dict[str, float]] = {}

        # 장기 시그널 (최신 시뮬레이션에서)
        sim_config = self.feature_flags.get_config("daily_simulation")
        primary = sim_config.get("primary_strategy", "multi_factor")
        sim_data = self._find_latest_simulation(primary)
        if sim_data:
            signals = {
                item["ticker"]: item["weight"]
                for item in sim_data.get("selected", [])
            }
            if signals:
                pool_signals["long_term"] = signals

        # ETF 시그널
        if self.feature_flags.is_enabled("etf_rotation"):
            date_str = datetime.now(KST).strftime("%Y%m%d")
            etf_signals = self._generate_etf_signals(date_str)
            if etf_signals:
                pool_signals["etf_rotation"] = etf_signals

        return pool_signals

    @staticmethod
    def _find_latest_simulation(strategy_name: str) -> Optional[dict]:
        """DailySimulator에서 최신 시뮬레이션 결과를 가져온다."""
        try:
            from src.data.daily_simulator import DailySimulator
            sim = DailySimulator()
            history = sim.get_history(strategy_name, days=7)
            return history[0] if history else None
        except Exception:
            return None

    def _generate_etf_preview(self) -> list[str]:
        """ETF 로테이션 모멘텀 순위 + 시그널 프리뷰를 생성한다."""
        strategy = self._create_etf_strategy()

        config = self.feature_flags.get_config("etf_rotation")
        lookback_months = config.get("lookback_months", 12)
        lookback_days = lookback_months * 21

        etf_prices = self._fetch_etf_prices(
            strategy.etf_universe, lookback_days
        )

        if not etf_prices:
            return ["  ETF 가격 수집 실패 (pykrx API 응답 없음)"]

        # 시그널 생성 (fallback + 진단 포함)
        date_str = datetime.now(KST).strftime("%Y%m%d")
        signals = strategy.generate_signals(
            date_str, {"etf_prices": etf_prices}
        )

        diag = strategy.last_diagnostics
        lines = []

        # Enhanced 전략 여부에 따라 분기
        is_enhanced = self.feature_flags.is_enabled("enhanced_etf_rotation")

        if is_enhanced:
            regime = diag.get("market_regime", "N/A")
            lines.append(f"  전략: Enhanced ETF (regime={regime})")

            composite_scores = diag.get("composite_scores", {})
            if composite_scores:
                ranked = sorted(
                    composite_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                lines.append("  복합 모멘텀 순위:")
                for i, (ticker, score) in enumerate(ranked, 1):
                    name = strategy.etf_universe.get(ticker, ticker)
                    lines.append(f"    {i}. {name}: {score:+.2f}")
        else:
            lb_used = diag.get("lookback_used", lookback_days)
            if diag.get("status") == "DEGRADED":
                lines.append(
                    f"  * lookback: {lb_used}일 "
                    f"({lookback_days}일 데이터 부족으로 하향)"
                )

            per_ticker = diag.get("per_ticker", {})
            ok_tickers = {
                t: info["momentum"]
                for t, info in per_ticker.items()
                if info.get("status") == "OK"
            }
            if ok_tickers:
                ranked = sorted(
                    ok_tickers.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                lines.append("  모멘텀 순위:")
                for i, (ticker, mom) in enumerate(ranked, 1):
                    name = strategy.etf_universe.get(ticker, ticker)
                    lines.append(f"    {i}. {name}: {mom:+.1%}")

            problem_tickers = [
                (t, info)
                for t, info in per_ticker.items()
                if info.get("status") in (
                    "DATA_MISSING", "DATA_SHORT", "DATA_INVALID"
                )
            ]
            if problem_tickers:
                for ticker, info in problem_tickers:
                    name = strategy.etf_universe.get(ticker, ticker)
                    status = info["status"]
                    avail = info.get("available_days", 0)
                    req = info.get("required_days", lb_used)
                    if status == "DATA_MISSING":
                        lines.append(f"  * {name}: 데이터 누락")
                    elif status == "DATA_SHORT":
                        lines.append(
                            f"  * {name}: 데이터 부족 ({avail}/{req}일)"
                        )
                    elif status == "DATA_INVALID":
                        lines.append(f"  * {name}: 데이터 오류")

        if signals:
            lines.append("  선정 ETF:")
            for ticker, weight in sorted(
                signals.items(), key=lambda x: x[1], reverse=True,
            ):
                name = strategy.etf_universe.get(ticker, ticker)
                lines.append(f"    {name}: {weight:.1%}")

            # Dry run
            if self.executor:
                try:
                    dry = self.executor.dry_run(
                        signals, pool="etf_rotation"
                    )
                    sell_count = len(dry.get("sell_orders", []))
                    buy_count = len(dry.get("buy_orders", []))
                    lines.append(
                        f"  예상 변경: 매도 {sell_count}건, 매수 {buy_count}건"
                    )
                except Exception:
                    pass
        else:
            reason = diag.get("reason", "원인 불명")
            lines.append(f"  시그널 없음: {reason}")

        return lines

    # ──────────────────────────────────────────────────────────
    # Phase 6: 일일 시뮬레이션 + 뉴스/공시 + 매크로 + 성과DB
    # ──────────────────────────────────────────────────────────

    def morning_news_checklist(self) -> None:
        """08:00 - DART 공시 + 매크로 오전 체크리스트.

        전일 공시와 글로벌 매크로 요약을 텔레그램으로 발송한다.
        Feature Flag 'news_collector'로 제어.
        """
        if not self.feature_flags.is_enabled("news_collector"):
            return
        if not self._is_trading_day():
            return

        try:
            from src.data.news_collector import NewsCollector
            collector = NewsCollector()
            disclosures = collector.fetch_recent_disclosures(days=1)

            # 매크로 데이터 추가
            macro_summary = None
            if self.feature_flags.is_enabled("macro_monitor"):
                try:
                    from src.data.macro_collector import MacroCollector
                    macro = MacroCollector()
                    macro_summary = macro.format_macro_report()
                except Exception as e:
                    logger.warning("매크로 데이터 수집 실패: %s", e)

            text = collector.format_morning_checklist(
                disclosures, macro_data=macro_summary
            )
            self._send_notification(text)
            logger.info("오전 뉴스 체크리스트 발송 완료")
        except Exception as e:
            logger.error("오전 뉴스 체크리스트 실패: %s", e)
            logger.debug(traceback.format_exc())

    def eod_news_summary(self) -> None:
        """15:40 - 보유종목 영향 뉴스 + 매크로 요약.

        장 마감 후 보유종목 관련 공시와 매크로 변동을 요약한다.
        포트폴리오 보유종목과 전략 시그널 상위 종목(관심종목)의 공시를 우선 표시하고,
        각 공시에 감성 분류(상승/하락/중립) 아이콘을 포함한다.
        Feature Flag 'news_collector'로 제어.
        """
        if not self.feature_flags.is_enabled("news_collector"):
            return
        if not self._is_trading_day():
            return

        try:
            from src.data.news_collector import NewsCollector
            collector = NewsCollector()
            disclosures = collector.fetch_recent_disclosures(days=1)

            # 보유종목 ticker 추출 (KIS holdings → stock_code 리스트)
            holding_tickers: list[str] = []
            if self.kis_client.is_configured():
                try:
                    balance = self.kis_client.get_balance()
                    raw_holdings = balance.get("holdings", [])
                    holding_tickers = [
                        h.get("ticker", "") for h in raw_holdings
                        if h.get("ticker")
                    ]
                except Exception:
                    pass

            # 전략 시그널 상위 종목 (관심종목/watchlist)
            watchlist_tickers: list[str] = []
            try:
                watchlist_tickers = self._collect_eod_watchlist()
            except Exception:
                logger.debug("EOD 관심종목 수집 실패", exc_info=True)

            text = collector.format_eod_news(
                disclosures, holding_tickers, watchlist_tickers
            )
            self._send_notification(text)
            logger.info("장마감 뉴스 요약 발송 완료")
        except Exception as e:
            logger.error("장마감 뉴스 요약 실패: %s", e)
            logger.debug(traceback.format_exc())

    def _collect_eod_watchlist(self, top_n: int = 20) -> list[str]:
        """전략 시그널 상위 종목 ticker 리스트를 수집한다.

        장기 전략 + ETF 로테이션 시그널의 상위 종목을 관심종목으로 반환.

        Args:
            top_n: 장기 전략 상위 종목 수. 기본 20.

        Returns:
            관심종목 ticker 리스트.
        """
        watchlist: list[str] = []

        # 장기 전략의 최근 시그널 캐시 (08:50 또는 09:05에 생성된 것)
        if hasattr(self, '_last_long_signals') and self._last_long_signals:
            top_tickers = sorted(
                self._last_long_signals.items(),
                key=lambda x: x[1], reverse=True,
            )[:top_n]
            watchlist.extend(t for t, _ in top_tickers)

        # ETF 로테이션 시그널
        if hasattr(self, '_last_etf_signals') and self._last_etf_signals:
            watchlist.extend(self._last_etf_signals.keys())

        return watchlist

    def daily_simulation_batch(self) -> None:
        """16:05 - 일일 리밸런싱 시뮬레이션.

        매일 장 마감 후 모든 등록 전략의 시그널을 생성하고,
        현재 보유 대비 매수/매도 예상을 텔레그램으로 발송한다.
        Feature Flag 'daily_simulation'로 제어.
        """
        if not self.feature_flags.is_enabled("daily_simulation"):
            return
        if not self._is_trading_day():
            return

        try:
            from src.data.daily_simulator import DailySimulator
            simulator = DailySimulator()

            # 전략 인스턴스 생성
            config = self.feature_flags.get_config("daily_simulation")
            strategy_names = config.get(
                "strategies", ["multi_factor", "three_factor"]
            )
            strategies = {}
            for name in strategy_names:
                try:
                    strategy = self._create_long_term_strategy(name)
                    if strategy:
                        strategies[name] = strategy
                except Exception as e:
                    logger.warning("시뮬레이션 전략 생성 실패 (%s): %s", name, e)

            if not strategies:
                logger.warning("시뮬레이션 가능한 전략이 없습니다.")
                return

            simulator.strategies = strategies

            # 장 마감 후이므로 당일(T) 데이터 사용 가능
            today_str = datetime.now(KST).strftime("%Y%m%d")
            strategy_data = self._collect_strategy_data(today_str)
            simulator.strategy_data = strategy_data

            # ETF 전략이 포함되어 있으면 가격 데이터 수집하여 주입
            if "etf_rotation" in strategies:
                etf_strategy = strategies["etf_rotation"]
                lookback = getattr(etf_strategy, "lookback", 252)
                simulator.etf_prices = self._fetch_etf_prices(
                    etf_strategy.etf_universe, lookback
                )

            result = simulator.run_daily_simulation()

            # Dry-run: 현재 포트폴리오 대비 매수/매도 예상
            if self.kis_client.is_configured():
                # 전략별 개별 dry_run (pool 파라미터 포함)
                dry_run_results: dict = {}
                pool_signals: dict[str, dict[str, float]] = {}

                for name, sim_data in result.items():
                    try:
                        signals = {
                            item["ticker"]: item["weight"]
                            for item in sim_data.get("selected", [])
                        }
                        if not signals:
                            continue

                        # 풀 결정: etf_rotation 전략은 etf_rotation 풀
                        pool = (
                            "etf_rotation"
                            if name == "etf_rotation"
                            else "long_term"
                        )
                        dry = self.executor.dry_run(signals, pool=pool)
                        dry_run_results[name] = dry
                        pool_signals[pool] = signals
                    except Exception as e:
                        logger.debug("dry_run 실패 (%s): %s", name, e)
                simulator.dry_run_results = dry_run_results

                # 통합 dry_run (allocator 활성 시)
                if self.allocator and pool_signals:
                    try:
                        integrated = self.executor.dry_run_integrated(
                            pool_signals
                        )
                        simulator.integrated_dry_run = integrated
                        simulator.pool_allocation = {
                            "long_term": self.allocator.get_pool_pct(
                                "long_term"
                            ),
                            "etf_rotation": self.allocator.get_pool_pct(
                                "etf_rotation"
                            ),
                        }
                    except Exception as e:
                        logger.debug("통합 dry_run 실패: %s", e)

            # 통합 리포트 사용 (allocator 활성 시)
            if simulator.integrated_dry_run:
                text = simulator.format_integrated_report()
            else:
                text = simulator.format_telegram_report()
            self._send_notification(text)
            logger.info("일일 시뮬레이션 완료: %d개 전략", len(strategies))
        except Exception as e:
            logger.error("일일 시뮬레이션 실패: %s", e)
            logger.debug(traceback.format_exc())

    def record_daily_performance(self) -> None:
        """15:35 EOD 리뷰 후 - 일일 성과를 DB에 기록.

        PerformanceDB에 NAV, 포지션, 거래 내역을 저장한다.
        """
        try:
            from src.data.performance_db import PerformanceDB
            db = PerformanceDB()

            if not self.kis_client.is_configured():
                return

            balance = self.kis_client.get_balance()
            total_eval = balance.get("total_eval", 0)
            cash = balance.get("cash", 0)
            holdings = balance.get("holdings", [])

            if total_eval <= 0:
                return

            # EOD 포트폴리오 트래커 스냅샷
            self.portfolio_tracker.update(total_eval)

            today = datetime.now(KST).strftime("%Y-%m-%d")

            # NAV 기록
            positions_value = total_eval - cash
            db.record_daily_nav(today, total_eval, cash, positions_value)

            # 포지션 기록
            positions = []
            for h in holdings:
                positions.append({
                    "ticker": h.get("ticker", ""),
                    "name": h.get("name", ""),
                    "qty": h.get("qty", 0),
                    "avg_price": h.get("avg_price", 0),
                    "market_value": h.get("eval_amount", 0),
                    "weight": (
                        h.get("eval_amount", 0) / total_eval
                        if total_eval > 0 else 0
                    ),
                })
            db.record_positions(today, positions)

            logger.info("일일 성과 DB 기록 완료: NAV=%s", f"{total_eval:,}")
        except Exception as e:
            logger.error("성과 DB 기록 실패: %s", e)
            logger.debug(traceback.format_exc())

    def _create_long_term_strategy(self, name: str):
        """전략 이름으로 장기 전략 인스턴스를 생성한다.

        feature flag config 값을 전략 생성 시 주입한다.

        Args:
            name: 전략 이름.

        Returns:
            Strategy 인스턴스 또는 None.
        """
        try:
            if name == "multi_factor":
                from src.strategy.multi_factor import MultiFactorStrategy
                return MultiFactorStrategy(
                    factors=["value", "momentum"],
                    weights=[0.5, 0.5],
                    combine_method="zscore",
                    num_stocks=7,
                )
            elif name == "three_factor":
                from src.strategy.three_factor import ThreeFactorStrategy
                return ThreeFactorStrategy(num_stocks=10)
            elif name == "shareholder_yield":
                from src.strategy.shareholder_yield import ShareholderYieldStrategy
                return ShareholderYieldStrategy(num_stocks=10)
            elif name == "low_vol_quality":
                from src.strategy.low_vol_quality import LowVolQualityStrategy
                return LowVolQualityStrategy(num_stocks=10)
            elif name == "etf_rotation":
                return self._create_etf_strategy()
            elif name == "accrual":
                from src.strategy.accrual import AccrualStrategy
                return AccrualStrategy(num_stocks=10)
            elif name == "pead":
                from src.strategy.pead import PEADStrategy
                return PEADStrategy(num_stocks=10)
            elif name == "value":
                from src.strategy.value import ValueStrategy
                return ValueStrategy(num_stocks=10)
            elif name == "momentum":
                from src.strategy.momentum import MomentumStrategy
                return MomentumStrategy(num_stocks=10)
            else:
                logger.warning("알 수 없는 장기 전략: %s", name)
                return None
        except Exception as e:
            logger.warning("전략 생성 실패 (%s): %s", name, e)
            return None

    # ──────────────────────────────────────────────────────────
    # 긴급 리밸런싱 모니터
    # ──────────────────────────────────────────────────────────

    def emergency_monitor_check(self) -> None:
        """장중 30분마다 - 긴급 리밸런싱 모니터링.

        보유 종목의 급등/급락, KOSPI 급변, 긴급 공시를 감지하여
        알림을 발송한다. auto_exit_enabled가 True이고 CRITICAL 레벨이면
        긴급 매도를 트리거한다.

        Feature Flag 'emergency_monitor'로 제어.
        setup_schedule()에서 CronTrigger(minute="*/30", hour="9-15")로 등록됨.
        """
        if not self.feature_flags.is_enabled("emergency_monitor"):
            return
        if not self._is_trading_day():
            return

        config = self.feature_flags.get_config("emergency_monitor")
        price_shock_pct = config.get("price_shock_pct", 5.0)
        market_crash_pct = config.get("market_crash_pct", 3.0)
        auto_exit_enabled = config.get("auto_exit_enabled", False)

        state: dict = {
            "price_shocks": [],
            "market_change_pct": 0.0,
            "portfolio_disclosures": [],
        }

        try:
            # 1. 보유 종목 급등/급락 체크
            holdings = []
            if self.kis_client.is_configured():
                try:
                    balance = self.kis_client.get_balance()
                    holdings = balance.get("holdings", [])
                except Exception as e:
                    logger.warning("긴급 모니터 잔고 조회 실패: %s", e)

            price_shocks = []
            for h in holdings:
                change_pct = float(h.get("change_pct", 0))
                if abs(change_pct) >= price_shock_pct:
                    price_shocks.append({
                        "name": h.get("name", ""),
                        "ticker": h.get("ticker", ""),
                        "change": change_pct,
                    })
            state["price_shocks"] = price_shocks

            # 2. KOSPI 지수 변동 체크
            try:
                from pykrx import stock as pykrx_stock
                today_str = datetime.now(KST).strftime("%Y%m%d")
                yesterday = (
                    datetime.now(KST) - timedelta(days=7)
                ).strftime("%Y%m%d")
                kospi = pykrx_stock.get_index_ohlcv(
                    yesterday, today_str, "1001"
                )
                if len(kospi) >= 2:
                    prev_close = float(kospi["종가"].iloc[-2])
                    curr_close = float(kospi["종가"].iloc[-1])
                    if prev_close > 0:
                        market_change = (
                            (curr_close - prev_close) / prev_close * 100
                        )
                        state["market_change_pct"] = market_change
            except Exception as e:
                logger.warning("긴급 모니터 KOSPI 조회 실패: %s", e)

            # 3. DART 긴급 공시 체크
            try:
                from src.data.news_collector import NewsCollector
                collector = NewsCollector()
                disclosures = collector.fetch_recent_disclosures(days=1)

                held_tickers = {
                    h.get("ticker", "") for h in holdings
                }
                held_names = {
                    h.get("name", "") for h in holdings
                }

                portfolio_disclosures = []
                for d in disclosures:
                    corp_name = d.get("corp_name", "")
                    is_held = (
                        d.get("corp_code", "") in held_tickers
                        or corp_name in held_names
                    )
                    category = d.get("category", "")
                    portfolio_disclosures.append({
                        "corp_name": corp_name,
                        "report_nm": d.get("report_nm", ""),
                        "category": category,
                        "is_held": is_held,
                    })
                state["portfolio_disclosures"] = portfolio_disclosures
            except Exception as e:
                logger.warning("긴급 모니터 공시 조회 실패: %s", e)

            # 4. 알림 조건 검사 및 발송
            triggered = self.alert_manager.check_and_alert(state)

            if triggered:
                logger.info(
                    "긴급 모니터 알림 %d건 발동", len(triggered)
                )

                # 5. auto_exit: CRITICAL 레벨이면 긴급 매도
                if auto_exit_enabled:
                    has_critical = (
                        len(state["price_shocks"]) > 0
                        and any(
                            abs(s["change"]) >= 7.0
                            for s in state["price_shocks"]
                        )
                    ) or abs(state["market_change_pct"]) > market_crash_pct

                    if has_critical and self.kis_client.is_configured():
                        self._send_notification(
                            "[긴급 매도] CRITICAL 조건 감지 — "
                            "자동 매도를 시도합니다.",
                            level="CRITICAL",
                        )
                        logger.warning(
                            "긴급 자동 매도 트리거 (auto_exit_enabled=True)"
                        )
            else:
                logger.debug("긴급 모니터 정상: 이상 없음")

        except Exception as e:
            logger.error("긴급 모니터 체크 실패: %s", e)
            logger.debug(traceback.format_exc())

    # ──────────────────────────────────────────────────────────
    # 스케줄 설정 및 실행
    # ──────────────────────────────────────────────────────────

    def setup_schedule(self) -> None:
        """스케줄을 등록한다.

        월~금 기준으로 작업을 등록한다.
        거래일 여부는 각 작업 내부에서 확인한다.
        """
        self.scheduler.add_job(
            self.morning_briefing,
            CronTrigger(hour=7, minute=0, day_of_week="mon-fri"),
            id="morning_briefing",
            name="모닝 브리핑",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.health_check,
            CronTrigger(hour=8, minute=0, day_of_week="mon-fri"),
            id="health_check",
            name="헬스 체크",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.premarket_check,
            CronTrigger(hour=8, minute=50, day_of_week="mon-fri"),
            id="premarket_check",
            name="장 전 시그널 체크",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.execute_rebalance,
            CronTrigger(hour=9, minute=5, day_of_week="mon-fri"),
            id="execute_rebalance",
            name="리밸런싱 실행",
            misfire_grace_time=600,
        )

        # ETF 로테이션 리밸런싱 (월간, 장기 리밸런싱 5분 후)
        self.scheduler.add_job(
            self.execute_etf_rotation_rebalance,
            CronTrigger(hour=9, minute=10, day_of_week="mon-fri"),
            id="etf_rotation_rebalance",
            name="ETF 로테이션 리밸런싱",
            misfire_grace_time=600,
        )

        self.scheduler.add_job(
            self.hourly_monitor,
            CronTrigger(minute=0, day_of_week="mon-fri", hour="9-15"),
            id="hourly_monitor",
            name="포트폴리오 모니터링",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.eod_review,
            CronTrigger(hour=15, minute=35, day_of_week="mon-fri"),
            id="eod_review",
            name="EOD 리뷰",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.evening_report,
            CronTrigger(hour=19, minute=0, day_of_week="mon-fri"),
            id="evening_report",
            name="이브닝 리포트",
            misfire_grace_time=300,
        )

        # Phase 5-A: 피처 플래그 기반 확장 스케줄
        self.scheduler.add_job(
            self.stock_review_job,
            CronTrigger(hour=16, minute=0, day_of_week="mon-fri"),
            id="stock_review",
            name="종목 리뷰",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.night_research_job,
            CronTrigger(hour=22, minute=0, day_of_week="mon-fri"),
            id="night_research",
            name="야간 리서치",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.global_market_check,
            CronTrigger(minute=30, hour="22-23,0-6"),
            id="global_monitor",
            name="글로벌 시장 모니터",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.auto_backtest_job,
            CronTrigger(hour=10, minute=0, day_of_week="sun"),
            id="auto_backtest",
            name="자동 백테스트",
            misfire_grace_time=600,
        )

        # Phase 6: 일일 시뮬레이션 + 뉴스/공시 + 매크로 + 성과DB
        self.scheduler.add_job(
            self.morning_news_checklist,
            CronTrigger(hour=8, minute=5, day_of_week="mon-fri"),
            id="morning_news_checklist",
            name="오전 뉴스 체크리스트",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.eod_news_summary,
            CronTrigger(hour=15, minute=40, day_of_week="mon-fri"),
            id="eod_news_summary",
            name="장마감 뉴스 요약",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.daily_simulation_batch,
            CronTrigger(hour=16, minute=5, day_of_week="mon-fri"),
            id="daily_simulation",
            name="일일 시뮬레이션",
            misfire_grace_time=600,
        )

        self.scheduler.add_job(
            self.record_daily_performance,
            CronTrigger(hour=15, minute=37, day_of_week="mon-fri"),
            id="record_performance",
            name="성과 DB 기록",
            misfire_grace_time=300,
        )

        # Phase 5-B: 단기 트레이딩 스케줄
        self.scheduler.add_job(
            self.short_term_scan,
            CronTrigger(hour="8,13", minute=50, day_of_week="mon-fri"),
            id="short_term_scan",
            name="단기 시그널 스캔",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.short_term_monitor,
            CronTrigger(minute="*/30", hour="9-15", day_of_week="mon-fri"),
            id="short_term_monitor",
            name="단기 포지션 모니터링",
            misfire_grace_time=300,
        )

        self.scheduler.add_job(
            self.daytrading_close,
            CronTrigger(hour=15, minute=20, day_of_week="mon-fri"),
            id="daytrading_close",
            name="데이트레이딩 청산",
            misfire_grace_time=300,
        )

        # 긴급 리밸런싱 모니터 (장중 30분 간격)
        self.scheduler.add_job(
            self.emergency_monitor_check,
            CronTrigger(minute="*/30", hour="9-15", day_of_week="mon-fri"),
            id="emergency_monitor",
            name="긴급 리밸런싱 모니터",
            misfire_grace_time=300,
        )

        logger.info(
            "스케줄 등록 완료: %d개 작업", len(self.scheduler.get_jobs())
        )
        for job in self.scheduler.get_jobs():
            logger.info("  - %s (%s)", job.name, job.trigger)

    def run(self) -> None:
        """트레이딩 봇을 실행한다.

        스케줄을 등록하고 스케줄러를 시작한다.
        SIGINT(Ctrl+C) 또는 SIGTERM 시 graceful shutdown을 수행한다.
        """
        self.setup_schedule()

        # 시그널 핸들러 등록
        def _shutdown_handler(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("시그널 수신: %s. 트레이딩 봇을 종료합니다.", sig_name)
            self.scheduler.shutdown(wait=False)

        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

        strategy_name = (
            getattr(self._strategy, "name", "미설정")
            if self._strategy
            else "미설정"
        )

        logger.info("=" * 60)
        logger.info("트레이딩 봇 시작")
        logger.info("  모드: %s", self.kis_client.trading_mode)
        logger.info("  리밸런싱 주기: %s", self.rebalance_freq)
        logger.info("  KIS API: %s", "설정됨" if self.kis_client.is_configured() else "미설정")
        logger.info("  텔레그램: %s", "설정됨" if self.notifier.is_configured() else "미설정")
        logger.info("  전략: %s", strategy_name)
        logger.info("  피처 플래그: %s", self.feature_flags.get_all_status())
        if self.allocator:
            logger.info(
                "  풀 할당: 장기=%.0f%%, ETF=%.0f%%, 단기=%.0f%%",
                self.allocator._long_term_pct * 100,
                self.allocator._etf_rotation_pct * 100,
                self.allocator._short_term_pct * 100,
            )
        else:
            logger.info("  풀 할당: 100%% 장기 (분할 없음)")
        logger.info("=" * 60)

        self._send_notification(
            f"[봇 시작] {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"모드: {self.kis_client.trading_mode}\n"
            f"전략: {strategy_name}\n"
            f"리밸런싱 주기: {self.rebalance_freq}\n"
            f"\n{self.feature_flags.get_summary()}"
        )

        # Telegram 양방향 커맨드 폴링 시작
        self.commander.start_polling()

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("트레이딩 봇 종료")
        except Exception as e:
            logger.error("트레이딩 봇 예외 종료: %s", e)
            logger.debug(traceback.format_exc())
        finally:
            self.commander.stop_polling()
            self._send_notification(
                f"[봇 종료] {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}"
            )


def main() -> None:
    """트레이딩 봇 진입점."""
    bot = TradingBot()

    # 전략 설정 (실제 운영 시 원하는 전략으로 교체)
    try:
        from src.strategy.multi_factor import MultiFactorStrategy

        strategy = MultiFactorStrategy(
            factors=["value", "momentum"],
            weights=[0.5, 0.5],
            combine_method="zscore",
            num_stocks=7,
            apply_market_timing=True,
            turnover_penalty=0.1,
        )
        bot.set_strategy(strategy)
    except Exception as e:
        logger.warning("기본 전략 로드 실패: %s. 전략 없이 시작합니다.", e)

    bot.run()


if __name__ == "__main__":
    main()
