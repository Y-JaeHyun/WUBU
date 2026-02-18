"""퀀트 트레이딩 봇 메인 스케줄러.

APScheduler를 사용하여 장 전/장중/장 후 작업을 자동 실행한다.
systemd 서비스로 등록하여 무인 운영할 수 있다.

스케줄:
    07:00 - 모닝 브리핑 (마켓 요약 + 포트폴리오 현황)
    08:50 - 장 전 시그널 체크 (리밸런싱 대상일 판별)
    09:05 - 리밸런싱 실행 (해당일에만)
    매시 정각(09~15) - 포트폴리오 모니터링
    15:35 - 장 마감 후 일일 리뷰
    19:00 - 이브닝 종합 리포트
"""

from __future__ import annotations

import signal
import sys
import traceback
from datetime import datetime
from typing import Optional

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from src.alert.alert_manager import AlertManager
from src.alert.telegram_bot import TelegramNotifier
from src.execution.executor import RebalanceExecutor
from src.execution.kis_client import KISClient
from src.execution.risk_guard import RiskGuard
from src.scheduler.holidays import KRXHolidays
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

        # 알림
        self.notifier: TelegramNotifier = TelegramNotifier()
        self.alert_manager: AlertManager = AlertManager()
        self.alert_manager.add_notifier(self.notifier)

        # 전략 (외부에서 주입 가능)
        self._strategy = None

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

            # 전략에 필요한 데이터를 수집하여 시그널 생성
            date_str = today.strftime("%Y%m%d")
            try:
                signals = self._strategy.generate_signals(date_str, {})
            except Exception as e:
                logger.warning("시그널 생성 중 오류: %s", e)
                signals = {}

            if signals:
                top_tickers = sorted(
                    signals.items(), key=lambda x: x[1], reverse=True
                )[:10]
                lines = [
                    f"[장 전 시그널 체크] {today.strftime('%Y-%m-%d')}",
                    "─" * 30,
                    f"목표 종목 수: {len(signals)}개",
                    "",
                    "Top 10 비중:",
                ]
                for ticker, weight in top_tickers:
                    lines.append(f"  {ticker}: {weight:.1%}")

                # Dry run
                dry_result = self.executor.dry_run(signals)
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

    def execute_rebalance(self) -> None:
        """09:05 - 리밸런싱 실행 (해당일만).

        리밸런싱일에만 실행되며, 전략 시그널을 기반으로
        매도 -> 매수 순서로 실제 주문을 실행한다.
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

            # 시그널 생성
            date_str = today.strftime("%Y%m%d")
            logger.info("리밸런싱 시그널 생성 중: %s", date_str)

            try:
                signals = self._strategy.generate_signals(date_str, {})
            except Exception as e:
                logger.error("시그널 생성 실패: %s", e)
                self._send_notification(
                    f"[리밸런싱 오류] 시그널 생성 실패: {e}",
                    level="CRITICAL",
                )
                return

            if not signals:
                logger.warning("생성된 시그널이 없습니다. 리밸런싱을 스킵합니다.")
                self._send_notification(
                    "[리밸런싱] 시그널이 없어 리밸런싱을 스킵합니다."
                )
                return

            # 리밸런싱 실행
            self._send_notification(
                f"[리밸런싱 시작] {today.strftime('%Y-%m-%d')}\n"
                f"종목 수: {len(signals)}개"
            )

            result = self.executor.execute_rebalance(signals)

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

    def hourly_monitor(self) -> None:
        """매시 정각 - 포트폴리오 모니터링.

        보유 종목의 이상 변동을 감지하고 알림을 발송한다.
        """
        if not self._is_trading_day():
            return

        try:
            if not self.kis_client.is_configured():
                return

            balance = self.kis_client.get_balance()
            holdings = balance.get("holdings", [])

            if not holdings:
                return

            # 이상 변동 감지: 개별 종목 일일 수익률이 +-5% 이상
            alerts = []
            holdings_daily_returns: dict[str, float] = {}

            for h in holdings:
                ticker = h.get("ticker", "")
                name = h.get("name", ticker)
                pnl_pct = h.get("pnl_pct", 0.0)

                # pnl_pct는 매입 대비 수익률이므로 참고용
                holdings_daily_returns[ticker] = pnl_pct / 100.0

                # 현재가 조회하여 일일 변동 확인
                try:
                    price_info = self.kis_client.get_current_price(ticker)
                    change_pct = price_info.get("change_pct", 0.0)

                    if abs(change_pct) >= 5.0:
                        direction = "급등" if change_pct > 0 else "급락"
                        alerts.append(
                            f"  {name}({ticker}): {change_pct:+.2f}% ({direction})"
                        )
                except Exception:
                    pass

            # AlertManager를 통한 조건 기반 알림
            state = {
                "holdings_daily_returns": holdings_daily_returns,
                "current_mdd": 0.0,  # 추후 계산 로직 추가
            }
            self.alert_manager.check_and_alert(state)

            # 급등락 직접 알림
            if alerts:
                message = (
                    f"[포트폴리오 모니터링] {datetime.now(KST).strftime('%H:%M')}\n"
                    "─" * 30 + "\n"
                    "이상 변동 감지:\n" + "\n".join(alerts)
                )
                self._send_notification(message, level="WARNING")

            now = datetime.now(KST)
            logger.info(
                "포트폴리오 모니터링 완료 (%s): %d개 보유, %d건 이상 변동",
                now.strftime("%H:%M"),
                len(holdings),
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
        logger.info("=" * 60)

        self._send_notification(
            f"[봇 시작] {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"모드: {self.kis_client.trading_mode}\n"
            f"전략: {strategy_name}\n"
            f"리밸런싱 주기: {self.rebalance_freq}"
        )

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("트레이딩 봇 종료")
        except Exception as e:
            logger.error("트레이딩 봇 예외 종료: %s", e)
            logger.debug(traceback.format_exc())
        finally:
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
            num_stocks=20,
            apply_market_timing=True,
        )
        bot.set_strategy(strategy)
    except Exception as e:
        logger.warning("기본 전략 로드 실패: %s. 전략 없이 시작합니다.", e)

    bot.run()


if __name__ == "__main__":
    main()
