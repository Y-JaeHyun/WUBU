"""ML 피처 엔지니어링 모듈.

팩터 투자에 활용할 수 있는 다양한 피처를 생성하고,
횡단면 정규화, 타겟(포워드 수익률) 생성 기능을 제공한다.

피처 카테고리:
- 밸류: inv_pbr, inv_per, div_yield
- 모멘텀: mom_12m, mom_6m, mom_3m (1개월 스킵)
- 퀄리티: roe, gpa
- 리스크: vol_20d, vol_60d
- 시장: log_market_cap
- 유동성: volume_ratio

사용 예시:
    features = build_factor_features(fundamentals, prices)
    targets = build_forward_returns(prices, date, forward_days=21)
    normalized = cross_sectional_normalize(features)
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_factor_features(
    fundamentals: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """팩터 기반 피처 DataFrame을 생성한다.

    펀더멘탈 데이터와 가격 데이터로부터 다양한 팩터 피처를 산출한다.

    피처 목록:
    - inv_pbr: PBR 역수 (높을수록 저평가)
    - inv_per: PER 역수 (높을수록 저평가)
    - div_yield: 배당수익률
    - mom_12m: 12개월 모멘텀 (1개월 스킵)
    - mom_6m: 6개월 모멘텀 (1개월 스킵)
    - mom_3m: 3개월 모멘텀 (1개월 스킵)
    - roe: 자기자본이익률
    - gpa: 매출총이익/총자산 (추정)
    - vol_20d: 20일 변동성
    - vol_60d: 60일 변동성
    - log_market_cap: 로그 시가총액
    - volume_ratio: 최근 5일 거래량 / 20일 평균 거래량

    Args:
        fundamentals: 전종목 기본 지표 DataFrame
            필수 컬럼: ticker
            선택 컬럼: pbr, per, eps, bps, div_yield, market_cap, roe, ...
        prices: {종목코드: OHLCV DataFrame} 딕셔너리

    Returns:
        피처 DataFrame (index=ticker, columns=feature_names).
        데이터가 없으면 빈 DataFrame 반환.
    """
    if fundamentals.empty or "ticker" not in fundamentals.columns:
        logger.warning("빈 펀더멘탈 데이터: 빈 피처 반환")
        return pd.DataFrame()

    df = fundamentals.copy()
    tickers = df["ticker"].tolist()
    feature_data = {}

    # --- 밸류 피처 ---
    # inv_pbr
    if "pbr" in df.columns:
        pbr_values = df.set_index("ticker")["pbr"]
        inv_pbr = pd.Series(np.nan, index=tickers)
        valid_pbr = pbr_values[pbr_values > 0]
        if not valid_pbr.empty:
            inv_pbr.loc[valid_pbr.index] = 1.0 / valid_pbr
        feature_data["inv_pbr"] = inv_pbr

    # inv_per
    if "per" in df.columns:
        per_values = df.set_index("ticker")["per"]
        inv_per = pd.Series(np.nan, index=tickers)
        valid_per = per_values[per_values > 0]
        if not valid_per.empty:
            inv_per.loc[valid_per.index] = 1.0 / valid_per
        feature_data["inv_per"] = inv_per

    # div_yield
    if "div_yield" in df.columns:
        feature_data["div_yield"] = df.set_index("ticker")["div_yield"]

    # --- 퀄리티 피처 ---
    # roe
    if "roe" in df.columns:
        feature_data["roe"] = df.set_index("ticker")["roe"]
    elif "eps" in df.columns and "bps" in df.columns:
        eps = df.set_index("ticker")["eps"]
        bps = df.set_index("ticker")["bps"].replace(0, np.nan)
        feature_data["roe"] = (eps / bps * 100)

    # gpa (매출총이익/총자산)
    if "gpa" in df.columns:
        feature_data["gpa"] = df.set_index("ticker")["gpa"]
    elif "gp_over_assets" in df.columns:
        feature_data["gpa"] = df.set_index("ticker")["gp_over_assets"]
    elif "roe" in df.columns:
        # ROE 기반 대략적 추정 (fallback)
        feature_data["gpa"] = df.set_index("ticker")["roe"].abs() / 200

    # --- 시장 피처 ---
    # log_market_cap
    if "market_cap" in df.columns:
        mc = df.set_index("ticker")["market_cap"]
        mc_positive = mc[mc > 0]
        log_mc = pd.Series(np.nan, index=tickers)
        if not mc_positive.empty:
            log_mc.loc[mc_positive.index] = np.log(mc_positive)
        feature_data["log_market_cap"] = log_mc

    # --- 가격 기반 피처 (모멘텀, 변동성, 거래량) ---
    mom_12m = {}
    mom_6m = {}
    mom_3m = {}
    vol_20d = {}
    vol_60d = {}
    volume_ratio = {}

    skip_days = 21  # 1개월 스킵

    for ticker in tickers:
        if ticker not in prices:
            continue

        price_df = prices[ticker]
        if price_df.empty or "close" not in price_df.columns:
            continue

        close = price_df["close"].dropna()
        if len(close) < skip_days + 5:
            continue

        # 스킵 적용: 직전 1개월 제외
        close_skipped = close.iloc[:-skip_days] if len(close) > skip_days else close

        # 모멘텀 피처 (1개월 스킵 적용)
        if len(close_skipped) >= 252:
            mom_12m[ticker] = float(
                close_skipped.iloc[-1] / close_skipped.iloc[-252] - 1
            )
        if len(close_skipped) >= 126:
            mom_6m[ticker] = float(
                close_skipped.iloc[-1] / close_skipped.iloc[-126] - 1
            )
        if len(close_skipped) >= 63:
            mom_3m[ticker] = float(
                close_skipped.iloc[-1] / close_skipped.iloc[-63] - 1
            )

        # 변동성 피처 (스킵 없이 최근 데이터 사용)
        daily_returns = close.pct_change().dropna()

        if len(daily_returns) >= 20:
            vol_20d[ticker] = float(
                daily_returns.iloc[-20:].std() * np.sqrt(252)
            )
        if len(daily_returns) >= 60:
            vol_60d[ticker] = float(
                daily_returns.iloc[-60:].std() * np.sqrt(252)
            )

        # 거래량 비율 피처
        if "volume" in price_df.columns:
            vol_data = price_df["volume"].dropna()
            if len(vol_data) >= 20:
                recent_5d = vol_data.iloc[-5:].mean()
                avg_20d = vol_data.iloc[-20:].mean()
                if avg_20d > 0:
                    volume_ratio[ticker] = float(recent_5d / avg_20d)

    # 가격 기반 피처를 딕셔너리에 추가
    if mom_12m:
        feature_data["mom_12m"] = pd.Series(mom_12m)
    if mom_6m:
        feature_data["mom_6m"] = pd.Series(mom_6m)
    if mom_3m:
        feature_data["mom_3m"] = pd.Series(mom_3m)
    if vol_20d:
        feature_data["vol_20d"] = pd.Series(vol_20d)
    if vol_60d:
        feature_data["vol_60d"] = pd.Series(vol_60d)
    if volume_ratio:
        feature_data["volume_ratio"] = pd.Series(volume_ratio)

    if not feature_data:
        logger.warning("피처 생성 실패: 빈 DataFrame 반환")
        return pd.DataFrame()

    # DataFrame 생성
    features_df = pd.DataFrame(feature_data)
    features_df.index.name = "ticker"

    logger.info(
        f"팩터 피처 생성 완료: {len(features_df)}개 종목, "
        f"{len(features_df.columns)}개 피처 ({list(features_df.columns)})"
    )

    return features_df


def build_forward_returns(
    prices: dict[str, pd.DataFrame],
    date: str,
    forward_days: int = 21,
) -> pd.Series:
    """지정 날짜 기준 포워드 수익률(타겟)을 산출한다.

    리밸런싱 날짜 이후 forward_days 거래일 동안의 수익률을 계산한다.
    ML 모델의 타겟(종속변수)으로 사용한다.

    Args:
        prices: {종목코드: OHLCV DataFrame} 딕셔너리
        date: 기준 날짜 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        forward_days: 포워드 수익률 기간 (기본 21 = 약 1개월)

    Returns:
        Series (index=ticker, values=forward_return).
        데이터가 없으면 빈 Series 반환.
    """
    if not prices:
        logger.warning("빈 가격 데이터: 빈 포워드 수익률 반환")
        return pd.Series(dtype=float)

    # 날짜 파싱
    try:
        ref_date = pd.Timestamp(date)
    except Exception:
        logger.warning(f"날짜 파싱 실패: {date}")
        return pd.Series(dtype=float)

    forward_returns = {}

    for ticker, price_df in prices.items():
        if price_df.empty or "close" not in price_df.columns:
            continue

        close = price_df["close"].dropna()
        if close.empty:
            continue

        # 기준 날짜 이후 데이터 추출
        future_data = close[close.index >= ref_date]

        if len(future_data) < forward_days + 1:
            continue

        # 기준일 종가와 forward_days 후 종가로 수익률 계산
        start_price = future_data.iloc[0]
        end_price = future_data.iloc[forward_days]

        if start_price > 0:
            forward_returns[ticker] = float(end_price / start_price - 1)

    if not forward_returns:
        logger.warning(f"포워드 수익률 계산 불가 ({date}): 빈 Series 반환")
        return pd.Series(dtype=float)

    result = pd.Series(forward_returns, dtype=float)
    result.index.name = "ticker"

    logger.info(
        f"포워드 수익률 산출 ({date}): {len(result)}개 종목, "
        f"평균={result.mean():.4f}, 중앙값={result.median():.4f}"
    )

    return result


def cross_sectional_normalize(features: pd.DataFrame) -> pd.DataFrame:
    """횡단면 Z-Score 정규화 + Winsorizing을 수행한다.

    각 피처(열)별로 전 종목(행)에 대해 Z-Score 정규화한 후,
    상위/하위 1% 이상치를 Winsorizing한다.

    Args:
        features: 피처 DataFrame (index=ticker, columns=feature_names)

    Returns:
        정규화된 피처 DataFrame.
        빈 입력이면 빈 DataFrame 반환.
    """
    if features.empty:
        logger.warning("빈 피처 데이터: 빈 정규화 결과 반환")
        return pd.DataFrame()

    normalized = features.copy()

    for col in normalized.columns:
        series = normalized[col].dropna()

        if len(series) < 3:
            # 데이터 부족 시 0으로 대체
            normalized[col] = 0.0
            continue

        # Z-Score 정규화
        mean_val = series.mean()
        std_val = series.std()

        if std_val == 0 or np.isnan(std_val):
            normalized[col] = 0.0
            continue

        z_scores = (normalized[col] - mean_val) / std_val

        # Winsorizing (상하 1%)
        lower = z_scores.quantile(0.01)
        upper = z_scores.quantile(0.99)
        z_scores = z_scores.clip(lower=lower, upper=upper)

        normalized[col] = z_scores

    # 남은 NaN은 0으로 채움
    normalized = normalized.fillna(0.0)

    logger.info(
        f"횡단면 정규화 완료: {len(normalized)}개 종목, "
        f"{len(normalized.columns)}개 피처"
    )

    return normalized
