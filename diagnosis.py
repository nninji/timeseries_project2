"""
diagnosis.py
- 시계열 데이터의 통계적 특성을 자동 진단
- 강의 자료(06_시계열데이터전처리, 08_평활법과분해법, 09_ARIMA) 반영
- 진단 결과를 바탕으로 적합한 이상탐지 알고리즘을 추천
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox


def infer_freq(index: pd.Index) -> Tuple[Optional[str], Optional[int]]:
    """시간 인덱스로부터 frequency와 1주기 길이(예: 24=일주기)를 추정."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None, None
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    # 빈도 기반 1주기 길이 추정 (대표적 계절성)
    if freq is None:
        deltas = np.diff(index.values).astype("timedelta64[s]").astype(int)
        median_dt = float(np.median(deltas)) if len(deltas) else 0
        if median_dt <= 0:
            return None, None
        # 일주기(24h) 기준으로 한 주기 sample 수
        period = max(2, int(round(86400.0 / median_dt)))
        if period > len(index) // 3:
            period = max(2, len(index) // 10)
        return None, period
    mapping = {
        "T": 60 * 24, "min": 60 * 24,
        "5T": 12 * 24, "10T": 6 * 24, "15T": 4 * 24, "30T": 2 * 24,
        "H": 24, "h": 24,
        "D": 7, "B": 5,
        "W": 52,
        "M": 12, "MS": 12,
        "Q": 4, "QS": 4,
        "Y": 1, "YS": 1, "A": 1,
    }
    # freq 문자열에서 첫 글자 매칭
    period = None
    for k in sorted(mapping.keys(), key=lambda x: -len(x)):
        if freq.startswith(k):
            period = mapping[k]
            break
    if period is None:
        period = 12
    return freq, period


def adf_test(series: np.ndarray) -> Dict[str, float]:
    """Augmented Dickey-Fuller 정상성 검정.
       H0: 단위근 존재(비정상) / p<0.05면 정상 시계열."""
    try:
        s = pd.Series(series).dropna().values
        if len(s) < 10 or np.std(s) < 1e-12:
            return {"statistic": float("nan"), "p_value": float("nan"),
                    "is_stationary": False, "n": int(len(s))}
        res = adfuller(s, autolag="AIC")
        return {
            "statistic": float(res[0]),
            "p_value": float(res[1]),
            "is_stationary": bool(res[1] < 0.05),
            "n": int(len(s)),
        }
    except Exception as e:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "is_stationary": False, "n": 0, "error": str(e)}


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Dict[str, float]:
    """Ljung-Box 백색잡음 검정.
       H0: 자기상관이 0(백색잡음) / p<0.05면 자기상관 존재(패턴 잔존)."""
    try:
        s = pd.Series(residuals).dropna()
        if len(s) < lags + 5:
            return {"statistic": float("nan"), "p_value": float("nan"),
                    "is_white_noise": False}
        out = acorr_ljungbox(s, lags=[min(lags, len(s) // 5)], return_df=True)
        stat = float(out["lb_stat"].iloc[0])
        p = float(out["lb_pvalue"].iloc[0])
        return {"statistic": stat, "p_value": p, "is_white_noise": bool(p >= 0.05)}
    except Exception as e:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "is_white_noise": False, "error": str(e)}


def compute_acf_pacf(series: np.ndarray, nlags: int = 40) -> Dict[str, np.ndarray]:
    """ACF / PACF 계산."""
    s = pd.Series(series).dropna().values
    nlags = max(5, min(nlags, len(s) // 3))
    try:
        a = acf(s, nlags=nlags, fft=True)
        p = pacf(s, nlags=nlags, method="ywm")
        ci = 1.96 / np.sqrt(len(s))
        return {"acf": a, "pacf": p, "ci": float(ci), "lags": np.arange(nlags + 1)}
    except Exception:
        return {"acf": np.zeros(1), "pacf": np.zeros(1),
                "ci": 0.0, "lags": np.arange(1)}


def stl_decompose(series: np.ndarray, period: int) -> Optional[Dict[str, np.ndarray]]:
    """STL 분해 (추세-계절성-잔차)."""
    period = max(2, int(period))
    s = np.asarray(series, dtype=float)
    if len(s) < 2 * period + 5:
        return None
    try:
        stl = STL(s, period=period, robust=True).fit()
        return {
            "observed": s,
            "trend": np.asarray(stl.trend),
            "seasonal": np.asarray(stl.seasonal),
            "resid": np.asarray(stl.resid),
        }
    except Exception:
        return None


def seasonal_strength(decomp: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """STL 분해 결과로부터 계절성 강도, 추세 강도를 0~1 스케일로 계산.
       Hyndman & Athanasopoulos(2018) 공식."""
    resid = decomp["resid"]
    seas = decomp["seasonal"]
    trend = decomp["trend"]
    var_resid = float(np.var(resid))
    var_resid_seas = float(np.var(resid + seas))
    var_resid_trend = float(np.var(resid + trend))
    fs = max(0.0, 1.0 - var_resid / max(var_resid_seas, 1e-9))
    ft = max(0.0, 1.0 - var_resid / max(var_resid_trend, 1e-9))
    return float(min(fs, 1.0)), float(min(ft, 1.0))


def diagnose_series(series: np.ndarray, period: Optional[int]) -> Dict:
    """시계열의 통계적 특성을 종합 진단."""
    out: Dict = {}
    s = pd.Series(series).dropna().values
    out["n"] = int(len(s))
    out["mean"] = float(np.mean(s))
    out["std"] = float(np.std(s))
    out["min"] = float(np.min(s))
    out["max"] = float(np.max(s))
    out["adf"] = adf_test(s)

    # STL 분해
    decomp = stl_decompose(s, period=period or 12)
    out["decomp"] = decomp
    if decomp is not None:
        fs, ft = seasonal_strength(decomp)
        out["seasonal_strength"] = fs
        out["trend_strength"] = ft
        out["ljung_box_resid"] = ljung_box_test(decomp["resid"])
    else:
        out["seasonal_strength"] = float("nan")
        out["trend_strength"] = float("nan")
        out["ljung_box_resid"] = {"statistic": float("nan"), "p_value": float("nan"),
                                   "is_white_noise": False}
    out["acf_pacf"] = compute_acf_pacf(s)
    return out


def diagnose_dataframe(ts_df: pd.DataFrame, period: Optional[int]) -> Dict[str, Dict]:
    """다변량 시계열의 변수별 진단."""
    return {col: diagnose_series(ts_df[col].values, period=period) for col in ts_df.columns}


# ============================================================
# 알고리즘 추천
# ============================================================
def recommend_algorithms(diagnoses: Dict[str, Dict]) -> List[Tuple[str, str]]:
    """진단 결과를 바탕으로 추천 알고리즘과 이유를 반환.
       (algorithm_name, reason)의 리스트."""
    recs: List[Tuple[str, str]] = []

    if not diagnoses:
        return recs

    # 집계 통계
    n_vars = len(diagnoses)
    avg_seas = float(np.nanmean([d.get("seasonal_strength", np.nan) for d in diagnoses.values()]))
    avg_trend = float(np.nanmean([d.get("trend_strength", np.nan) for d in diagnoses.values()]))
    p_stationary = float(np.mean([d.get("adf", {}).get("is_stationary", False)
                                   for d in diagnoses.values()]))

    # 계절성 강함 → STL/Forecasting/Darts
    if avg_seas >= 0.5:
        recs.append(("STL 잔차 (통계)",
                     f"계절성 강도 {avg_seas:.2f} → 계절성 분해 후 잔차 기반이 효과적"))
        recs.append(("Darts Forecasting (혼합)",
                     "계절성 강함 → SKLearnModel + 캘린더 공변량(hour/dayofweek)으로 예측 후 NormScorer"))

    # 추세 강함 → STL/Forecasting (안 비정상성)
    if avg_trend >= 0.5:
        recs.append(("Darts Forecasting (혼합)",
                     f"추세 강도 {avg_trend:.2f} → 예측 잔차 방식이 비정상성에 강건"))

    # 비정상 시계열 비율 높음 → 예측 기반/딥러닝
    if p_stationary < 0.5:
        recs.append(("Autoencoder (딥러닝)",
                     "비정상 시계열 다수 → 비선형 패턴 학습 가능한 오토인코더 권장"))

    # 다변량
    if n_vars >= 3:
        recs.append(("Isolation Forest (ML)",
                     f"다변량({n_vars}개) → 고차원에서 효율적인 격리 기반 탐지"))
        recs.append(("Local Outlier Factor (ML)",
                     "다변량 → 지역 밀도 기반으로 군집 이상에 효과적"))

    # 정상 시계열 비율 높음 + 단변량 가까움 → 통계
    if p_stationary >= 0.7:
        recs.append(("Z-Score (통계)",
                     f"정상 시계열({p_stationary*100:.0f}%) → 단순 통계 기반 베이스라인 적합"))
        recs.append(("MAD (통계, Robust)",
                     "정상성 충족 시 Robust 통계 방법이 outlier에 강건"))

    # 베이스라인은 항상 추가
    if not any(r[0] == "Isolation Forest (ML)" for r in recs):
        recs.append(("Isolation Forest (ML)",
                     "범용 베이스라인 — 데이터 특성에 무관하게 우수한 성능"))

    # 중복 제거 (유지 순서)
    seen = set()
    uniq = []
    for name, reason in recs:
        if name not in seen:
            uniq.append((name, reason))
            seen.add(name)
    return uniq
