"""
anomaly_types.py
- 강의 자료 14_시계열이상탐지의 분류 체계를 반영
  · 점이상(Point Anomaly): 전역(Global) / 맥락(Contextual)
  · 패턴이상(Pattern Anomaly): Shapelet / Seasonal / Trend
- 탐지된 각 이상에 대해 유형을 자동 분류
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def find_anomaly_runs(pred: np.ndarray) -> List[Dict]:
    """0/1 예측 배열에서 연속된 이상 구간을 찾아 (start, end, length) 리스트로 반환."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(pred):
        if v == 1 and not in_run:
            start = i
            in_run = True
        elif v == 0 and in_run:
            runs.append({"start": int(start), "end": int(i - 1),
                         "length": int(i - start)})
            in_run = False
    if in_run:
        runs.append({"start": int(start), "end": int(len(pred) - 1),
                     "length": int(len(pred) - start)})
    return runs


def _zscore_segment(values: np.ndarray, ref: np.ndarray) -> float:
    """ref 분포 대비 values의 평균 z-score."""
    mu, sigma = float(np.mean(ref)), float(np.std(ref) + 1e-9)
    return float(np.mean(np.abs((values - mu) / sigma)))


def classify_anomaly(series: np.ndarray,
                     run: Dict,
                     context_window: int = 50,
                     period: Optional[int] = None) -> str:
    """단일 이상 구간을 분류.
       강의 자료의 5분류 체계:
       - Global: 전체 시계열 평균에서 크게 벗어남 (run_length==1, 큰 |z|)
       - Contextual: 인접 구간 대비 크게 벗어남 (단일점, 전역 z는 작음)
       - Shapelet: 짧은 모양 이상 (2~period/2 길이)
       - Seasonal: 계절주기와 어긋남 (1주기 이상 길고, 전역 통계와 유사하지만 lag-correlation 손상)
       - Trend: 긴 구간에 걸친 추세 변화
    """
    n = len(series)
    s, e = run["start"], run["end"]
    length = run["length"]
    seg = np.asarray(series[s:e + 1], dtype=float)

    # 컨텍스트(인접 구간)
    cs = max(0, s - context_window)
    ce = min(n, e + 1 + context_window)
    context = np.concatenate([series[cs:s], series[e + 1:ce]])
    if len(context) < 5:
        context = series

    z_global = _zscore_segment(seg, series)
    z_local = _zscore_segment(seg, context) if len(context) >= 5 else z_global

    # 분류 규칙 (강의 자료의 분류 정의에 기반)
    if length == 1:
        # 점 이상
        if z_global > 3.0:
            return "Global (점이상-전역)"
        else:
            return "Contextual (점이상-맥락)"
    # 다중 시점 이상 → 패턴 이상
    p = int(period) if period and period > 1 else max(2, n // 50)
    if length >= max(p, 20):
        # 매우 긴 구간 → Trend
        # 추세 변화 추정: 구간 평균과 인접 평균의 차이
        ctx_mean = float(np.mean(context)) if len(context) else 0.0
        if abs(float(np.mean(seg)) - ctx_mean) > 0.5 * float(np.std(series) + 1e-9):
            return "Trend (패턴-추세)"
        else:
            return "Seasonal (패턴-계절)"
    if length >= max(2, p // 2):
        # 계절 주기 근처
        return "Seasonal (패턴-계절)"
    # 짧은 다중점
    return "Shapelet (패턴-모양)"


def classify_predictions(series: np.ndarray,
                         pred: np.ndarray,
                         period: Optional[int] = None) -> pd.DataFrame:
    """다변량의 경우 첫 번째 컬럼 기준으로 분류 (또는 평균 합산).
       반환: DataFrame(start_idx, end_idx, length, type, mean_z_global, mean_z_local)."""
    if series.ndim > 1:
        s = np.mean(series, axis=1)
    else:
        s = series.astype(float)
    runs = find_anomaly_runs(np.asarray(pred).astype(int))
    rows = []
    n = len(s)
    for r in runs:
        rtype = classify_anomaly(s, r, period=period)
        seg = s[r["start"]:r["end"] + 1]
        cs = max(0, r["start"] - 50)
        ce = min(n, r["end"] + 1 + 50)
        context = np.concatenate([s[cs:r["start"]], s[r["end"] + 1:ce]])
        rows.append({
            "start_idx": r["start"],
            "end_idx": r["end"],
            "length": r["length"],
            "type": rtype,
            "mean_z_global": _zscore_segment(seg, s),
            "mean_z_local": _zscore_segment(seg, context) if len(context) >= 5 else 0.0,
        })
    return pd.DataFrame(rows)


def summarize_types(class_df: pd.DataFrame) -> Dict[str, int]:
    """유형별 카운트 요약."""
    cats = ["Global (점이상-전역)", "Contextual (점이상-맥락)",
            "Shapelet (패턴-모양)", "Seasonal (패턴-계절)", "Trend (패턴-추세)"]
    return {c: int((class_df["type"] == c).sum()) if not class_df.empty else 0 for c in cats}
