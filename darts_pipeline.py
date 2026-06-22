"""
darts_pipeline.py
- 강의 자료 14_시계열이상탐지의 Darts 4단 구조를 그대로 구현
  1) Scorers  : KMeansScorer, NormScorer, WassersteinScorer (다양한 이상 점수)
  2) Detectors: 점수 → 이진 라벨 (Threshold / Quantile)
  3) Aggregators: 여러 이진 예측 → 단일 이진 예측 (Or / And / Majority)
  4) Anomaly Model: ForecastingAnomalyModel (SKLearnModel + 캘린더 공변량)
- darts 미설치 환경에서도 동작하도록 폴백 구현 포함
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------
# Scorers (이상 점수 계산)
# --------------------------------------------------------------
def scorer_kmeans(X: np.ndarray, k: int = 2, window: int = 10) -> np.ndarray:
    """KMeans 군집 중심까지 거리 기반 점수 (Darts KMeansScorer 유사)."""
    from sklearn.cluster import KMeans
    Xs = StandardScaler().fit_transform(X)
    n, d = Xs.shape
    w = max(1, min(int(window), max(1, n // 20)))
    if w == 1:
        emb = Xs
    else:
        emb = np.zeros((n, d * w))
        for i in range(n):
            start = max(0, i - w + 1)
            chunk = Xs[start:i + 1]
            if len(chunk) < w:
                chunk = np.vstack([np.tile(Xs[0], (w - len(chunk), 1)), chunk])
            emb[i] = chunk.reshape(-1)
    km = KMeans(n_clusters=max(2, k), n_init=10, random_state=42).fit(emb)
    dists = np.min(np.linalg.norm(emb[:, None, :] - km.cluster_centers_[None, :, :], axis=2),
                   axis=1)
    return dists


def scorer_norm(X: np.ndarray, baseline: Optional[np.ndarray] = None) -> np.ndarray:
    """NormScorer 유사: 기준값(또는 평균) 대비 L2 거리."""
    Xs = StandardScaler().fit_transform(X)
    if baseline is None:
        baseline = np.zeros(Xs.shape[1])
    return np.linalg.norm(Xs - baseline, axis=1)


def scorer_wasserstein(X: np.ndarray, window: int = 30) -> np.ndarray:
    """슬라이딩 윈도우 vs 전체 분포의 Wasserstein-1 거리 기반 점수
       (Darts WassersteinScorer 유사). 분포 변화 감지에 효과적."""
    from scipy.stats import wasserstein_distance
    Xs = StandardScaler().fit_transform(X)
    n, d = Xs.shape
    w = max(5, min(int(window), max(5, n // 10)))
    scores = np.zeros(n)
    for j in range(d):
        ref = Xs[:, j]
        ref_sample = ref  # 전체 분포를 reference로
        for i in range(n):
            lo, hi = max(0, i - w + 1), i + 1
            if hi - lo < 3:
                continue
            scores[i] += wasserstein_distance(ref[lo:hi], ref_sample)
    return scores / max(1, d)


def scorer_residual_norm(series: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """예측-실제 잔차의 L2 norm 점수."""
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    if forecast.ndim == 1:
        forecast = forecast.reshape(-1, 1)
    return np.linalg.norm(series - forecast, axis=1)


# --------------------------------------------------------------
# Detectors (이상 점수 → 0/1 이진 라벨)
# --------------------------------------------------------------
def detector_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """고정 임계값 기반 이진화."""
    return (np.asarray(scores) >= threshold).astype(int)


def detector_quantile(scores: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """상위 분위수 기반 이진화 (Darts QuantileDetector 유사)."""
    q = float(np.clip(quantile, 0.5, 0.9999))
    thr = float(np.quantile(scores, q))
    return (np.asarray(scores) >= thr).astype(int)


# --------------------------------------------------------------
# Aggregators (여러 이진 예측 → 단일 이진 예측)
# --------------------------------------------------------------
def aggregator_or(preds: List[np.ndarray]) -> np.ndarray:
    """OR 집계: 하나라도 이상이면 이상."""
    if not preds:
        return np.zeros(0, dtype=int)
    arr = np.vstack(preds)
    return (arr.max(axis=0) >= 1).astype(int)


def aggregator_and(preds: List[np.ndarray]) -> np.ndarray:
    """AND 집계: 모두가 이상으로 판정해야 이상."""
    if not preds:
        return np.zeros(0, dtype=int)
    arr = np.vstack(preds)
    return (arr.min(axis=0) >= 1).astype(int)


def aggregator_majority(preds: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """다수결: threshold 이상 비율로 이상 판정."""
    if not preds:
        return np.zeros(0, dtype=int)
    arr = np.vstack(preds)
    return ((arr.mean(axis=0)) >= float(threshold)).astype(int)


# --------------------------------------------------------------
# Forecasting Anomaly Model
# --------------------------------------------------------------
def forecasting_anomaly_model(ts_df: pd.DataFrame,
                              lags: int = 24,
                              add_calendar_covariates: bool = True,
                              train_ratio: float = 0.6,
                              use_wasserstein: bool = True,
                              auto_lags: bool = True,
                              ) -> Dict[str, Any]:
    """강의 14주차 예제와 동일한 구조:
       ForecastingAnomalyModel(model=SKLearnModel(lags=..., lags_future_covariates=[0],
       add_encoders={'cyclic': {'future': ['hour', 'dayofweek']}}),
       scorer=[NormScorer(), KMeansScorer(), WassersteinScorer()])

       강의 11주차 백테스팅: train_ratio로 학습 구간 비율을 조정.
       auto_lags=True 시 강의 p.9의 'one_week = 7*24*2 = 336' (30분 빈도)을 자동 적용.

       반환: {
          'scores_norm': np.ndarray,
          'scores_kmeans': np.ndarray,
          'scores_wasserstein': np.ndarray (optional),
          'fitted': bool,
          'train_n': int,
          'lags_used': int,
       }
    """
    X = ts_df.values.astype(float)
    n = X.shape[0]
    out: Dict[str, Any] = {"fitted": False}
    try:
        from darts import TimeSeries
        from darts.ad import ForecastingAnomalyModel, NormScorer, KMeansScorer
        try:
            from darts.ad import WassersteinScorer  # 일부 버전에서만 존재
            HAS_WASS = True
        except Exception:
            HAS_WASS = False
        from darts.models import LinearRegressionModel
    except Exception:
        out["scores_norm"] = scorer_norm(X)
        out["scores_kmeans"] = scorer_kmeans(X)
        if use_wasserstein:
            out["scores_wasserstein"] = scorer_wasserstein(X)
        return out

    if not isinstance(ts_df.index, pd.DatetimeIndex):
        out["scores_norm"] = scorer_norm(X)
        out["scores_kmeans"] = scorer_kmeans(X)
        if use_wasserstein:
            out["scores_wasserstein"] = scorer_wasserstein(X)
        return out

    # 강의 p.9의 자동 lags 설정: 30분 빈도면 one_week = 7*24*2 = 336
    if auto_lags:
        try:
            inferred = pd.infer_freq(ts_df.index)
        except Exception:
            inferred = None
        if inferred:
            if inferred.startswith("30T") or inferred.startswith("30min"):
                lags = 7 * 24 * 2  # 336 (강의록 동일)
            elif inferred.upper().startswith("H") or inferred.lower().startswith("h"):
                lags = 7 * 24      # 일주일 (시간 빈도)
            elif inferred.upper().startswith("D"):
                lags = 14          # 2주
        # 데이터 길이에 비해 너무 큰 lags는 줄임
        lags = max(2, min(int(lags), max(2, n // 5)))

    try:
        df_in = ts_df.copy()
        if df_in.index.name is None:
            df_in.index.name = "time"
        df_in = df_in.reset_index()
        ts = TimeSeries.from_dataframe(
            df_in, time_col=df_in.columns[0],
            value_cols=list(ts_df.columns),
        )
    except Exception:
        try:
            ts = TimeSeries.from_values(X)
        except Exception:
            out["scores_norm"] = scorer_norm(X)
            out["scores_kmeans"] = scorer_kmeans(X)
            if use_wasserstein:
                out["scores_wasserstein"] = scorer_wasserstein(X)
            return out

    # train/test 분리 (강의 11주차 백테스팅 — 사용자 지정 train_ratio)
    try:
        n_train = max(min(int(float(train_ratio) * n), n - 50), lags + 5)
        out["train_n"] = int(n_train)
        out["lags_used"] = int(lags)
        ts_train = ts[:n_train]

        add_encoders = None
        if add_calendar_covariates and isinstance(ts_df.index, pd.DatetimeIndex):
            add_encoders = {"cyclic": {"future": ["hour", "dayofweek"]}}

        try:
            forecaster = LinearRegressionModel(
                lags=lags,
                lags_future_covariates=[0] if add_encoders else None,
                output_chunk_length=1,
                add_encoders=add_encoders,
            )
            forecaster.fit(ts_train)
        except Exception:
            forecaster = LinearRegressionModel(lags=lags, output_chunk_length=1)
            forecaster.fit(ts_train)

        # 강의 p.10·p.13: NormScorer + KMeansScorer + (WassersteinScorer)
        scorer_list = [NormScorer(), KMeansScorer(k=2)]
        scorer_names = ["norm", "kmeans"]
        if use_wasserstein and HAS_WASS:
            try:
                scorer_list.append(WassersteinScorer(window=10, component_wise=False))
                scorer_names.append("wasserstein")
            except Exception:
                pass

        am = ForecastingAnomalyModel(model=forecaster, scorer=scorer_list)
        am.fit(ts_train, allow_model_training=False)
        scores_out = am.score(ts)
        if isinstance(scores_out, (list, tuple)):
            scores_list = list(scores_out)
        else:
            scores_list = [scores_out]
        flat = []
        for it in scores_list:
            if isinstance(it, (list, tuple)):
                flat.extend(list(it))
            else:
                flat.append(it)
        scores_list = flat

        def _to_array(s):
            try:
                return s.values().flatten()
            except Exception:
                return np.asarray(s).flatten()

        for idx, sname in enumerate(scorer_names):
            if idx >= len(scores_list):
                continue
            arr = _to_array(scores_list[idx])
            pad = np.zeros(n)
            pad[n - len(arr):] = arr
            out[f"scores_{sname}"] = pad

        # 호환성: scores_norm/scores_kmeans 키가 반드시 존재하도록 폴백
        if "scores_norm" not in out:
            out["scores_norm"] = scorer_norm(X)
        if "scores_kmeans" not in out:
            out["scores_kmeans"] = scorer_kmeans(X)

        out["fitted"] = True
        return out
    except Exception as e:
        out["scores_norm"] = scorer_norm(X)
        out["scores_kmeans"] = scorer_kmeans(X)
        if use_wasserstein:
            out["scores_wasserstein"] = scorer_wasserstein(X)
        out["error"] = str(e)
        return out


# --------------------------------------------------------------
# Aggregator 카탈로그
# --------------------------------------------------------------
AGGREGATORS = {
    "Or (하나라도 이상)": aggregator_or,
    "And (모두 이상)": aggregator_and,
    "Majority (다수결)": aggregator_majority,
}
