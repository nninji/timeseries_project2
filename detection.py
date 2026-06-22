"""
detection.py
- 통계적, 머신러닝, 딥러닝 기반 시계열 이상탐지 알고리즘 모음
- 강의 10주차의 4종 스케일러(Standard/Robust/MinMax/Power) 지원
- 모든 detector는 동일한 인터페이스를 따름:
    fit_predict(X: np.ndarray, contamination: float, scaler: str) -> dict:
        {
            "scores": np.ndarray (shape=(n,)),  # 이상 점수 (높을수록 이상)
            "pred":   np.ndarray (shape=(n,)),  # 0/1 예측 라벨
            "threshold": float,                  # 사용된 임계값
        }
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer,
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from scipy import stats


# ============================================================
# 강의 10주차의 4종 스케일러
# ============================================================
def get_scaler(name: str = "standard"):
    """강의 10주차 Darts의 스케일링 클래스 4종.
       - standard: 평균 0, 분산 1 (가장 일반적)
       - robust:   중앙값 0, 분산 1 (이상치에 강건)
       - minmax:   [0, 1] 범위로 정규화
       - power:    Yeo-Johnson 변환 (정규분포에 가깝게)
    """
    name = (name or "standard").lower()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "power":
        return PowerTransformer(method="yeo-johnson")
    return StandardScaler()


SCALER_NAMES = ["standard", "robust", "minmax", "power"]


def _to_2d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _binarize(scores: np.ndarray, contamination: float) -> tuple:
    """상위 contamination 비율을 이상으로 표시."""
    contamination = float(np.clip(contamination, 1e-4, 0.5))
    thr = float(np.quantile(scores, 1 - contamination))
    pred = (scores >= thr).astype(int)
    return pred, thr


# ============================================================
# 1. 통계적 방법
# ============================================================

def detect_zscore(X, contamination=0.05, scaler="standard", **kwargs) -> Dict[str, Any]:
    """다변량 Z-score (각 변수의 |z| 평균을 점수로). 선택한 스케일러 적용."""
    X = _to_2d(X)
    Xs = get_scaler(scaler).fit_transform(X)
    scores = np.mean(np.abs(Xs), axis=1)
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


def detect_mad(X, contamination=0.05, **kwargs) -> Dict[str, Any]:
    """Median Absolute Deviation 기반 이상탐지 (Robust)."""
    X = _to_2d(X)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-9
    # 1.4826: 정규분포에서 MAD->stddev 변환 상수
    z = np.abs(X - med) / (1.4826 * mad)
    scores = np.mean(z, axis=1)
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


def detect_iqr(X, contamination=0.05, **kwargs) -> Dict[str, Any]:
    """IQR(Interquartile Range) 기반 이상탐지."""
    X = _to_2d(X)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1 + 1e-9
    # 0이면 정상, 클수록 이상
    dist = np.maximum(np.maximum(q1 - X, X - q3), 0) / iqr
    scores = np.mean(dist, axis=1)
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


def detect_stl_residual(X, contamination=0.05, period=None, **kwargs) -> Dict[str, Any]:
    """STL(Seasonal-Trend Decomposition) 잔차 기반 이상탐지.
       다변량이면 각 변수에 대해 STL 분해 후 |residual|의 평균을 점수로 사용."""
    from statsmodels.tsa.seasonal import STL
    X = _to_2d(X)
    n, d = X.shape
    if period is None or period < 2:
        period = max(7, min(n // 4, 24))
    period = int(period)
    if period < 2:
        period = 2
    resid_all = np.zeros_like(X)
    for j in range(d):
        try:
            stl = STL(X[:, j], period=period, robust=True).fit()
            resid_all[:, j] = stl.resid
        except Exception:
            resid_all[:, j] = X[:, j] - np.mean(X[:, j])
    # 표준화된 잔차의 절대값
    sigma = np.std(resid_all, axis=0) + 1e-9
    scores = np.mean(np.abs(resid_all) / sigma, axis=1)
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


# ============================================================
# 2. 머신러닝 방법
# ============================================================

def detect_isolation_forest(X, contamination=0.05, random_state=42, scaler="standard", **kwargs) -> Dict[str, Any]:
    X = _to_2d(X)
    Xs = get_scaler(scaler).fit_transform(X)
    model = IsolationForest(contamination=contamination,
                            random_state=random_state, n_estimators=200)
    model.fit(Xs)
    # 낮을수록 이상이므로 부호 반전
    scores = -model.score_samples(Xs)
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


def detect_ocsvm(X, contamination=0.05, scaler="standard", **kwargs) -> Dict[str, Any]:
    X = _to_2d(X)
    Xs = get_scaler(scaler).fit_transform(X)
    nu = float(np.clip(contamination, 1e-3, 0.5))
    model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    model.fit(Xs)
    scores = -model.score_samples(Xs)
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


def detect_lof(X, contamination=0.05, scaler="standard", **kwargs) -> Dict[str, Any]:
    X = _to_2d(X)
    Xs = get_scaler(scaler).fit_transform(X)
    n_neighbors = max(5, min(35, len(X) // 20))
    model = LocalOutlierFactor(n_neighbors=n_neighbors,
                               contamination=contamination, novelty=False)
    model.fit_predict(Xs)
    scores = -model.negative_outlier_factor_
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


def detect_kmeans_scorer(X, contamination=0.05, k=2, window=10, scaler="standard", **kwargs) -> Dict[str, Any]:
    """슬라이딩 윈도우 + KMeans 거리 기반 (Darts의 KMeansScorer와 유사한 발상)."""
    from sklearn.cluster import KMeans
    X = _to_2d(X)
    Xs = get_scaler(scaler).fit_transform(X)
    n, d = Xs.shape
    w = max(1, min(int(window), max(1, n // 20)))
    # 윈도우 임베딩
    if w == 1:
        emb = Xs
    else:
        emb = np.zeros((n, d * w))
        for i in range(n):
            start = max(0, i - w + 1)
            chunk = Xs[start:i + 1]
            if len(chunk) < w:
                pad = np.tile(Xs[0], (w - len(chunk), 1))
                chunk = np.vstack([pad, chunk])
            emb[i] = chunk.reshape(-1)
    k = max(2, int(k))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(emb)
    # 가장 가까운 중심까지의 거리
    dists = np.min(np.linalg.norm(emb[:, None, :] - km.cluster_centers_[None, :, :], axis=2),
                   axis=1)
    scores = dists
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


# ============================================================
# 3. 딥러닝 방법 (Autoencoder)
# ============================================================

def detect_autoencoder(X, contamination=0.05, hidden=8, max_iter=400, random_state=42, scaler="standard", **kwargs) -> Dict[str, Any]:
    """간단한 MLP Autoencoder (sklearn MLPRegressor 기반).
       재구성 오차(reconstruction error)를 이상 점수로 사용."""
    X = _to_2d(X)
    Xs = get_scaler(scaler).fit_transform(X)
    n, d = Xs.shape
    hidden = max(2, min(int(hidden), d * 2))
    # encoder-decoder를 한 MLP로 표현 (병목층 사용)
    model = MLPRegressor(
        hidden_layer_sizes=(max(d, hidden * 2), hidden, max(d, hidden * 2)),
        activation="relu",
        solver="adam",
        max_iter=int(max_iter),
        random_state=random_state,
        early_stopping=False,
    )
    model.fit(Xs, Xs)
    rec = model.predict(Xs)
    if rec.ndim == 1:
        rec = rec.reshape(-1, 1)
    err = np.mean((Xs - rec) ** 2, axis=1)
    scores = err
    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


# ============================================================
# 4. Darts 기반 ForecastingAnomalyModel (선택적)
# ============================================================

def detect_darts_forecasting(X, contamination=0.05, lags=24, **kwargs) -> Dict[str, Any]:
    """Darts의 ForecastingAnomalyModel + NormScorer 기반.
       단변량 또는 다변량 모두 지원."""
    try:
        from darts import TimeSeries
        from darts.ad import ForecastingAnomalyModel, NormScorer
        from darts.models import LinearRegressionModel
    except ImportError:
        # darts가 설치되지 않은 환경에서는 z-score로 폴백
        return detect_zscore(X, contamination=contamination)

    X = _to_2d(X)
    n, d = X.shape
    lags = max(2, min(int(lags), max(2, n // 10)))
    try:
        # darts는 DatetimeIndex 또는 RangeIndex가 필요
        ts = TimeSeries.from_values(X.astype(float))
        forecaster = LinearRegressionModel(lags=lags, output_chunk_length=1)
        model = ForecastingAnomalyModel(model=forecaster, scorer=NormScorer())
        model.fit(ts, allow_model_training=True)
        scores_ts = model.score(ts)
        if isinstance(scores_ts, list):
            scores_ts = scores_ts[0]
        s_arr = scores_ts.values().flatten()
        # darts score 길이는 input - lags 만큼 줄어들 수 있음 -> 앞쪽을 0으로 패딩
        scores = np.zeros(n)
        scores[n - len(s_arr):] = s_arr
    except Exception:
        return detect_zscore(X, contamination=contamination)

    pred, thr = _binarize(scores, contamination)
    return {"scores": scores, "pred": pred, "threshold": thr}


# ============================================================
# 알고리즘 레지스트리
# ============================================================

ALGORITHMS = {
    "Z-Score (통계)": detect_zscore,
    "MAD (통계, Robust)": detect_mad,
    "IQR (통계)": detect_iqr,
    "STL 잔차 (통계)": detect_stl_residual,
    "Isolation Forest (ML)": detect_isolation_forest,
    "One-Class SVM (ML)": detect_ocsvm,
    "Local Outlier Factor (ML)": detect_lof,
    "KMeans Scorer (ML, 윈도우)": detect_kmeans_scorer,
    "Autoencoder (딥러닝)": detect_autoencoder,
    "Darts Forecasting (혼합)": detect_darts_forecasting,
}

ALGORITHM_DESCRIPTIONS = {
    "Z-Score (통계)": "각 변수의 표준점수 절대값 평균을 이상 점수로 사용. 빠르고 해석 쉬움.",
    "MAD (통계, Robust)": "Median Absolute Deviation 기반. 극단치에 강건함.",
    "IQR (통계)": "사분위범위를 벗어난 정도를 점수화. 단순하고 직관적.",
    "STL 잔차 (통계)": "추세·계절성 분해 후 잔차의 크기로 판단. 계절성 있는 데이터에 적합.",
    "Isolation Forest (ML)": "랜덤 분할로 격리 깊이가 짧은 데이터를 이상으로 판단. 고차원 우수.",
    "One-Class SVM (ML)": "정상 영역을 학습하여 경계 밖을 이상으로 판단. 비선형 경계 학습.",
    "Local Outlier Factor (ML)": "지역 밀도 차이로 이상을 판단. 군집 데이터에 효과적.",
    "KMeans Scorer (ML, 윈도우)": "슬라이딩 윈도우 임베딩 + KMeans 군집 중심 거리. 패턴 이상에 강함.",
    "Autoencoder (딥러닝)": "오토인코더 재구성 오차로 판단. 복잡한 패턴 학습.",
    "Darts Forecasting (혼합)": "예측 모형의 예측 오차를 NormScorer로 점수화. 통계+ML 혼합.",
}
