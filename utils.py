"""
utils.py
- CSV 로드, 컬럼 자동 감지, 전처리 유틸리티
"""
import pandas as pd
import numpy as np
import io
from typing import Optional, List, Tuple


def load_csv(uploaded_file) -> pd.DataFrame:
    """다양한 형식의 CSV를 읽음 (인코딩, 구분자 자동 추정)."""
    raw = uploaded_file.read()
    # 인코딩 시도
    text = None
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    # 구분자 추정
    sample = text[:5000]
    sep_candidates = [",", ";", "\t", "|"]
    sep = max(sep_candidates, key=lambda s: sample.count(s))
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception:
        df = pd.read_csv(io.StringIO(text))
    return df


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """시간 컬럼을 자동 감지."""
    candidates = []
    for col in df.columns:
        cname = str(col).lower()
        # 이름 기반 후보
        if any(k in cname for k in ["time", "date", "timestamp", "datetime", "시간", "날짜", "일자", "ds"]):
            candidates.append((col, 2))
        # 값 기반: parse 가능 여부 체크
        sample = df[col].dropna().head(20)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                ratio = parsed.notna().mean()
                if ratio > 0.8:
                    candidates.append((col, 1 + ratio))
            except Exception:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def detect_label_column(df: pd.DataFrame,
                        value_cols: Optional[List[str]] = None,
                        exclude: Optional[List[str]] = None) -> Optional[str]:
    """이상 라벨 컬럼을 자동 감지 (0/1, True/False, normal/anomaly 등).
       이름 우선 → 값 기반."""
    exclude = set(exclude or [])
    # 1) 이름 기반 우선
    for col in df.columns:
        if col in exclude:
            continue
        cname = str(col).lower()
        if any(k in cname for k in ["label", "anomaly", "is_anomaly", "outlier",
                                     "이상", "라벨", "anomal"]):
            uniq = df[col].dropna().unique()
            if len(uniq) <= 5:
                return col
    # 2) 값 기반
    for col in df.columns:
        if col in exclude:
            continue
        if value_cols and col in value_cols:
            continue
        try:
            uniq = pd.Series(df[col].dropna().unique())
            if len(uniq) == 2:
                u_str = set(uniq.astype(str).str.lower())
                if u_str <= {"0", "1", "true", "false", "yes", "no",
                             "normal", "anomaly", "outlier"}:
                    return col
        except Exception:
            pass
    return None


def to_binary_label(series: pd.Series) -> pd.Series:
    """라벨 시리즈를 0/1로 변환."""
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s != 0).astype(int)
    s_str = s.astype(str).str.lower().str.strip()
    pos = {"1", "true", "yes", "anomaly", "outlier", "abnormal", "이상", "y", "t"}
    return s_str.isin(pos).astype(int)


def prepare_timeseries(df: pd.DataFrame,
                       time_col: Optional[str],
                       value_cols: List[str],
                       missing_strategy: str = "interpolate") -> pd.DataFrame:
    """시간 컬럼을 인덱스로 설정한 시계열 데이터프레임 반환.

       Args:
           missing_strategy: 결측치 처리 방법 (강의 06주차)
               - "interpolate": 선형 보간 (기본, 가장 부드러움)
               - "ffill": forward fill (앞 값으로 채움)
               - "bfill": backward fill (뒤 값으로 채움)
               - "drop": 결측 행 제거
    """
    work = df.copy()
    if time_col is not None and time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col])
        work = work.sort_values(time_col)
        work = work.set_index(time_col)
    # 수치형 변환
    for c in value_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[value_cols]
    # 결측 처리
    if missing_strategy == "drop":
        work = work.dropna()
    elif missing_strategy == "ffill":
        work = work.ffill().bfill()
    elif missing_strategy == "bfill":
        work = work.bfill().ffill()
    else:  # interpolate (기본)
        work = work.interpolate(method="linear").bfill().ffill()
    return work


def labels_from_intervals(index: pd.Index,
                          intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    """사용자가 지정한 (start, end) 구간 리스트로부터 0/1 라벨 배열 생성."""
    labels = np.zeros(len(index), dtype=int)
    if not intervals:
        return labels
    idx_series = pd.Series(index)
    for start, end in intervals:
        mask = (idx_series >= start) & (idx_series <= end)
        labels[mask.values] = 1
    return labels


def infer_value_columns(df: pd.DataFrame,
                        time_col: Optional[str],
                        label_col: Optional[str]) -> List[str]:
    """수치형 값 컬럼을 자동 추론.
       - 시간/라벨 컬럼 제외
       - 단일값 컬럼 제외
       - 이름이 label/anomaly로 보이거나 unique값이 0/1뿐인 이진 컬럼은 제외."""
    candidates = []
    label_keywords = ["label", "anomaly", "is_anomaly", "outlier", "이상", "라벨"]
    for col in df.columns:
        if col == time_col or col == label_col:
            continue
        cname = str(col).lower()
        if any(k in cname for k in label_keywords):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        # 이진 0/1 컬럼은 라벨일 가능성이 높으므로 제외
        uniq = df[col].dropna().unique()
        if len(uniq) == 2 and set(pd.Series(uniq).astype(float).astype(int)) <= {0, 1}:
            continue
        candidates.append(col)
    return candidates
