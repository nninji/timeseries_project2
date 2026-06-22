"""
report.py
- 모든 분석 결과를 종합한 Markdown 보고서 자동 생성
"""
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def build_report(meta: Dict,
                 diagnoses: Dict[str, Dict],
                 recommendations: List[tuple],
                 results: Dict[str, Dict],
                 metric_df: Optional[pd.DataFrame],
                 class_df: Optional[pd.DataFrame],
                 type_summary: Optional[Dict[str, int]],
                 has_labels: bool) -> str:
    """전체 분석 보고서를 Markdown 문자열로 반환."""
    lines: List[str] = []
    add = lines.append

    add(f"# 시계열 이상탐지 분석 보고서")
    add("")
    add(f"- **생성 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- **데이터 소스**: {meta.get('source', '-')}")
    add(f"- **샘플 수**: {meta.get('n_rows', '-')}")
    add(f"- **변수(다변량)**: {', '.join(meta.get('value_cols', []))}")
    add(f"- **시간 컬럼**: {meta.get('time_col', '(없음)')}")
    add(f"- **추정 빈도**: {meta.get('freq', '-')} / **주기 길이**: {meta.get('period', '-')}")
    add(f"- **Ground Truth 라벨**: {'있음' if has_labels else '없음'}")
    add("")

    # 1. 시계열 진단
    add("## 1. 시계열 진단")
    add("")
    add("강의 자료에 따른 표준 진단 절차(STL 분해 → ADF 정상성 검정 → STL 잔차의 Ljung-Box 백색잡음 검정)를 수행했습니다.")
    add("")
    add("| 변수 | n | 평균 | 표준편차 | ADF p-value | 정상성 | 계절성강도 | 추세강도 | 잔차 백색잡음 |")
    add("|---|---:|---:|---:|---:|:-:|---:|---:|:-:|")
    for col, d in diagnoses.items():
        adf = d.get("adf", {})
        lb = d.get("ljung_box_resid", {})
        add(f"| {col} | {d.get('n','-')} | {_fmt(d.get('mean'))} | "
            f"{_fmt(d.get('std'))} | {_fmt(adf.get('p_value'))} | "
            f"{'✅' if adf.get('is_stationary') else '❌'} | "
            f"{_fmt(d.get('seasonal_strength'))} | {_fmt(d.get('trend_strength'))} | "
            f"{'✅' if lb.get('is_white_noise') else '❌'} |")
    add("")
    add("- **ADF Test**: p<0.05면 정상 시계열 (강의 09_ARIMA)")
    add("- **계절성/추세 강도**: 1에 가까울수록 강함 (Hyndman & Athanasopoulos 공식)")
    add("- **Ljung-Box**: STL 잔차가 백색잡음(p≥0.05)이면 분해가 잘 된 것")
    add("")

    # 2. 알고리즘 추천
    if recommendations:
        add("## 2. 데이터 특성 기반 알고리즘 추천")
        add("")
        add("진단 결과를 바탕으로 다음 알고리즘들을 우선 추천합니다.")
        add("")
        for i, (name, reason) in enumerate(recommendations, 1):
            add(f"{i}. **{name}** — {reason}")
        add("")

    # 3. 평가 결과
    if metric_df is not None and not metric_df.empty:
        add("## 3. 알고리즘별 평가지표")
        add("")
        if has_labels:
            add("Ground Truth 라벨 기반 정량 평가 결과입니다.")
            add("")
            cols = [c for c in ["Precision", "Recall", "F1", "AUC-ROC", "AUC-PR", "FPR",
                                 "TP", "FP", "FN", "TN"] if c in metric_df.columns]
            header = "| 알고리즘 | " + " | ".join(cols) + " |"
            sep = "|---|" + "|".join([":-:" if c in {"TP","FP","FN","TN"} else "---:"
                                       for c in cols]) + "|"
            add(header); add(sep)
            for idx, row in metric_df.iterrows():
                vals = []
                for c in cols:
                    if c in ("TP", "FP", "FN", "TN"):
                        vals.append(str(int(row[c])) if pd.notna(row[c]) else "-")
                    else:
                        vals.append(_fmt(row[c]))
                add(f"| {idx} | " + " | ".join(vals) + " |")
            add("")
            # 최고 성능
            if "F1" in metric_df.columns and metric_df["F1"].notna().any():
                best = metric_df["F1"].idxmax()
                add(f"🏆 **F1 기준 최고 성능**: {best} (F1 = {_fmt(metric_df.loc[best, 'F1'])})")
                add("")
        else:
            add("Ground Truth 라벨이 없어 탐지 비율만 표시합니다.")
            add("")
            add("| 알고리즘 | 탐지 비율 |")
            add("|---|---:|")
            for idx, row in metric_df.iterrows():
                add(f"| {idx} | {_fmt(row.get('탐지 비율'), nd=4)} |")
            add("")

    # 4. 이상 유형 분류
    if class_df is not None and not class_df.empty and type_summary is not None:
        add("## 4. 탐지된 이상의 유형 분류")
        add("")
        add("강의 자료 14주차의 분류 체계에 따라 탐지된 각 이상 구간을 자동 분류했습니다.")
        add("")
        total = sum(type_summary.values())
        add("| 유형 | 개수 | 비율 |")
        add("|---|---:|---:|")
        for k, v in type_summary.items():
            ratio = (v / total * 100) if total > 0 else 0.0
            add(f"| {k} | {v} | {ratio:.1f}% |")
        add(f"| **합계** | **{total}** | 100.0% |")
        add("")
        add("**해석**")
        add("- **Global**: 전체 시계열의 정상 범주에서 크게 벗어난 단일 점 (예: 평소 50, 갑자기 200)")
        add("- **Contextual**: 전체로는 정상 범주지만 인접 구간 대비 이상 (예: 여름철 낮은 전력)")
        add("- **Shapelet**: 짧은 구간에 일반적이지 않은 모양/사이클")
        add("- **Seasonal**: 계절성에서 벗어나는 부분 시계열")
        add("- **Trend**: 영구적인 추세 변화")
        add("")

    # 5. 결론
    add("## 5. 결론 및 권장사항")
    add("")
    if metric_df is not None and "F1" in metric_df.columns and metric_df["F1"].notna().any():
        best = metric_df["F1"].idxmax()
        add(f"- 현재 데이터에서는 **{best}**가 F1 기준 가장 우수한 성능을 보였습니다.")
    add("- 운영 시에는 단일 알고리즘보다 **Aggregator(OR/AND/Majority)** 를 통한 앙상블을 권장합니다.")
    add("- 라벨이 부족할 경우, 알려진 이상 구간을 수동 라벨로 추가하여 임계값을 보정하세요.")
    add("- 데이터 분포가 시간에 따라 변할 수 있으므로 주기적으로 재학습(re-fit) 하는 것이 좋습니다.")
    add("")
    add("---")
    add("*본 보고서는 시계열 이상탐지 자동화 대시보드에 의해 자동 생성되었습니다.*")
    return "\n".join(lines)
