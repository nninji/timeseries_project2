"""
app.py - 시계열 이상탐지 자동화 대시보드 

사용법:
    streamlit run app.py

설계 사상 :
    [1] 데이터 업로드 & 매핑   — 자동 감지 + 수동 보정
    [2] 시계열 자동 진단       — STL 분해, ADF/Ljung-Box, ACF/PACF 
    [3] 이상탐지 실행          — Scorer → Detector → Aggregator 
    [4] 평가 대시보드          — P/R/F1, AUC-ROC/PR, 혼동행렬, 이상유형 분류
    [5] 자동 분석 보고서       — 모든 결과를 Markdown으로 종합
"""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    load_csv, detect_time_column, detect_label_column,
    to_binary_label, prepare_timeseries, labels_from_intervals,
    infer_value_columns,
)
from detection import ALGORITHMS, ALGORITHM_DESCRIPTIONS
from metrics import compute_metrics, get_roc_curve, get_pr_curve
from diagnosis import (
    diagnose_dataframe, infer_freq, recommend_algorithms,
)
from anomaly_types import classify_predictions, summarize_types
from darts_pipeline import (
    AGGREGATORS, aggregator_or, aggregator_and, aggregator_majority,
    forecasting_anomaly_model,
)
from report import build_report
from lecture_visuals import (
    make_anomaly_types_grid, make_normal_anomaly_diagram_html,
    make_darts_pipeline_diagram_html, make_nyc_known_anomalies_zoom,
    show_anomalies_from_scores_plot, scorer_guide_html,
)


# ===============================================================
# 페이지 설정
# ===============================================================
st.set_page_config(
    page_title="시계열 이상탐지 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stMetric { background-color: #ffffff; padding: 0.4rem 0.6rem; border-radius: 0.4rem;
            border: 1px solid #eef0f4; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 6px 6px 0 0; }
div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 시계열 이상탐지 자동화 대시보드")
st.caption(
    "다변량 CSV → **자동 진단 → 알고리즘 추천 → Darts 4단 파이프라인(Scorer·Detector·Aggregator) "
    "→ 평가 → 보고서**까지 워크플로우를 그대로 구현했습니다."
)


# ===============================================================
# 세션 상태 초기화
# ===============================================================
DEFAULTS = {
    "results": {},
    "agg_pred": None,
    "agg_name": None,
    "manual_intervals": [],
    "last_file_id": None,
    "diagnoses": None,
    "period": None,
    "freq": None,
    "recommendations": [],
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ===============================================================
# 사이드바: 데이터 업로드 & 핵심 설정
# ===============================================================
with st.sidebar:
    st.header("⚙️ 설정")

    uploaded = st.file_uploader(
        "CSV 파일 업로드 (다변량 시계열)",
        type=["csv"],
        help="시간 컬럼 + 1개 이상의 수치 컬럼 필요. (선택) 0/1 라벨 컬럼이 있으면 자동 인식.",
    )

    sample_choice = st.radio(
        "샘플 데이터",
        options=["(사용 안 함)",
                 "내장 다변량 합성 데이터",
                 "NYC Taxi (강의 예제)",
                 "다변량 센서 (라벨 포함)",
                 "시스템 메트릭 (라벨 없음)"],
        index=0 if uploaded else 1,
        help="업로드한 파일이 없을 때 시연용으로 사용합니다.",
    )

    st.divider()
    st.subheader("🧰 결측치 처리")
    missing_strategy = st.selectbox(
        "결측치 처리 방법",
        options=["interpolate", "ffill", "bfill", "drop"],
        index=0,
        help=("• interpolate: 선형 보간(기본, 가장 부드러움)\n"
              "• ffill: 앞 값으로 채움\n"
              "• bfill: 뒤 값으로 채움\n"
              "• drop: 결측 행 제거"),
    )

    st.divider()
    st.subheader("🔧 탐지 파라미터")
    contamination = st.slider(
        "이상 비율(예상)",
        min_value=0.001, max_value=0.30, value=0.05, step=0.005,
        help="데이터 중 이상으로 추정되는 비율. 상위 N% 점수를 이상으로 분류.",
    )
    scaler_name = st.selectbox(
        "스케일러 ",
        options=["standard", "robust", "minmax", "power"],
        index=0,
        help=("• standard: 평균 0, 분산 1 (가장 일반적)\n"
              "• robust: 중앙값 0, 분산 1 (이상치에 강건)\n"
              "• minmax: [0, 1] 범위로 정규화\n"
              "• power: Yeo-Johnson 변환 (정규분포에 가깝게)"),
    )
    train_ratio = st.slider(
        "학습 데이터 비율 (Forecasting Model용)",
        min_value=0.3, max_value=0.9, value=0.6, step=0.05,
        help="ForecastingAnomalyModel이 사용할 학습 구간 비율. "
             "강의 11주차 백테스팅 방식 — 학습/평가 기간을 다르게 했을 때도 잘 적용되는지 확인.",
    )

    st.divider()
    st.subheader("🧪 알고리즘 선택")
    all_algos = list(ALGORITHMS.keys())
    default_algos = [
        "Z-Score (통계)",
        "STL 잔차 (통계)",
        "Isolation Forest (ML)",
        "Local Outlier Factor (ML)",
        "Autoencoder (딥러닝)",
    ]
    selected_algos = st.multiselect(
        "사용할 알고리즘",
        options=all_algos,
        default=default_algos,
    )
    use_forecasting_pipeline = st.checkbox(
        "ForecastingAnomalyModel 사용",
        value=False,
        help="SKLearnModel + 캘린더 공변량(hour/dayofweek) + NormScorer/KMeansScorer. "
             "파이프라인입니다."
    )

    st.divider()
    st.subheader("🔗 Aggregator (앙상블)")
    aggregator_name = st.selectbox(
        "여러 알고리즘 결과 통합 방식",
        options=list(AGGREGATORS.keys()),
        index=0,
    )

    with st.expander("ℹ️ 알고리즘 설명"):
        for name in all_algos:
            st.markdown(f"**{name}**: {ALGORITHM_DESCRIPTIONS[name]}")


# ===============================================================
# 샘플 데이터 생성기
# ===============================================================
def make_synthetic_multivariate() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 600
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    s1 = 10 + 5 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / (24 * 7))
    s2 = 20 + 3 * np.cos(2 * np.pi * t / 24) + 0.01 * t
    s3 = 5 + np.sin(2 * np.pi * t / 48)
    sensor_a = s1 + rng.normal(0, 0.6, n)
    sensor_b = s2 + rng.normal(0, 0.4, n)
    sensor_c = s3 + rng.normal(0, 0.3, n)
    labels = np.zeros(n, dtype=int)
    for i in [80, 81, 82, 200, 350, 351, 352, 353, 480, 481, 550]:
        sensor_a[i] += rng.choice([-1, 1]) * rng.uniform(8, 15)
        sensor_b[i] += rng.choice([-1, 1]) * rng.uniform(6, 12)
        labels[i] = 1
    sensor_a[300:310] += np.linspace(0, 8, 10)
    labels[300:310] = 1
    return pd.DataFrame({
        "timestamp": idx, "sensor_a": sensor_a, "sensor_b": sensor_b,
        "sensor_c": sensor_c, "is_anomaly": labels,
    })


# ===============================================================
# 데이터 로드
# ===============================================================
raw_df = None
data_source_name = None
file_id = None

if uploaded is not None:
    try:
        file_id = ("UP", uploaded.name, uploaded.size)
        raw_df = load_csv(uploaded)
        data_source_name = uploaded.name
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")
elif sample_choice == "내장 다변량 합성 데이터":
    raw_df = make_synthetic_multivariate()
    data_source_name = "synthetic_multivariate"
    file_id = ("SAMPLE", "synthetic")
elif sample_choice == "NYC Taxi (강의 예제)":
    try:
        raw_df = pd.read_csv("sample_data/nyc_taxi.csv")
        data_source_name = "nyc_taxi.csv"
        file_id = ("SAMPLE", "nyc_taxi")
    except Exception as e:
        st.error(f"NYC Taxi 샘플 로드 실패: {e}")
elif sample_choice == "다변량 센서 (라벨 포함)":
    try:
        raw_df = pd.read_csv("sample_data/multivariate_sensor.csv")
        data_source_name = "multivariate_sensor.csv"
        file_id = ("SAMPLE", "multivariate_sensor")
    except Exception as e:
        st.error(f"센서 샘플 로드 실패: {e}")
elif sample_choice == "시스템 메트릭 (라벨 없음)":
    try:
        raw_df = pd.read_csv("sample_data/system_metrics_nolabel.csv")
        data_source_name = "system_metrics_nolabel.csv"
        file_id = ("SAMPLE", "system_metrics")
    except Exception as e:
        st.error(f"시스템 메트릭 샘플 로드 실패: {e}")

# 파일 변경 시 결과 초기화
if file_id is not None and st.session_state["last_file_id"] != file_id:
    st.session_state["results"] = {}
    st.session_state["agg_pred"] = None
    st.session_state["agg_name"] = None
    st.session_state["manual_intervals"] = []
    st.session_state["diagnoses"] = None
    st.session_state["recommendations"] = []
    st.session_state["last_file_id"] = file_id

if raw_df is None:
    st.info("👈 사이드바에서 CSV를 업로드하거나 샘플 데이터를 선택해 주세요.")
    st.stop()


# ===============================================================
# 컬럼 자동 감지 (사이드바 위 / 본문 진입 전)
# ===============================================================
auto_time = detect_time_column(raw_df)
auto_label_pre = detect_label_column(raw_df, value_cols=None,
                                       exclude=[auto_time] if auto_time else None)


# ===============================================================
# 메인 탭 구조
# ===============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 1. 데이터 & 매핑",
    "🔍 2. 시계열 진단",
    "🎯 3. 이상탐지 실행",
    "📊 4. 평가 대시보드",
    "📑 5. 분석 보고서",
])


# ===============================================================
# Tab 1: 데이터 & 매핑
# ===============================================================
with tab1:
    st.subheader("데이터 미리보기")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("데이터 소스", data_source_name)
    c2.metric("행 수", f"{len(raw_df):,}")
    c3.metric("컬럼 수", len(raw_df.columns))
    c4.metric("결측치", int(raw_df.isna().sum().sum()))

    with st.expander("📋 원본 데이터 (상위 50행)"):
        st.dataframe(raw_df.head(50), use_container_width=True)

    st.subheader("컬럼 매핑")
    st.caption("자동으로 추정된 컬럼을 그대로 사용하거나, 직접 지정할 수 있습니다.")

    colmap1, colmap2 = st.columns(2)
    with colmap1:
        time_options = ["(없음 - 행 순서로 사용)"] + list(raw_df.columns)
        default_time = time_options.index(auto_time) if auto_time in raw_df.columns else 0
        time_col_sel = st.selectbox("⏱️ 시간 컬럼", time_options, index=default_time)
        time_col = None if time_col_sel == "(없음 - 행 순서로 사용)" else time_col_sel

    with colmap2:
        cand_value_cols = infer_value_columns(raw_df, time_col, auto_label_pre)
        default_values = cand_value_cols if cand_value_cols else [
            c for c in raw_df.columns if c != time_col and c != auto_label_pre
        ][:3]
        value_cols = st.multiselect(
            "📊 값 컬럼 (다변량 가능)",
            options=[c for c in raw_df.columns if c != time_col],
            default=default_values,
        )

    auto_label = detect_label_column(raw_df, value_cols=value_cols,
                                       exclude=[time_col] if time_col else None)
    label_options = ["(없음)"] + [c for c in raw_df.columns
                                    if c not in value_cols and c != time_col]
    default_label_idx = label_options.index(auto_label) if auto_label in label_options else 0
    label_col_sel = st.selectbox(
        "🏷️ Ground Truth 라벨 컬럼 (선택)",
        label_options,
        index=default_label_idx,
        help="0=정상 / 1=이상. 다양한 표현(True/False, normal/anomaly)도 자동 인식.",
    )
    label_col = None if label_col_sel == "(없음)" else label_col_sel

    if not value_cols:
        st.warning("값 컬럼을 1개 이상 선택해 주세요.")
        st.stop()

    # 시계열 준비
    ts_df = prepare_timeseries(raw_df, time_col, value_cols,
                                 missing_strategy=missing_strategy)
    freq, period = infer_freq(ts_df.index)
    st.session_state["freq"] = freq
    st.session_state["period"] = period

    # 결측치 처리 정보 표시
    n_missing_raw = int(raw_df[value_cols].isna().sum().sum()) if value_cols else 0
    if n_missing_raw > 0:
        st.info(f"📌 원본에 결측치 **{n_missing_raw}개** 발견 → **'{missing_strategy}'** 방식으로 처리됨. "
                f"(처리 방법은 사이드바에서 변경 가능)")

    # 라벨 처리
    labels = None
    if label_col is not None and label_col in raw_df.columns:
        if time_col is not None:
            tmp = raw_df[[time_col, label_col]].copy()
            tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
            tmp = tmp.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
            tmp = tmp.reindex(ts_df.index, method="nearest")
            labels = to_binary_label(tmp[label_col]).values
        else:
            labels = to_binary_label(raw_df[label_col]).values
            if len(labels) != len(ts_df):
                labels = labels[: len(ts_df)] if len(labels) > len(ts_df) else np.pad(
                    labels, (0, len(ts_df) - len(labels))
                )

    # 수동 이상 구간 지정
    with st.expander("✍️ 수동 이상 구간 지정 (선택)"):
        st.caption("라벨이 없거나, 알려진 이상 구간을 추가하고 싶을 때 사용하세요.")
        if not isinstance(ts_df.index, pd.DatetimeIndex):
            st.info("시간 컬럼이 없거나 날짜 파싱이 안 되어 수동 구간 지정이 비활성화됩니다.")
        else:
            idx_min = ts_df.index.min().to_pydatetime()
            idx_max = ts_df.index.max().to_pydatetime()
            cA, cB, cC = st.columns([1, 1, 1])
            with cA:
                start = st.date_input("시작일", value=idx_min.date(),
                                       min_value=idx_min.date(),
                                       max_value=idx_max.date(),
                                       key="man_start")
            with cB:
                end = st.date_input("종료일", value=idx_min.date(),
                                     min_value=idx_min.date(),
                                     max_value=idx_max.date(),
                                     key="man_end")
            with cC:
                st.write(""); st.write("")
                if st.button("➕ 구간 추가"):
                    if start <= end:
                        st.session_state["manual_intervals"].append(
                            (pd.Timestamp(start),
                             pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                        )
                        st.rerun()
            if st.session_state["manual_intervals"]:
                st.write("**현재 지정된 구간:**")
                for i, (s, e) in enumerate(st.session_state["manual_intervals"]):
                    cc1, cc2 = st.columns([4, 1])
                    cc1.write(f"{i+1}. {s} ~ {e}")
                    if cc2.button("삭제", key=f"del_{i}"):
                        st.session_state["manual_intervals"].pop(i)
                        st.rerun()

    if st.session_state["manual_intervals"] and isinstance(ts_df.index, pd.DatetimeIndex):
        manual_labels = labels_from_intervals(ts_df.index, st.session_state["manual_intervals"])
        if labels is None:
            labels = manual_labels
        else:
            labels = np.maximum(labels, manual_labels)

    # 시계열 시각화
    st.subheader("시계열 시각화")
    fig = make_subplots(
        rows=len(value_cols), cols=1, shared_xaxes=True,
        subplot_titles=value_cols, vertical_spacing=0.05,
    )
    n_total = len(ts_df)
    n_train_viz = int(train_ratio * n_total)
    for i, col in enumerate(value_cols, start=1):
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[col],
                                  mode="lines", name=col, line=dict(width=1)),
                      row=i, col=1)
    # Train/Test 분할 영역 표시 (강의 11주차 백테스팅 시각화)
    if n_train_viz < n_total:
        split_time = ts_df.index[n_train_viz - 1]
        for r in range(1, len(value_cols) + 1):
            fig.add_vline(x=split_time, line_dash="dot", line_color="navy",
                           line_width=1.5, row=r, col=1)
            if r == 1:
                fig.add_annotation(x=split_time, y=1.02, yref="y domain",
                                     text=f"Train/Test 분할 ({train_ratio:.0%})",
                                     showarrow=False, font=dict(size=10, color="navy"),
                                     row=r, col=1)
    if labels is not None and np.any(labels == 1):
        in_run, start_i, runs = False, 0, []
        for i, v in enumerate(labels):
            if v == 1 and not in_run:
                start_i = i; in_run = True
            elif v == 0 and in_run:
                runs.append((start_i, i - 1)); in_run = False
        if in_run:
            runs.append((start_i, len(labels) - 1))
        for s, e in runs:
            for r in range(1, len(value_cols) + 1):
                fig.add_vrect(x0=ts_df.index[s], x1=ts_df.index[e],
                                fillcolor="red", opacity=0.15, line_width=0,
                                row=r, col=1)
    fig.update_layout(height=180 * len(value_cols) + 60, showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if labels is not None:
        st.success(f"✅ Ground Truth: 이상 비율 {np.mean(labels == 1):.2%} "
                   f"({int(np.sum(labels == 1))}/{len(labels)}). 빨간 음영이 알려진 이상 구간.")
    else:
        st.info("ℹ️ Ground Truth 라벨이 없습니다. 정성적 평가만 가능합니다.")

    # 빈도·주기 정보
    cinfo1, cinfo2 = st.columns(2)
    cinfo1.metric("추정 빈도(freq)", str(freq) if freq else "수동/추정 불가")
    cinfo2.metric("주기 길이(period)", str(period) if period else "—")

    # 강의 14주차 p.7-9: NYC Taxi 알려진 이상 5개 확대 시각화 (NYC Taxi 샘플 사용 시)
    is_nyc_taxi = (data_source_name == "nyc_taxi.csv"
                    and isinstance(ts_df.index, pd.DatetimeIndex)
                    and len(value_cols) >= 1)
    if is_nyc_taxi:
        with st.expander("🗽 NYC Taxi 알려진 이상 5개 확대 시각화",
                          expanded=True):
            st.caption("`plot_anom` 함수를 재현. 각 알려진 이상에 대해 ±3일 패드로 확대 시각화.")
            fig_zoom = make_nyc_known_anomalies_zoom(ts_df, value_cols[0], delta_days=3)
            st.plotly_chart(fig_zoom, use_container_width=True)


# 본문 외부에 변수 보존
ts_df_ref = ts_df
labels_ref = labels
value_cols_ref = value_cols
period_ref = period


# ===============================================================
# Tab 2: 시계열 자동 진단 
# ===============================================================
with tab2:
    # 이상의 큰 분류 (Normal/Novelty/Outlier/Anomaly)
    with st.expander("📘 이상의 유형 분류", expanded=False):
        st.markdown(make_normal_anomaly_diagram_html(), unsafe_allow_html=True)
        st.markdown(
            "**유형별 분류**\n"
            "- **점이상**: 데이터의 속성/분포가 다름 (주로 수치형, 이미지 데이터)\n"
            "- **맥락이상**: 데이터가 논리적으로 틀림 (주로 시계열, 공간 데이터)\n"
            "- **패턴이상**: 데이터가 패턴에서 벗어남 (주로 시계열, 자연어 데이터)"
        )

    # 시계열 이상 5종 유형 시각화
    with st.expander("📊 시계열 이상 5종 유형 시각화", expanded=True):
        st.caption("**Point** (점이상): Global, Contextual  &nbsp;|&nbsp;  "
                   "**Pattern** (패턴이상): Shapelet, Seasonal, Trend")
        fig_types = make_anomaly_types_grid()
        st.plotly_chart(fig_types, use_container_width=True)
        st.markdown(
            "- **Global (전역 이상)**: 전체 시계열을 고려했을 때 정상 범주를 크게 벗어남 — 예: CPU 50% 평균인데 갑자기 90%\n"
            "- **Contextual (맥락 이상)**: 인접 시계열의 맥락을 고려했을 때 이상 — 예: 여름철에 에어컨 사용 전력이 낮음\n"
            "- **Shapelet (모양 이상)**: 일반적인 shapelet/cycle과 다른 부분 시계열\n"
            "- **Seasonal (계절성 이상)**: 모양·트렌드는 유사하지만 계절성에서 벗어남\n"
            "- **Trend (추세 이상)**: 시계열의 추세에 영구적인 변화"
        )

    st.subheader("🔬 시계열 통계 진단")
    st.caption("**STL 분해 → ADF 정상성 검정 → STL 잔차의 Ljung-Box 검정 → ACF/PACF** "
               "를 자동 수행합니다.")

    if st.session_state["diagnoses"] is None:
        with st.spinner("진단 수행 중..."):
            st.session_state["diagnoses"] = diagnose_dataframe(ts_df_ref, period=period_ref)
            st.session_state["recommendations"] = recommend_algorithms(
                st.session_state["diagnoses"]
            )

    diagnoses = st.session_state["diagnoses"]

    # 진단 결과 요약 표
    rows = []
    for col, d in diagnoses.items():
        adf = d.get("adf", {})
        lb = d.get("ljung_box_resid", {})
        rows.append({
            "변수": col,
            "n": d.get("n"),
            "평균": d.get("mean"),
            "표준편차": d.get("std"),
            "ADF p-value": adf.get("p_value"),
            "정상성": "✅" if adf.get("is_stationary") else "❌",
            "계절성 강도": d.get("seasonal_strength"),
            "추세 강도": d.get("trend_strength"),
            "잔차 백색잡음 p": lb.get("p_value"),
            "백색잡음": "✅" if lb.get("is_white_noise") else "❌",
        })
    diag_df = pd.DataFrame(rows).set_index("변수")
    st.dataframe(
        diag_df.style.format({
            "평균": "{:.3f}", "표준편차": "{:.3f}",
            "ADF p-value": "{:.4f}",
            "계절성 강도": "{:.3f}", "추세 강도": "{:.3f}",
            "잔차 백색잡음 p": "{:.4f}",
        }),
        use_container_width=True,
    )
    st.caption("• ADF p<0.05 → 정상 시계열  • 계절성/추세 강도 1에 가까울수록 강함  "
               "• Ljung-Box p≥0.05 → STL 잔차가 백색잡음(분해 성공)")

    # 알고리즘 추천
    recs = st.session_state["recommendations"]
    if recs:
        st.subheader("🎯 데이터 특성 기반 추천 알고리즘")
        for i, (name, reason) in enumerate(recs, 1):
            st.markdown(f"**{i}. {name}** — {reason}")

    # STL 분해 시각화
    st.subheader("📉 STL 분해")
    var_for_decomp = st.selectbox(
        "분해할 변수 선택",
        options=value_cols_ref,
        key="stl_var",
    )
    d = diagnoses[var_for_decomp]
    decomp = d.get("decomp")
    if decomp is None:
        st.warning(f"데이터 길이가 부족하여 STL 분해를 수행할 수 없습니다 "
                   f"(필요: 2×period+5 = {2*(period_ref or 12)+5}, 실제: {d['n']}).")
    else:
        fig_stl = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                  subplot_titles=("관측값(Observed)", "추세(Trend)",
                                                  "계절성(Seasonal)", "잔차(Residual)"),
                                  vertical_spacing=0.05)
        x = ts_df_ref.index
        for i, key in enumerate(["observed", "trend", "seasonal", "resid"], start=1):
            fig_stl.add_trace(go.Scatter(x=x, y=decomp[key], mode="lines",
                                          line=dict(width=1)), row=i, col=1)
        fig_stl.update_layout(height=620, showlegend=False,
                                margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_stl, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("계절성 강도", f"{d['seasonal_strength']:.3f}")
        c2.metric("추세 강도", f"{d['trend_strength']:.3f}")

    # ACF / PACF
    st.subheader("🔁 ACF / PACF")
    ap = d.get("acf_pacf", {})
    if "acf" in ap and len(ap["acf"]) > 1:
        fig_ap = make_subplots(rows=1, cols=2,
                                 subplot_titles=("ACF (자기상관함수)", "PACF (편자기상관함수)"))
        lags = ap["lags"]
        ci = ap["ci"]
        fig_ap.add_trace(go.Bar(x=lags, y=ap["acf"], name="ACF",
                                 marker_color="#6464ff"), row=1, col=1)
        fig_ap.add_trace(go.Bar(x=lags, y=ap["pacf"], name="PACF",
                                 marker_color="#e07a5f"), row=1, col=2)
        for col in (1, 2):
            fig_ap.add_hline(y=ci, line_dash="dash", line_color="gray", row=1, col=col)
            fig_ap.add_hline(y=-ci, line_dash="dash", line_color="gray", row=1, col=col)
        fig_ap.update_layout(height=350, showlegend=False,
                              margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_ap, use_container_width=True)
        st.caption(f"점선: 95% 신뢰구간 ±{ci:.3f}. 신뢰구간을 벗어난 lag는 유의한 자기상관.")


# ===============================================================
# Tab 3: 이상탐지 실행
# ===============================================================
with tab3:
    st.subheader("🎯 이상탐지 실행")
    st.caption(
        "**Darts 4단 구조**(Scorer → Detector → Aggregator → Anomaly Model)를 "
        "기반으로 다양한 알고리즘을 일괄 수행합니다."
    )

    # Darts 이상탐지 4단 파이프라인 다이어그램
    with st.expander("📘 Darts 이상탐지 모듈 구조 (4단 파이프라인)",
                      expanded=False):
        st.markdown(make_darts_pipeline_diagram_html(), unsafe_allow_html=True)

    # Scorer 선택 가이드
    with st.expander("💡 Scorer 선택 가이드", expanded=False):
        st.markdown(scorer_guide_html(), unsafe_allow_html=True)

    info_c1, info_c2, info_c3 = st.columns(3)
    info_c1.metric("선택된 알고리즘", len(selected_algos))
    info_c2.metric("Aggregator", aggregator_name)
    info_c3.metric("Forecasting 사용", "예" if use_forecasting_pipeline else "아니오")

    run_button = st.button("🚀 선택한 알고리즘으로 탐지 실행", type="primary",
                            use_container_width=True)

    if run_button:
        if not selected_algos and not use_forecasting_pipeline:
            st.warning("알고리즘을 1개 이상 선택하거나 ForecastingAnomalyModel을 사용하세요.")
        else:
            st.session_state["results"] = {}
            X = ts_df_ref.values
            progress = st.progress(0.0, text="이상탐지 진행 중...")
            total = len(selected_algos) + (1 if use_forecasting_pipeline else 0)
            done = 0
            for name in selected_algos:
                try:
                    fn = ALGORITHMS[name]
                    with st.spinner(f"{name} 실행 중..."):
                        out = fn(X, contamination=contamination, scaler=scaler_name)
                    st.session_state["results"][name] = out
                except Exception as e:
                    st.error(f"{name} 실패: {e}")
                done += 1
                progress.progress(done / total, text=f"{name} 완료")

            # ForecastingAnomalyModel 
            if use_forecasting_pipeline:
                with st.spinner("ForecastingAnomalyModel (SKLearnModel + 캘린더 공변량) 실행 중..."):
                    try:
                        fam = forecasting_anomaly_model(ts_df_ref, lags=24,
                                                          add_calendar_covariates=True,
                                                          train_ratio=train_ratio,
                                                          use_wasserstein=True,
                                                          auto_lags=True)
                        # NormScorer + KMeansScorer + WassersteinScorer
                        for key, label in [
                            ("scores_norm",        "ForecastingAnomalyModel + NormScorer"),
                            ("scores_kmeans",      "ForecastingAnomalyModel + KMeansScorer"),
                            ("scores_wasserstein", "ForecastingAnomalyModel + WassersteinScorer"),
                        ]:
                            if key not in fam:
                                continue
                            scores = fam[key]
                            thr = float(np.quantile(scores, 1 - contamination))
                            pred = (scores >= thr).astype(int)
                            st.session_state["results"][label] = {
                                "scores": scores, "pred": pred, "threshold": thr,
                            }
                        if fam.get("fitted"):
                            st.info(f"✅ Darts ForecastingAnomalyModel 학습 완료. "
                                    f"lags={fam.get('lags_used')}, "
                                    f"train_n={fam.get('train_n')}, "
                                    f"캘린더 공변량(hour, dayofweek), Scorer 3종(Norm/KMeans/Wasserstein)")
                        else:
                            st.info("⚠️ Darts 미사용/실패 → 동일 인터페이스의 폴백 Scorer로 대체했습니다.")
                    except Exception as e:
                        st.error(f"ForecastingAnomalyModel 실패: {e}")
                done += 1
                progress.progress(done / total, text="ForecastingAnomalyModel 완료")
            progress.empty()

            # Aggregator로 앙상블
            preds_list = [r["pred"] for r in st.session_state["results"].values()]
            if preds_list:
                agg_fn = AGGREGATORS[aggregator_name]
                try:
                    agg_pred = agg_fn(preds_list)
                    st.session_state["agg_pred"] = agg_pred
                    st.session_state["agg_name"] = aggregator_name
                except Exception as e:
                    st.warning(f"Aggregator 실패: {e}")
            st.success(f"완료: {len(st.session_state['results'])}개 알고리즘 + "
                       f"Aggregator({aggregator_name})")

    if st.session_state["results"]:
        st.subheader("선택한 알고리즘별 점수·임계값")
        meta_rows = []
        for name, r in st.session_state["results"].items():
            meta_rows.append({
                "알고리즘": name,
                "임계값": r["threshold"],
                "탐지 개수": int(r["pred"].sum()),
                "탐지 비율": float(r["pred"].mean()),
                "점수 최댓값": float(r["scores"].max()),
                "점수 최솟값": float(r["scores"].min()),
            })
        st.dataframe(pd.DataFrame(meta_rows).set_index("알고리즘").style.format({
            "임계값": "{:.3f}", "탐지 비율": "{:.2%}",
            "점수 최댓값": "{:.3f}", "점수 최솟값": "{:.3f}",
        }), use_container_width=True)


# ===============================================================
# Tab 4: 평가 대시보드
# ===============================================================
with tab4:
    results = st.session_state.get("results", {})
    if not results:
        st.info("👈 먼저 **3. 이상탐지 실행** 탭에서 분석을 수행하세요.")
        st.stop()

    has_labels = labels_ref is not None

    # ---- 4.1 종합 지표
    st.subheader("📊 알고리즘별 종합 지표")
    if has_labels:
        st.caption("`eval_metric_from_scores(metric='AUC_ROC')` 와 동등한 평가를 모든 알고리즘에 대해 수행합니다.")
    rows = []
    for name, out in results.items():
        row = {"알고리즘": name}
        if has_labels:
            m = compute_metrics(labels_ref, out["pred"], out["scores"])
            row.update({
                "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
                "AUC-ROC": m["AUC-ROC"], "AUC-PR": m["AUC-PR"],
                "FPR": m["오탐율(FPR)"],
                "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            })
        else:
            row["탐지 비율"] = float(np.mean(out["pred"] == 1))
        rows.append(row)
    # Aggregator도 평가에 포함
    if st.session_state["agg_pred"] is not None:
        row = {"알고리즘": f"⭐ Aggregator({st.session_state['agg_name']})"}
        if has_labels:
            # Aggregator는 점수가 없으므로 AUC는 계산하지 않음
            m = compute_metrics(labels_ref, st.session_state["agg_pred"], None)
            row.update({
                "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
                "AUC-ROC": np.nan, "AUC-PR": np.nan,
                "FPR": m["오탐율(FPR)"],
                "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            })
        else:
            row["탐지 비율"] = float(np.mean(st.session_state["agg_pred"] == 1))
        rows.append(row)
    metric_df = pd.DataFrame(rows).set_index("알고리즘")

    if has_labels:
        st.dataframe(
            metric_df.style.format({
                "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
                "AUC-ROC": "{:.3f}", "AUC-PR": "{:.3f}", "FPR": "{:.3f}",
            }).background_gradient(subset=[c for c in ["Precision", "Recall", "F1",
                                                         "AUC-ROC", "AUC-PR"]
                                            if c in metric_df.columns],
                                    cmap="Greens"),
            use_container_width=True,
        )
        if "F1" in metric_df.columns and metric_df["F1"].notna().any():
            best = metric_df["F1"].idxmax()
            st.success(f"🏆 F1 기준 최고 성능: **{best}** "
                       f"(F1 = {metric_df.loc[best, 'F1']:.3f})")
    else:
        st.dataframe(metric_df.style.format({"탐지 비율": "{:.2%}"}),
                      use_container_width=True)

    # ---- 4.2 알고리즘 성능 비교 (막대)
    if has_labels and len(metric_df) > 1:
        st.subheader("📈 알고리즘 성능 비교")
        plot_cols = [c for c in ["Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"]
                     if c in metric_df.columns]
        comp_df = metric_df[plot_cols].reset_index()
        comp_long = comp_df.melt(id_vars="알고리즘", var_name="지표", value_name="값")
        fig_bar = px.bar(comp_long.dropna(), x="알고리즘", y="값", color="지표",
                          barmode="group", text_auto=".2f",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_bar.update_layout(yaxis=dict(range=[0, 1.05]),
                                xaxis_tickangle=-15,
                                margin=dict(l=10, r=10, t=10, b=80), height=480)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- 4.3 ROC / PR 곡선
    if has_labels:
        st.subheader("📉 ROC & Precision-Recall 곡선")
        c_roc, c_pr = st.columns(2)
        fig_roc, fig_pr = go.Figure(), go.Figure()
        for name, out in results.items():
            roc = get_roc_curve(labels_ref, out["scores"])
            pr = get_pr_curve(labels_ref, out["scores"])
            if roc is not None:
                fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"],
                                              mode="lines", name=name))
            if pr is not None:
                fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"],
                                             mode="lines", name=name))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(dash="dash", color="gray"),
                                      name="Random", showlegend=False))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                                height=420, margin=dict(l=10, r=10, t=40, b=10))
        fig_pr.update_layout(title="Precision-Recall Curve",
                              xaxis_title="Recall", yaxis_title="Precision",
                              height=420, margin=dict(l=10, r=10, t=40, b=10))
        c_roc.plotly_chart(fig_roc, use_container_width=True)
        c_pr.plotly_chart(fig_pr, use_container_width=True)

    # ---- 4.4 알고리즘별 상세 시각화 (show_anomalies_from_scores 스타일)
    st.subheader("🔍 알고리즘별 상세 탐지 결과")
    st.caption("`show_anomalies_from_scores()` 스타일로 시각화합니다.")
    options = list(results.keys())
    if st.session_state["agg_pred"] is not None:
        options.append(f"⭐ Aggregator({st.session_state['agg_name']})")
    selected_view = st.selectbox("자세히 볼 항목", options)

    if selected_view.startswith("⭐"):
        sel_pred = st.session_state["agg_pred"]
        sel_scores = None
        sel_thr = None
    else:
        out = results[selected_view]
        sel_pred = out["pred"]
        sel_scores = out["scores"]
        sel_thr = out["threshold"]

    # show_anomalies_from_scores 시각화 (점수가 있는 경우)
    if sel_scores is not None:
        title_lecture = f"{selected_view} 이상 탐지 결과"
        series_label = value_cols_ref[0] if value_cols_ref else "#Series"
        if data_source_name == "nyc_taxi.csv":
            series_label = "#Passengers"
        fig_show = show_anomalies_from_scores_plot(
            ts_df_ref, value_cols_ref[0], sel_scores, sel_pred,
            title=title_lecture, score_label="score_0",
            series_label=series_label, window=1,
        )
        st.plotly_chart(fig_show, use_container_width=True)

    st.markdown("**상세 비교**: 시계열 + 탐지된 이상 점 + 점수·임계값 + 예측/정답 라벨")
    fig_det = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=(f"시계열 ({value_cols_ref[0]})",
                        "이상 점수 (Anomaly Score)",
                        "이상 여부 (anomalies)"),
        vertical_spacing=0.08, row_heights=[0.45, 0.30, 0.25],
    )
    first_col = value_cols_ref[0]
    fig_det.add_trace(go.Scatter(x=ts_df_ref.index, y=ts_df_ref[first_col],
                                   mode="lines", name=first_col,
                                   line=dict(width=1, color="#6464ff")),
                       row=1, col=1)
    pred_mask = (sel_pred == 1)
    if np.any(pred_mask):
        fig_det.add_trace(go.Scatter(
            x=ts_df_ref.index[pred_mask],
            y=ts_df_ref[first_col].values[pred_mask],
            mode="markers", name="탐지된 이상",
            marker=dict(color="red", size=7, symbol="x"),
        ), row=1, col=1)

    if sel_scores is not None:
        fig_det.add_trace(go.Scatter(x=ts_df_ref.index, y=sel_scores, mode="lines",
                                       name="anomaly score",
                                       line=dict(color="#e07a5f", width=1)),
                           row=2, col=1)
        fig_det.add_hline(y=sel_thr, line_dash="dash", line_color="black",
                            annotation_text=f"임계값={sel_thr:.3f}", row=2, col=1)
    else:
        fig_det.add_annotation(text="Aggregator는 점수가 없는 이진 예측입니다",
                                 x=0.5, y=0.5, xref="x2 domain", yref="y2 domain",
                                 showarrow=False, font=dict(color="gray"))

    # show_anomalies_from_scores 스타일: yes/no 라벨로 명시
    fig_det.add_trace(go.Scatter(x=ts_df_ref.index, y=sel_pred,
                                   mode="lines", name="예측 (anomalies)",
                                   line=dict(color="red", width=1.2, shape="hv")),
                       row=3, col=1)
    if has_labels:
        fig_det.add_trace(go.Scatter(x=ts_df_ref.index, y=labels_ref, mode="lines",
                                       name="정답 (true)",
                                       line=dict(color="green", width=1, dash="dot",
                                                  shape="hv")),
                           row=3, col=1)
    fig_det.update_yaxes(tickvals=[0, 1], ticktext=["no", "yes"], row=3, col=1)
    fig_det.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_det, use_container_width=True)

    # 혼동행렬
    if has_labels:
        m = compute_metrics(labels_ref, sel_pred, sel_scores)
        cm_data = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        fig_cm = px.imshow(cm_data, text_auto=True, aspect="auto",
                            x=["Pred:정상", "Pred:이상"],
                            y=["True:정상", "True:이상"],
                            color_continuous_scale="Blues",
                            title=f"{selected_view} 혼동행렬")
        fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        cm1, cm2 = st.columns([1, 1.4])
        with cm1:
            st.plotly_chart(fig_cm, use_container_width=True)
        with cm2:
            st.markdown("**선택한 항목 주요 지표**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Precision", f"{m['Precision']:.3f}")
            mc2.metric("Recall", f"{m['Recall']:.3f}")
            mc3.metric("F1", f"{m['F1']:.3f}")
            mc1.metric("AUC-ROC", "—" if np.isnan(m['AUC-ROC']) else f"{m['AUC-ROC']:.3f}")
            mc2.metric("AUC-PR", "—" if np.isnan(m['AUC-PR']) else f"{m['AUC-PR']:.3f}")
            mc3.metric("오탐율(FPR)",
                       "—" if np.isnan(m['오탐율(FPR)']) else f"{m['오탐율(FPR)']:.3f}")
            st.caption(f"TP {m['TP']} / FP {m['FP']} / FN {m['FN']} / TN {m['TN']}")

    # ---- 4.5 이상 유형 분류 (강의 14주차)
    st.subheader("🏷️ 탐지된 이상의 유형 분류")
    st.caption("분류 체계 — **점이상(Global/Contextual), 패턴이상(Shapelet/Seasonal/Trend)** "
               "에 따라 자동 분류합니다.")
    try:
        class_df = classify_predictions(ts_df_ref.values, sel_pred, period=period_ref)
        summary = summarize_types(class_df)
    except Exception as e:
        class_df, summary = pd.DataFrame(), {}
        st.warning(f"이상 유형 분류 실패: {e}")

    if class_df.empty:
        st.info("탐지된 이상 구간이 없습니다.")
    else:
        ccA, ccB = st.columns([1, 2])
        with ccA:
            sum_df = pd.DataFrame({"유형": list(summary.keys()), "개수": list(summary.values())})
            fig_pie = px.pie(sum_df[sum_df["개수"] > 0], values="개수", names="유형",
                               hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        with ccB:
            view = class_df.copy()
            if isinstance(ts_df_ref.index, pd.DatetimeIndex):
                view["start"] = ts_df_ref.index[view["start_idx"]]
                view["end"] = ts_df_ref.index[view["end_idx"]]
                view = view[["start", "end", "length", "type",
                              "mean_z_global", "mean_z_local"]]
            st.dataframe(view.style.format({"mean_z_global": "{:.2f}",
                                              "mean_z_local": "{:.2f}"}),
                          use_container_width=True, height=320)

    # ---- 4.6 결과 다운로드
    st.subheader("💾 결과 다운로드")
    out_df = ts_df_ref.copy()
    for name, r in results.items():
        out_df[f"score__{name}"] = r["scores"]
        out_df[f"pred__{name}"] = r["pred"]
    if st.session_state["agg_pred"] is not None:
        out_df[f"pred__Aggregator({st.session_state['agg_name']})"] = (
            st.session_state["agg_pred"]
        )
    if has_labels:
        out_df["__label__"] = labels_ref
    csv_data = out_df.to_csv(index=True).encode("utf-8-sig")
    st.download_button("📥 탐지 결과 CSV", data=csv_data,
                        file_name="anomaly_results.csv", mime="text/csv")


# ===============================================================
# Tab 5: 분석 보고서
# ===============================================================
with tab5:
    st.subheader("📑 자동 분석 보고서")
    st.caption("진단·추천·평가·유형분류 결과를 Markdown 형태로 종합합니다.")

    results = st.session_state.get("results", {})
    if not results:
        st.info("👈 먼저 **3. 이상탐지 실행** 탭에서 분석을 수행하세요.")
        st.stop()

    has_labels = labels_ref is not None
    diagnoses = st.session_state.get("diagnoses", {})
    recs = st.session_state.get("recommendations", [])

    # 평가 지표 재계산
    rows = []
    for name, out in results.items():
        row = {"알고리즘": name}
        if has_labels:
            m = compute_metrics(labels_ref, out["pred"], out["scores"])
            row.update({
                "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
                "AUC-ROC": m["AUC-ROC"], "AUC-PR": m["AUC-PR"],
                "FPR": m["오탐율(FPR)"],
                "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            })
        else:
            row["탐지 비율"] = float(np.mean(out["pred"] == 1))
        rows.append(row)
    metric_df = pd.DataFrame(rows).set_index("알고리즘")

    # 분류 (베스트 알고리즘 기준)
    if has_labels and "F1" in metric_df.columns and metric_df["F1"].notna().any():
        best_name = metric_df["F1"].idxmax()
        best_pred = results[best_name]["pred"]
    else:
        best_name = list(results.keys())[0]
        best_pred = results[best_name]["pred"]
    class_df = classify_predictions(ts_df_ref.values, best_pred, period=period_ref)
    summary = summarize_types(class_df)

    meta = {
        "source": data_source_name,
        "n_rows": int(len(ts_df_ref)),
        "value_cols": list(ts_df_ref.columns),
        "time_col": time_col or "(없음)",
        "freq": st.session_state.get("freq"),
        "period": st.session_state.get("period"),
    }
    md = build_report(meta, diagnoses, recs, results,
                      metric_df, class_df, summary, has_labels)
    st.markdown(md)

    st.download_button("📥 보고서 다운로드 (Markdown)",
                        data=md.encode("utf-8"),
                        file_name="anomaly_detection_report.md",
                        mime="text/markdown")


# ===============================================================
# 푸터
# ===============================================================
st.divider()
st.caption(
    "© 시계열 이상탐지 자동화 대시보드 · 워크플로우 반영 · Streamlit 기반"
)
