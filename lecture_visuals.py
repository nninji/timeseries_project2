"""
lecture_visuals.py
- 강의 14주차 강의록의 모든 시각화를 그대로 재현
- p.2  : 시계열 이상 5종 유형 예시 (Point: Global/Contextual, Pattern: Shapelet/Seasonal/Trend)
- p.2  : 이상의 유형 분류 다이어그램 (Normal/Novelty/Outlier/Anormaly)
- p.4  : Darts 4단 파이프라인 다이어그램 (Scorer→Detector→Aggregator→Anomaly Model)
- p.7-9: NYC Taxi 알려진 이상 5개 확대 시각화
- p.11-12: show_anomalies_from_scores 정확 재현 (3단: Time series / Window:1 score_0 / anomalies yes-no)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 1) p.2 - 5종 이상 유형 예시 그래프 (강의록의 작은 그래프 그대로 재현)
# ============================================================
def make_anomaly_types_grid() -> go.Figure:
    """강의 14주차 p.2의 'Time Series Anomaly' 5종 분류 그림을 Plotly로 재현.
       Point: Global / Contextual
       Pattern: Shapelet / Seasonal / Trend
    """
    n = 100
    t = np.arange(n)
    rng = np.random.default_rng(0)

    # Global: 단일 점이 전체에서 크게 벗어남
    global_y = np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.15, n)
    global_y[40] = 4.0  # 명확한 글로벌 이상
    global_anom = [40]

    # Contextual: 단일 점이 인접 맥락과 다름 (글로벌로는 정상 범주)
    ctx_y = np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.15, n)
    ctx_y[60] = -1.4  # 60번째: 주변보다 낮음 (sin이 양수 근처)
    ctx_anom = [60]

    # Shapelet: 짧은 구간에 다른 모양/사이클
    shap_y = np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.1, n)
    shap_y[15:25] = 1.0  # 평평한 모양 (정상은 사인파)
    shap_range = (15, 24)

    # Seasonal: 계절성에서 벗어나는 부분 (위상 깨짐)
    seas_y = np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.1, n)
    seas_y[35:60] = np.sin(2 * np.pi * (t[35:60]) / 8)  # 빠른 주기 삽입
    seas_range = (35, 59)

    # Trend: 추세 영구 변화
    trend_y = 0.005 * t + rng.normal(0, 0.05, n)
    trend_y[70:] += np.linspace(0, 0.8, 30)  # 급격한 상승 추세
    trend_range = (70, 99)

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=("<b>Global</b><br><sub>(점이상)</sub>",
                        "<b>Contextual</b><br><sub>(점이상)</sub>",
                        "<b>Shapelet</b><br><sub>(패턴이상)</sub>",
                        "<b>Seasonal</b><br><sub>(패턴이상)</sub>",
                        "<b>Trend</b><br><sub>(패턴이상)</sub>"),
        horizontal_spacing=0.04, shared_yaxes=False,
    )

    series_list = [
        (global_y, global_anom, None),
        (ctx_y, ctx_anom, None),
        (shap_y, None, shap_range),
        (seas_y, None, seas_range),
        (trend_y, None, trend_range),
    ]
    for i, (y, points, rng_) in enumerate(series_list, start=1):
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines",
                                   line=dict(color="#222", width=1)),
                      row=1, col=i)
        if points:
            fig.add_trace(go.Scatter(x=points, y=[y[p] for p in points],
                                       mode="markers",
                                       marker=dict(color="red", size=10,
                                                    line=dict(color="red", width=2),
                                                    symbol="circle-open")),
                          row=1, col=i)
        if rng_:
            fig.add_vrect(x0=rng_[0], x1=rng_[1], fillcolor="red", opacity=0.18,
                            line_width=0, row=1, col=i)

    # 그림 위쪽에 그룹 라벨 (Point / Pattern)
    fig.add_annotation(text="<b>Point</b>", xref="paper", yref="paper",
                         x=0.10, y=1.18, showarrow=False,
                         font=dict(size=14, color="#444"))
    fig.add_annotation(text="<b>Pattern</b>", xref="paper", yref="paper",
                         x=0.72, y=1.18, showarrow=False,
                         font=dict(size=14, color="#444"))

    fig.update_layout(
        height=260, showlegend=False,
        margin=dict(l=10, r=10, t=80, b=10),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="Time", showgrid=True, gridcolor="#eee",
                      title_font_size=10)
    fig.update_yaxes(title_text="Input Time Series", col=1,
                      showgrid=True, gridcolor="#eee", title_font_size=10)
    return fig


# ============================================================
# 2) p.2 - 이상 분류 다이어그램 (Normal / Novelty / Outlier / Anormaly)
# ============================================================
def make_normal_anomaly_diagram_html() -> str:
    """이상의 큰 분류 (Normal/Novelty/Outlier/Anomaly) 다이어그램을 HTML로."""
    return """
    <style>
    .ano-card {display:flex; gap:14px; flex-wrap:wrap; margin:10px 0;}
    .ano-box {flex:1; min-width:200px; padding:12px 14px; border-radius:10px;
               border:1px solid #d6dbe0; background:#fafbfc;}
    .ano-box h4 {margin:0 0 6px 0; font-size:14px;}
    .ano-box p  {margin:0; font-size:12px; color:#444; line-height:1.4;}
    .ano-normal  {background:#eaf2ff; border-color:#b8cef0;}
    .ano-novel   {background:#e7f5ff; border-color:#9ec9ed;}
    .ano-outlier {background:#fdebeb; border-color:#f0b8b8;}
    .ano-anomaly {background:#fce6e6; border-color:#ed9c9c;}
    </style>
    <div class='ano-card'>
      <div class='ano-box ano-normal'>
        <h4>🔵 Normal <span style='font-weight:400'>(정상)</span></h4>
        <p>대부분을 구성하는 주류 데이터. 데이터의 속성과 분포가 일반적인 패턴을 따름.</p>
      </div>
      <div class='ano-box ano-novel'>
        <h4>🟦 Novelty <span style='font-weight:400'>(신규)</span></h4>
        <p>정상과 속성이 유사하나 새로운 특성을 가짐. 발생 가능성이 낮음.
           <b>→ 탐색의 대상 (긍정적)</b></p>
      </div>
      <div class='ano-box ano-outlier'>
        <h4>🟥 Outlier <span style='font-weight:400'>(이상치)</span></h4>
        <p>정상과 속성이 다르고 정상으로부터 완전히 벗어난 데이터(움직임이 적은).
           주로 수치형 데이터에서의 이상.
           <b>→ 통계적 방법으로 검출 / 처리의 대상</b></p>
      </div>
      <div class='ano-box ano-anomaly'>
        <h4>🟥 Anomaly <span style='font-weight:400'>(이상)</span></h4>
        <p>정상과 속성이 다르고 발생 가능성이 낮은 데이터. 주로 이미지·시계열에서의 이상.
           <b>→ 딥러닝 방법으로 검출 / 탐색의 대상</b></p>
      </div>
    </div>
    """


# ============================================================
# 3) p.4 - Darts 4단 파이프라인 다이어그램 (HTML)
# ============================================================
def make_darts_pipeline_diagram_html() -> str:
    """강의 p.4의 Darts 이상탐지 4단 다이어그램을 HTML로 재현."""
    return """
    <style>
    .darts-flow {display:flex; flex-direction:column; gap:8px; margin:8px 0;}
    .darts-row {display:grid; grid-template-columns: 1fr 50px 1fr 50px 1fr;
                 align-items:center; gap:4px;}
    .darts-block {padding:10px; border-radius:8px; text-align:center;
                   border:1px solid #d6dbe0; font-size:12px;}
    .darts-arrow {text-align:center; font-size:20px; color:#888;}
    .stage-name {font-weight:600; font-size:13px; padding:8px 0;}
    .scorer    {background:#6a4a3c; color:white;}
    .detector  {background:#c0392b; color:white;}
    .aggregator{background:#a052b6; color:white;}
    .anom-model{background:#3a6dab; color:white;}
    .data-block{background:#eef3f8; color:#333;}
    .score-block{background:#fce8e0; color:#333;}
    .pred-block{background:#ffd9d9; color:#333;}
    .multi-block{background:#fff4cf; color:#333;}
    </style>
    <div class='darts-flow'>
      <div class='darts-row'>
        <div class='darts-block data-block'>Time series<br><small>(시계열)</small></div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block scorer'><div class='stage-name'>Scorer</div>
          <small>ex: KMeansScorer</small></div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block score-block'>Anomaly score<br><small>(이상 점수)</small></div>
      </div>
      <div class='darts-row'>
        <div class='darts-block score-block'>Anomaly score</div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block detector'><div class='stage-name'>Detector</div>
          <small>ex: ThresholdDetector</small></div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block pred-block'>Binary prediction<br><small>(이진 예측)</small></div>
      </div>
      <div class='darts-row'>
        <div class='darts-block multi-block'>Multiple binary predictions<br>
          <small>(여러 알고리즘 결과)</small></div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block aggregator'><div class='stage-name'>Aggregator</div>
          <small>ex: OrAggregator</small></div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block pred-block'>Binary prediction<br><small>(단일 예측)</small></div>
      </div>
      <div class='darts-row'>
        <div class='darts-block data-block'>Time series</div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block anom-model'>
          <div class='stage-name'>Anomaly Model</div>
          <small>Forecasting/Filtering Model + Scorer<br>(ex: ARIMA + NormScorer)</small>
        </div>
        <div class='darts-arrow'>→</div>
        <div class='darts-block score-block'>Anomaly score</div>
      </div>
    </div>
    <ul style='font-size:12px; margin-top:8px; color:#444;'>
      <li><b>Scorers</b>: 대상 시계열에 대한 이상 점수를 계산</li>
      <li><b>Detectors</b>: 시계열의 이상 점수를 이진 시계열로 변환 (1=이상)</li>
      <li><b>Aggregators</b>: 다변량 이진 시계열을 단변량 이진 시계열로 축소</li>
      <li><b>Anomaly Models</b>: 글로벌 예측/필터링 모델을 사용하여 이상 점수 생성</li>
    </ul>
    """


# ============================================================
# 4) p.7-9 - NYC Taxi 알려진 이상 5개 확대 시각화
# ============================================================
KNOWN_NYC_ANOMALIES = {
    "NYC Marathon":  ("2014-11-02", "2014-11-02"),
    "Thanksgiving":  ("2014-11-27", "2014-11-27"),
    "Christmas":     ("2014-12-24", "2014-12-25"),
    "New Years":     ("2014-12-31", "2015-01-01"),
    "Snow Blizzard": ("2015-01-26", "2015-01-27"),
}


def make_nyc_known_anomalies_zoom(ts_df: pd.DataFrame,
                                    value_col: str,
                                    delta_days: int = 3) -> go.Figure:
    """강의 p.7-9의 plot_anom 함수를 재현.
       각 알려진 이상에 대해 ±delta_days 만큼 확대한 5개의 subplot."""
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=list(KNOWN_NYC_ANOMALIES.keys()),
        vertical_spacing=0.06,
    )
    if not isinstance(ts_df.index, pd.DatetimeIndex):
        return fig

    for i, (name, (s_str, e_str)) in enumerate(KNOWN_NYC_ANOMALIES.items(), start=1):
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str) + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)
        start = s - pd.Timedelta(days=delta_days)
        end = e + pd.Timedelta(days=delta_days)
        seg = ts_df.loc[(ts_df.index >= start) & (ts_df.index <= end), value_col]
        if seg.empty:
            continue
        fig.add_trace(go.Scatter(x=seg.index, y=seg.values, mode="lines",
                                   name="Number of taxi passengers",
                                   line=dict(color="#6464ff", width=1),
                                   showlegend=(i == 1)),
                       row=i, col=1)
        # 알려진 이상 구간: 강의록처럼 빨간 line + 10000 스케일 (y축이 다르므로 ymax * 0.45)
        ymax = float(seg.max())
        fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.15, line_width=0,
                        row=i, col=1)
        fig.add_trace(go.Scatter(
            x=[s, e, e, s, s],
            y=[0, 0, ymax * 0.45, ymax * 0.45, 0],
            mode="lines", line=dict(color="red", width=1.2),
            name="Known anomaly", showlegend=(i == 1),
        ), row=i, col=1)
    fig.update_layout(
        height=900, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(title_text="time", row=5, col=1)
    return fig


# ============================================================
# 5) p.11-12 - show_anomalies_from_scores 정확 재현
# ============================================================
def show_anomalies_from_scores_plot(ts_df: pd.DataFrame,
                                       value_col: str,
                                       scores: np.ndarray,
                                       pred: np.ndarray,
                                       title: str = "이상 탐지 결과",
                                       score_label: str = "score_0",
                                       series_label: str = "#Passengers",
                                       window: int = 1) -> go.Figure:
    """강의 14주차 p.11-12의 show_anomalies_from_scores() 함수를 그대로 재현.
       3단 구성: 시계열 / Window: N + score / anomalies (yes/no)
    """
    n = len(ts_df)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("", f"Window: {window}", ""),
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
    )

    # 1행: 시계열 (강의록과 동일한 파란색)
    fig.add_trace(go.Scatter(
        x=ts_df.index, y=ts_df[value_col].values, mode="lines",
        line=dict(color="#1f77b4", width=1), name=series_label,
    ), row=1, col=1)

    # 2행: 이상 점수
    fig.add_trace(go.Scatter(
        x=ts_df.index, y=np.asarray(scores).flatten(), mode="lines",
        line=dict(color="#1f77b4", width=1), name=score_label,
    ), row=2, col=1)

    # 3행: anomalies (yes/no) - 강의록의 빨간 step
    fig.add_trace(go.Scatter(
        x=ts_df.index, y=np.asarray(pred).astype(int), mode="lines",
        line=dict(color="red", width=1.2, shape="hv"),
        name="anomalies",
    ), row=3, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["no", "yes"], row=3, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=620, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(title_text="time", row=3, col=1)
    return fig


# ============================================================
# 6) p.13 - Scorer 선택 가이드 HTML
# ============================================================
def scorer_guide_html() -> str:
    """강의 p.13의 '이상 탐지 모델 선택 가이드' 박스를 재현."""
    return """
    <style>
    .scorer-guide {border-left:4px solid #4ca64c; padding:10px 14px;
                    background:#eef9ec; border-radius:6px; margin:8px 0;
                    font-size:13px;}
    .scorer-guide h4 {margin:0 0 6px 0; color:#2b6e2b;}
    .scorer-guide ul {margin:4px 0 0 18px; padding:0;}
    .scorer-guide li {margin:3px 0;}
    .scorer-guide b {color:#2b6e2b;}
    </style>
    <div class='scorer-guide'>
      <h4>💡 이상 탐지 모델 선택 가이드 (강의 14주차 p.13)</h4>
      <ul>
        <li><b>NormScorer</b>: 예측 오차의 크기만으로 이상을 판단. 간단하고 해석이 쉬움</li>
        <li><b>KMeansScorer</b>: 오차 패턴을 클러스터링하여 이상을 판단. 복잡한 패턴에 유리</li>
        <li><b>WassersteinScorer</b>: 분포 간 거리로 이상을 판단. 분포 변화 감지에 효과적</li>
        <li>Scorer를 여러 개 조합하면 다양한 유형의 이상을 탐지할 수 있음</li>
      </ul>
    </div>
    """
