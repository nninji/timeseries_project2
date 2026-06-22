# 📈 시계열 이상탐지 자동화 대시보드

다변량 시계열 CSV 파일을 업로드하면 **자동 진단 → 알고리즘 추천 → Darts 4단 파이프라인 → 평가 → 자동 보고서**까지 한 번에 수행하는 Streamlit 웹앱입니다.

> 본 프로젝트는 시계열 분석 강의(02·06·08·09·10·11·14주차)의 표준 워크플로우를 그대로 반영한 통합 이상탐지 도구입니다.

---

## 🎯 강의 자료(프로젝트 지식) 반영 요소

| 강의 주차 | 반영된 기능 |
|---|---|
| 02 시계열 개요 | 단변량·다변량·정상성·계절성 기본 개념 |
| 06 데이터 전처리 | **결측치 4종 처리(interpolate/ffill/bfill/drop)**, 자동 인코딩(UTF-8/CP949/EUC-KR), 빈도 자동 추정 |
| 08 평활법·분해법 | **STL 분해** (관측·추세·계절성·잔차 4단 시각화) |
| 09 ARIMA | **ADF 정상성 검정**, **Ljung-Box 백색잡음 검정**, **ACF/PACF** 자동 계산·시각화 |
| 10 Darts | TimeSeries 변환, **4종 스케일러 선택**(Standard/Robust/MinMax/Power) |
| 11 Darts 예측 | SKLearnModel, 캘린더 공변량(`hour`, `dayofweek`), **사용자 지정 Train/Test 비율 백테스팅** |
| 14 이상탐지 | **강의록 13페이지 시각화 모두 재현** —<br>p.2 이상의 큰 분류 다이어그램 (Normal/Novelty/Outlier/Anomaly)<br>p.2 시계열 이상 5종 유형 시각화 (Point: Global/Contextual, Pattern: Shapelet/Seasonal/Trend)<br>p.4 Darts 이상탐지 4단 파이프라인 다이어그램 (Scorer/Detector/Aggregator/Anomaly Model)<br>p.7-9 NYC Taxi 알려진 이상 5개 확대 시각화 (Marathon·Thanksgiving·Christmas·New Years·Snow Blizzard)<br>p.9 자동 `lags=one_week=7×24×2=336` 설정 (30분 빈도)<br>p.10 ForecastingAnomalyModel + Scorer 3종 (NormScorer/KMeansScorer/**WassersteinScorer**) 동시 사용<br>p.11-12 `show_anomalies_from_scores` 3단 시각화 (시계열 / Window:1 score_0 / anomalies yes-no)<br>p.13 **Scorer 선택 가이드** 정보 박스 + eval_metric_from_scores AUC-ROC 동등 평가<br>**Darts 4단 구조 그대로 구현**, **이상 유형 자동 분류** (5종) |

---

## ✨ 주요 기능

### 📂 1. 데이터 & 매핑 탭
- 인코딩·구분자 자동 추정 (UTF-8, CP949, EUC-KR, Latin-1, `,` `;` `\t` `|`)
- **시간/값/라벨 컬럼 자동 감지** (이름 + 값 패턴 기반)
- 다변량 시계열 시각화 + Ground Truth 음영 표시
- **수동 이상 구간 지정** (라벨 보강용)
- 4종 샘플 데이터 내장: 합성 다변량, **NYC Taxi(강의 예제)**, 라벨 포함 센서, 라벨 없는 시스템 메트릭

### 🔍 2. 시계열 진단 탭 (자동 통계 분석)
- **STL 분해 4단 시각화** (관측 / 추세 / 계절성 / 잔차)
- **ADF 정상성 검정** (변수별 p-value, 정상성 판정)
- **Ljung-Box 백색잡음 검정** (STL 잔차에 대해)
- **계절성·추세 강도** 정량화 (Hyndman & Athanasopoulos 공식)
- **ACF / PACF** + 95% 신뢰구간 시각화
- **데이터 특성 기반 알고리즘 자동 추천** (이유 명시)

### 🎯 3. 이상탐지 실행 탭 (Darts 4단 구조)
- **10종 알고리즘 통합**:
  - 통계 4종: Z-Score, MAD, IQR, STL 잔차
  - ML 4종: Isolation Forest, One-Class SVM, LOF, KMeans Scorer
  - DL 1종: Autoencoder (MLP 재구성 오차)
  - 혼합 1종: Darts Forecasting (예측 잔차 + NormScorer)
- **ForecastingAnomalyModel 옵션**: 강의 예제 그대로 `SKLearnModel(lags=24, lags_future_covariates=[0]) + add_encoders={cyclic: {future: [hour, dayofweek]}}`
- **Aggregator(앙상블) 3종**: OR / AND / Majority
- 알고리즘별 점수·임계값·탐지 비율 요약

### 📊 4. 평가 대시보드 탭
- **종합 지표 표**: Precision / Recall / F1 / AUC-ROC / AUC-PR / FPR (+TP/FP/FN/TN), 색상 그라데이션
- **F1 기준 최고 성능 자동 강조**
- **알고리즘 간 막대그래프 비교**
- **ROC & Precision-Recall 곡선** (모든 알고리즘 중첩)
- **상세 탐지 시각화**: 시계열 + 점수 + 임계값 + 정답/예측 (강의의 `show_anomalies_from_scores` 스타일)
- **혼동행렬** 히트맵
- **🆕 이상 유형 자동 분류**: 강의 14주차의 5분류 체계 (Global/Contextual/Shapelet/Seasonal/Trend) 도넛 차트 + 구간 테이블
- 결과 CSV 다운로드

### 📑 5. 분석 보고서 탭
- 모든 분석 결과를 종합한 **Markdown 보고서 자동 생성**
- 진단·추천·평가·유형분류·결론을 한 문서에 정리
- 다운로드 버튼으로 `.md` 파일 저장

---

## 🚀 빠른 시작

### 로컬 실행

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 `http://localhost:8501` 을 자동으로 엽니다. 좌측 **샘플 데이터** 라디오에서 **"NYC Taxi (강의 예제)"** 를 선택하면 강의 14주차 예제와 동일한 데이터로 즉시 체험할 수 있습니다.

---

## 🌐 온라인 배포 (Streamlit Community Cloud · 무료)

```bash
git init
git add .
git commit -m "init: time series anomaly detection app"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git push -u origin main
```

1. <https://share.streamlit.io> 접속 → GitHub 로그인
2. **New app** → 리포지토리·브랜치(`main`)·메인 파일(`app.py`) 선택 → **Deploy**
3. 1~3분 내 `https://<your-app>.streamlit.app/` 공개 URL 발급

> 메모리가 제한된 환경(Streamlit Cloud free tier 1GB)에서 ForecastingAnomalyModel 기능이 무거우면 사이드바 토글로 비활성화하면 됩니다. 다른 9개 알고리즘은 그대로 사용 가능합니다.

---

## 📂 프로젝트 구조

```
anomaly_app/
├── app.py                  # Streamlit 메인 (5개 탭 UI)
├── utils.py                # CSV 로드·컬럼 자동 감지·라벨 처리
├── detection.py            # 10종 이상탐지 알고리즘
├── metrics.py              # 평가지표 + ROC/PR 곡선
├── diagnosis.py            # 🆕 STL/ADF/Ljung-Box/ACF·PACF + 알고리즘 추천
├── anomaly_types.py        # 🆕 점이상/패턴이상 자동 분류
├── darts_pipeline.py       # 🆕 Darts 4단 구조 (Scorer/Detector/Aggregator/Anomaly Model)
├── report.py               # 🆕 Markdown 분석 보고서 자동 생성
├── requirements.txt
├── README.md
├── DEMO_SCRIPT.md          # 3분 시연 동영상 스크립트
├── .streamlit/config.toml
├── .gitignore
└── sample_data/
    ├── nyc_taxi.csv                    # 🆕 강의 예제 데이터 (10,320행, 30분 빈도)
    ├── multivariate_sensor.csv         # 라벨 포함 다변량 (온도·습도·전력)
    └── system_metrics_nolabel.csv      # 라벨 없는 시스템 메트릭
```

---

## 📋 CSV 입력 형식

```csv
timestamp,temperature,humidity,power_usage,is_anomaly
2024-01-01 00:00:00,18.5,59.8,102.3,0
2024-01-01 01:00:00,17.9,61.2,98.7,0
2024-01-01 02:00:00,32.1,45.3,180.5,1
...
```

| 컬럼 | 자동 감지 규칙 |
|---|---|
| 시간 | 이름이 `time/date/timestamp/datetime/날짜/일자/ds` 이거나, 값이 80% 이상 날짜로 파싱 가능 |
| 값 | 수치형, 단일값 아님, 이진 0/1 아님, 이름이 라벨로 보이지 않음 |
| 라벨 | 이름이 `label/anomaly/outlier/이상/라벨` 이거나, 값이 `{0,1}` `{True,False}` `{normal,anomaly}` 의 2진 분류 |

자동 감지가 잘못되면 앱 내 **컬럼 매핑** 영역에서 직접 지정 가능.

---

## 🧪 알고리즘 가이드 & 추천 로직

| 데이터 특성 | 추천 알고리즘 | 이유 |
|---|---|---|
| 계절성 강도 ≥ 0.5 | STL 잔차, Darts Forecasting | 분해 후 잔차 기반이 효과적 |
| 추세 강도 ≥ 0.5 | Darts Forecasting | 예측 잔차가 비정상성에 강건 |
| 정상 시계열 비율 < 50% | Autoencoder | 비선형 패턴 학습 |
| 변수 3개 이상 | Isolation Forest, LOF | 고차원에서 효율적 |
| 정상 시계열 70% 이상 | Z-Score, MAD | 단순 통계 베이스라인 적합 |
| (항상) | Isolation Forest | 범용 베이스라인 |

앱이 데이터 진단 결과를 보고 이 규칙들에 따라 자동으로 **이유와 함께** 추천 알고리즘 리스트를 제시합니다.

---

## 📊 평가지표 해석

- **Precision** = TP / (TP+FP) — 오탐 최소화 관점
- **Recall (탐지율)** = TP / (TP+FN) — 누락 최소화 관점
- **F1** = 조화평균 — 균형
- **AUC-ROC** — 임계값 무관 분리 능력
- **AUC-PR** — 불균형 환경(이상 비율 낮음)에서 더 신뢰 가능
- **FPR** = FP / (FP+TN) — 정상을 이상으로 본 비율

이상탐지에서는 통상 **AUC-PR**과 **F1**을 우선 봅니다. 강의 14주차의 `eval_metric_from_scores(metric="AUC_ROC")`도 본 앱에서 동일하게 제공됩니다.

---

## 📑 이상 유형 분류 (강의 14주차)

| 유형 | 정의 | 판단 기준 |
|---|---|---|
| **Global (점이상-전역)** | 전체 시계열에서 정상 범주를 크게 벗어난 단일 점 | length=1, 전체 |z| > 3 |
| **Contextual (점이상-맥락)** | 전체로는 정상이지만 인접 구간 대비 이상 | length=1, 전체 |z| ≤ 3 |
| **Shapelet (패턴-모양)** | 짧은 구간에 일반과 다른 모양 | 짧은 다중 시점 |
| **Seasonal (패턴-계절)** | 계절성과 어긋난 부분 시계열 | 주기 근처 길이 |
| **Trend (패턴-추세)** | 영구적 추세 변화 | 매우 긴 구간 + 평균 이동 |

---

## 🔧 트러블슈팅

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: darts` | `pip install darts` (없어도 폴백으로 자동 동작) |
| CSV 한글 깨짐 | 자동 인코딩 탐지 동작. 그래도 안되면 UTF-8로 재저장 |
| 시간 컬럼 인식 실패 | 컬럼명을 `timestamp` 또는 `date`로 변경, 또는 매핑에서 직접 선택 |
| STL 분해 실패 | 데이터 길이 < 2×주기+5 필요. 더 긴 데이터 사용 |
| Streamlit Cloud OOM | 사이드바의 "ForecastingAnomalyModel 사용" 체크 해제 |

---

## 📚 참고 자료

- 강의자료: 14주차 시계열 이상탐지 (Chunghun Ha, Hongik University)
- Darts AD 문서: <https://unit8co.github.io/darts/examples/22-anomaly-detection-examples.html>
- Hyndman & Athanasopoulos: <https://otexts.com/fpp3/>
- Scikit-learn outlier detection: <https://scikit-learn.org/stable/modules/outlier_detection.html>
