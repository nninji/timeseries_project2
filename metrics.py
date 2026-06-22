"""
metrics.py
- 이상탐지 성능 평가지표
"""
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    """이상탐지 성능 지표 계산.
       y_true, y_pred: 0/1
       y_score: 이상 점수 (선택)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {}
    # 정상-only인 경우 ROC/AUC 계산 불가
    try:
        out["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        out["Precision"] = out["Recall"] = out["F1"] = 0.0
    try:
        if y_score is not None and len(np.unique(y_true)) > 1:
            out["AUC-ROC"] = float(roc_auc_score(y_true, y_score))
            out["AUC-PR"] = float(average_precision_score(y_true, y_score))
        else:
            out["AUC-ROC"] = float("nan")
            out["AUC-PR"] = float("nan")
    except Exception:
        out["AUC-ROC"] = float("nan")
        out["AUC-PR"] = float("nan")
    # 혼동행렬
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        out["TN"], out["FP"], out["FN"], out["TP"] = (
            int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        )
    except Exception:
        out["TN"] = out["FP"] = out["FN"] = out["TP"] = 0
    # 탐지율, 오탐율
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    out["탐지율(Detection Rate)"] = out["Recall"]
    out["오탐율(FPR)"] = float(out["FP"] / n_neg) if n_neg > 0 else float("nan")
    out["이상 비율(true)"] = float(n_pos / len(y_true)) if len(y_true) > 0 else 0.0
    out["이상 비율(pred)"] = float(np.mean(y_pred == 1)) if len(y_pred) > 0 else 0.0
    return out


def get_roc_curve(y_true, y_score):
    """ROC curve points."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "threshold": thr}


def get_pr_curve(y_true, y_score):
    """Precision-Recall curve points."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    return {"precision": precision, "recall": recall, "threshold": thr}
