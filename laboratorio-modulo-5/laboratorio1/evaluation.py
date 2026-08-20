"""Seleccion de umbral optimo (F2) y metricas de evaluacion para los modelos de fraude."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Resultado de buscar el umbral que maximiza F2 sobre un conjunto de validacion."""

    threshold: float
    f2_score: float
    precision_at_threshold: float
    recall_at_threshold: float
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    thresholds_curve: np.ndarray


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metricas de un modelo evaluadas en un umbral de decision fijo."""

    name: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f2_score: float
    specificity: float
    mcc: float
    roc_auc: float
    pr_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_alerts_per_10k_legit: float
    captured_fraud_amount_pct: float | None = None

    def as_dict(self) -> dict:
        """Representacion plana, util para volcar a un DataFrame/CSV de resultados."""
        return {
            "modelo": self.name,
            "umbral": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1_score,
            "f2": self.f2_score,
            "especificidad": self.specificity,
            "mcc": self.mcc,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "vp": self.true_positives,
            "fp": self.false_positives,
            "vn": self.true_negatives,
            "fn": self.false_negatives,
            "falsas_alertas_10k_legit": self.false_alerts_per_10k_legit,
            "monto_fraude_capturado_pct": self.captured_fraud_amount_pct,
        }


class F2ThresholdSelector:
    """Selecciona el umbral de decision que maximiza F2 sobre un conjunto de validacion.

    Se usa F2 (en vez de F1) porque en deteccion de fraude omitir un fraude real
    (falso negativo) suele ser mas costoso que revisar una alerta que resulta ser
    legitima (falso positivo); F2 pondera el recall con el doble de peso que la
    precision.
    """

    def select(self, y_true: np.ndarray, scores: np.ndarray) -> ThresholdSelectionResult:
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        precision, recall = precision[:-1], recall[:-1]

        denominator = 4 * precision + recall
        f2 = np.divide(
            5 * precision * recall,
            denominator,
            out=np.zeros_like(precision),
            where=denominator > 0,
        )

        if len(f2) == 0:
            raise ValueError("No hay umbrales candidatos: revisa y_true/scores")

        best_idx = int(np.argmax(f2))

        return ThresholdSelectionResult(
            threshold=float(thresholds[best_idx]),
            f2_score=float(f2[best_idx]),
            precision_at_threshold=float(precision[best_idx]),
            recall_at_threshold=float(recall[best_idx]),
            precision_curve=precision,
            recall_curve=recall,
            thresholds_curve=thresholds,
        )


class ModelEvaluator:
    """Calcula metricas de clasificacion para un modelo a un umbral de decision fijo."""

    def evaluate(
        self,
        name: str,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        amounts: np.ndarray | None = None,
    ) -> EvaluationMetrics:
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)
        predictions = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        f2 = self._fbeta(precision, recall, beta=2)
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = matthews_corrcoef(y_true, predictions)

        has_both_classes = len(np.unique(y_true)) > 1
        roc_auc = roc_auc_score(y_true, scores) if has_both_classes else float("nan")
        pr_auc = average_precision_score(y_true, scores) if has_both_classes else float("nan")

        false_alerts_per_10k = (fp / (tn + fp) * 10_000) if (tn + fp) > 0 else 0.0

        captured_pct = None
        if amounts is not None:
            amounts = np.asarray(amounts)
            fraud_mask = y_true == 1
            detected_mask = fraud_mask & (predictions == 1)
            total_fraud_amount = amounts[fraud_mask].sum()
            captured_pct = (
                float(amounts[detected_mask].sum() / total_fraud_amount * 100)
                if total_fraud_amount > 0
                else 0.0
            )

        return EvaluationMetrics(
            name=name,
            threshold=float(threshold),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            f2_score=float(f2),
            specificity=float(specificity),
            mcc=float(mcc),
            roc_auc=float(roc_auc),
            pr_auc=float(pr_auc),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            false_alerts_per_10k_legit=float(false_alerts_per_10k),
            captured_fraud_amount_pct=captured_pct,
        )

    @staticmethod
    def _fbeta(precision: float, recall: float, *, beta: int) -> float:
        denominator = (beta ** 2) * precision + recall
        if denominator == 0:
            return 0.0
        return (1 + beta ** 2) * precision * recall / denominator