"""Selección de umbral F2 y evaluación de detectores de fraude."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_vectors(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true y scores deben ser arreglos unidimensionales")
    if labels.size == 0 or values.size == 0:
        raise ValueError("y_true y scores no pueden estar vacíos")
    if labels.shape != values.shape:
        raise ValueError("y_true y scores deben tener la misma longitud")
    if not np.isfinite(values).all():
        raise ValueError("scores contiene valores no finitos")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true solo puede contener 0 y 1")
    return labels.astype(np.int32), values


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Resultado de maximizar F2 sobre validación."""

    threshold: float
    f2_score: float
    precision_at_threshold: float
    recall_at_threshold: float
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    thresholds_curve: np.ndarray


@dataclass(frozen=True)
class EvaluationMetrics:
    """Métricas calculadas con un umbral de decisión fijo."""

    name: str
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    f2_score: float
    roc_auc: float
    pr_auc: float
    mcc: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    false_alerts_per_10k_legit: float
    captured_fraud_amount_pct: float | None = None

    @property
    def confusion_matrix(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Devuelve la matriz en el orden [[TN, FP], [FN, TP]]."""
        return (
            (self.true_negatives, self.false_positives),
            (self.false_negatives, self.true_positives),
        )

    def as_dict(self) -> dict[str, str | int | float | None]:
        """Devuelve una representación plana apta para CSV."""
        return {
            "Modelo": self.name,
            "Umbral": self.threshold,
            "Accuracy": self.accuracy,
            "Balanced Accuracy": self.balanced_accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "Especificidad": self.specificity,
            "F1": self.f1_score,
            "F2": self.f2_score,
            "ROC-AUC": self.roc_auc,
            "PR-AUC": self.pr_auc,
            "MCC": self.mcc,
            "TP": self.true_positives,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "TN": self.true_negatives,
            "Falsas alertas / 10.000 legítimas": self.false_alerts_per_10k_legit,
            "Monto de fraude capturado (%)": self.captured_fraud_amount_pct,
        }


class F2ThresholdSelector:
    """Selecciona el umbral que maximiza F2 en validación."""

    def select(self, y_true: np.ndarray, scores: np.ndarray) -> ThresholdSelectionResult:
        labels, values = _validated_vectors(y_true, scores)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("La selección de umbral requiere ambas clases")

        precision, recall, thresholds = precision_recall_curve(labels, values)
        precision = precision[:-1]
        recall = recall[:-1]
        denominator = 4 * precision + recall
        f2 = np.divide(
            5 * precision * recall,
            denominator,
            out=np.zeros_like(precision),
            where=denominator > 0,
        )
        if thresholds.size == 0:
            raise ValueError("No hay umbrales candidatos")
        best_index = int(np.nanargmax(f2))
        return ThresholdSelectionResult(
            threshold=float(thresholds[best_index]),
            f2_score=float(f2[best_index]),
            precision_at_threshold=float(precision[best_index]),
            recall_at_threshold=float(recall[best_index]),
            precision_curve=precision,
            recall_curve=recall,
            thresholds_curve=thresholds,
        )


class ModelEvaluator:
    """Evalúa puntuaciones sin modificar el umbral seleccionado."""

    def evaluate(
        self,
        name: str,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        amounts: np.ndarray | None = None,
    ) -> EvaluationMetrics:
        labels, values = _validated_vectors(y_true, scores)
        if not np.isfinite(threshold):
            raise ValueError("threshold debe ser finito")

        transaction_amounts = None
        if amounts is not None:
            transaction_amounts = np.asarray(amounts, dtype=float)
            if transaction_amounts.ndim != 1 or transaction_amounts.shape != labels.shape:
                raise ValueError("amounts debe tener la misma longitud que y_true")
            if not np.isfinite(transaction_amounts).all():
                raise ValueError("amounts contiene valores no finitos")

        predictions = (values >= threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        has_both_classes = np.unique(labels).size == 2
        balanced_accuracy = (
            balanced_accuracy_score(labels, predictions)
            if has_both_classes
            else float(np.mean(labels == predictions))
        )
        mcc = matthews_corrcoef(labels, predictions) if has_both_classes else 0.0
        captured_percentage = self._captured_amount(
            labels,
            predictions,
            transaction_amounts,
        )

        return EvaluationMetrics(
            name=name,
            threshold=float(threshold),
            accuracy=float(accuracy_score(labels, predictions)),
            balanced_accuracy=float(balanced_accuracy),
            precision=float(precision_score(labels, predictions, zero_division=0)),
            recall=float(recall_score(labels, predictions, zero_division=0)),
            specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
            f1_score=float(f1_score(labels, predictions, zero_division=0)),
            f2_score=float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
            roc_auc=float(roc_auc_score(labels, values)) if has_both_classes else float("nan"),
            pr_auc=(
                float(average_precision_score(labels, values))
                if has_both_classes
                else float("nan")
            ),
            mcc=float(mcc),
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn),
            false_alerts_per_10k_legit=float(fp / (tn + fp) * 10_000) if tn + fp else 0.0,
            captured_fraud_amount_pct=captured_percentage,
        )

    @staticmethod
    def _captured_amount(
        labels: np.ndarray,
        predictions: np.ndarray,
        amounts: np.ndarray | None,
    ) -> float | None:
        if amounts is None:
            return None
        fraud_mask = labels == 1
        total = amounts[fraud_mask].sum()
        if total <= 0:
            return 0.0
        captured = amounts[fraud_mask & (predictions == 1)].sum()
        return float(captured / total * 100)
