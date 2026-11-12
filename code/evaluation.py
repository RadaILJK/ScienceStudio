"""Subject-level evaluation of fixed BrainIAC features (no MRI imports)."""
from __future__ import annotations

import time
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def validate_features(X, y, ids, n_splits=5):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    ids = np.asarray(ids).astype(str)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError('X must be a nonempty 2D feature matrix.')
    if y.ndim != 1 or ids.ndim != 1 or len(X) != len(y) or len(y) != len(ids):
        raise ValueError('X, y and ids must describe the same participants.')
    if not np.isfinite(X).all() or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError('Expected finite features and labels 0=HC, 1=ULD.')
    if len(set(ids)) != len(ids) or any(not s.strip() for s in ids):
        raise ValueError('Each participant ID must occur exactly once.')
    y = y.astype(int)
    if min(np.bincount(y)) < n_splits:
        raise ValueError('Too few participants per class for stratified CV.')
    return X, y, ids


def score_predictions(y, probabilities, threshold=0.5):
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'oof_log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
        'sensitivity_ULD': float(tp / (tp + fn)),
        'specificity_HC': float(tn / (tn + fp)),
        'confusion_matrix': matrix.tolist(),
        'classification_report': classification_report(
            y, predicted, labels=[0, 1], target_names=['HC', 'ULD'],
            output_dict=True, zero_division=0),
    }


def evaluate_features(X, y, ids, cv_config, classifier_config, threshold=0.5):
    """Same estimators/splits as the original cross_val_predict, with logging.

    Scaler and classifier are fitted together on training participants only.
    The pooled OOF AUC matches the aggregation used in the original notebook.
    """
    X, y, ids = validate_features(X, y, ids, cv_config['n_splits'])
    estimator = make_pipeline(StandardScaler(), LogisticRegression(**classifier_config))
    cv = StratifiedKFold(**cv_config)
    probabilities = np.full(len(y), np.nan, dtype=float)
    fold_index = np.full(len(y), -1, dtype=int)
    assignments, fold_metrics = [], []
    started = time.perf_counter()
    for fold, (train, test) in enumerate(cv.split(X, y)):
        if set(ids[train]) & set(ids[test]):
            raise ValueError('Participant leakage between train and test.')
        model = clone(estimator)
        fit_start = time.perf_counter()
        model.fit(X[train], y[train])
        probabilities[test] = model.predict_proba(X[test])[:, 1]
        fold_index[test] = fold
        row = score_predictions(y[test], probabilities[test], threshold)
        row.update({
            'fold': fold, 'n_train': len(train), 'n_test': len(test),
            'fit_and_predict_seconds': time.perf_counter() - fit_start,
            # This is cross-entropy, not the full regularized objective.
            'train_weighted_log_loss_without_penalty': float(log_loss(
                y[train], model.predict_proba(X[train])[:, 1], labels=[0, 1],
                sample_weight=compute_sample_weight('balanced', y[train]))),
        })
        fold_metrics.append(row)
        assignments.append({'fold': fold,
            'train_indices': train.tolist(), 'test_indices': test.tolist(),
            'train_ids': ids[train].tolist(), 'test_ids': ids[test].tolist()})
    if np.any(fold_index < 0) or not np.isfinite(probabilities).all():
        raise RuntimeError('Missing OOF predictions.')
    metrics = score_predictions(y, probabilities, threshold)
    metrics.update({'n_subjects': len(y), 'n_HC': int((y == 0).sum()),
        'n_ULD': int((y == 1).sum()), 'feature_dimension': X.shape[1],
        'aggregation': 'pooled out-of-fold predictions, not mean fold AUC',
        'threshold': threshold, 'cv_seconds': time.perf_counter() - started,
        'status': 'computed_by_this_run'})
    return metrics, probabilities, fold_index, assignments, fold_metrics
