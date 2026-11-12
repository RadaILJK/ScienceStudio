"""Artificial data verify protocol equivalence, not medical performance."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluation import evaluate_features, validate_features


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(19)
        self.X = rng.normal(size=(30, 12)).astype(np.float32)
        self.y = np.array([0] * 17 + [1] * 13)
        self.ids = np.array(['synthetic-%02d' % i for i in range(30)])
        self.cv = dict(n_splits=5, shuffle=True, random_state=42)
        self.clf = dict(C=0.1, max_iter=3000, class_weight='balanced', random_state=42)

    def test_matches_original_and_participants_are_disjoint(self):
        expected = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(**self.clf)),
            self.X, self.y, cv=StratifiedKFold(**self.cv), method='predict_proba')[:, 1]
        metrics, actual, folds, splits, per_fold = evaluate_features(
            self.X, self.y, self.ids, self.cv, self.clf)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        all_test = []
        for split in splits:
            self.assertFalse(set(split['train_ids']) & set(split['test_ids']))
            all_test.extend(split['test_ids'])
        self.assertEqual(sorted(all_test), sorted(self.ids))
        self.assertEqual(len(per_fold), 5)
        self.assertTrue(np.all(folds >= 0))
        self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), 30)

    def test_rejects_duplicate_subjects(self):
        self.ids[1] = self.ids[0]
        with self.assertRaises(ValueError):
            validate_features(self.X, self.y, self.ids)

    def test_rejects_nonfinite_and_invalid_labels(self):
        self.X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_features(self.X, self.y, self.ids)
        self.X[0, 0] = 0
        self.y[0] = 2
        with self.assertRaises(ValueError):
            validate_features(self.X, self.y, self.ids)


if __name__ == '__main__':
    unittest.main()
