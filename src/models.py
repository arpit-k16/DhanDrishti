"""Fraud model used in notebook 02 and the dashboard.

Model selection (rolling time-series CV: Feb, Mar, Apr folds, test never touched) showed that with only
~240 training frauds neither a linear nor a tree model wins on its own: logistic regression captures smooth
risk gradients, a shallow gradient-boosting model captures interactions (e.g. COLLECT x new payee). Averaging
them was the best or near-best model on every fold, so that is what we ship.

The blend averages each model's log-odds after z-scoring them with TRAINING statistics, so a new payment can
be scored on its own (no batch or test-set information needed).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=5, max_iter=3000))


def make_gbm():
    return HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, min_samples_leaf=80,
                                          l2_regularization=5.0, max_iter=500, early_stopping=False,
                                          random_state=42)


def _logodds(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


class FraudBlend(BaseEstimator, ClassifierMixin):
    """Average of z-scored log-odds from a logistic regression and a shallow gradient-boosting model."""

    def fit(self, X, y):
        self.models_ = [make_logit().fit(X, y), make_gbm().fit(X, y)]
        z = [_logodds(m.predict_proba(X)[:, 1]) for m in self.models_]
        self.stats_ = [(v.mean(), v.std()) for v in z]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        z = [(_logodds(m.predict_proba(X)[:, 1]) - mu) / sd for m, (mu, sd) in zip(self.models_, self.stats_)]
        return np.mean(z, axis=0)

    def predict_proba(self, X):
        s = 1 / (1 + np.exp(-self.decision_function(X)))  # monotone risk score in (0, 1), NOT a calibrated probability
        return np.c_[1 - s, s]
