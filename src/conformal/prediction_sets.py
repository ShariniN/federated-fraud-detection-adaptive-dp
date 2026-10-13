"""Prediction sets, coverage evaluation and the operational cost model.

Thin wrappers over ``ConformalCalibrator`` that take a trained model and raw
feature matrices, so experiment scripts do not have to handle probabilities.
"""

import numpy as np
from .calibration import ConformalCalibrator
from .scores import predict_proba


def calibrate_model(model, X_cal, y_cal, alpha=0.1, score='lac', mondrian=True,
                    lambda_reg=0.01, device='cpu'):
    probs_cal = predict_proba(model, X_cal, device)
    return ConformalCalibrator(alpha=alpha, score=score, mondrian=mondrian,
                               lambda_reg=lambda_reg).fit(probs_cal, y_cal)


def evaluate_coverage(model, X_test, y_test, calibrator: ConformalCalibrator, device='cpu') -> dict:
    probs = predict_proba(model, X_test, device)
    return calibrator.evaluate(probs, y_test)


def construct_prediction_set(model, X, calibrator: ConformalCalibrator, device='cpu') -> np.ndarray:
    return calibrator.prediction_sets(predict_proba(model, X, device))


def cost_model(tier: dict, n_transactions: int, avg_fraud_loss=250.0, decline_friction=15.0,
               analyst_minutes=4.0, analyst_hourly=45.0) -> dict:
    """Translate review-tier rates into an illustrative operating cost per n transactions.

    The unit costs are placeholders (typical order of magnitude for card fraud
    operations) and must be replaced with the deploying bank's own figures.
    """
    n = float(n_transactions)
    missed = tier.get('fraud_auto_approved', 0)
    wrongly_declined = tier.get('benign_auto_blocked', 0)
    reviewed = tier['rate_human_review'] * n
    return {
        'assumptions': {'avg_fraud_loss': avg_fraud_loss, 'decline_friction': decline_friction,
                        'analyst_minutes': analyst_minutes, 'analyst_hourly': analyst_hourly},
        'n_transactions': int(n),
        'cost_missed_fraud': float(missed * avg_fraud_loss),
        'cost_false_declines': float(wrongly_declined * decline_friction),
        'cost_human_review': float(reviewed * analyst_minutes / 60.0 * analyst_hourly),
    } | {'cost_total': float(missed * avg_fraud_loss + wrongly_declined * decline_friction
                             + reviewed * analyst_minutes / 60.0 * analyst_hourly)}
