import pytest
from src.evaluation.metrics import StatisticalMetrics

def test_agreement_rate():
    preds = [True, True, False, False, True]
    golds = [True, False, False, False, True]
    assert StatisticalMetrics.agreement_rate(preds, golds) == 0.8

def test_cohens_kappa_perfect():
    preds = [True, False, True, False]
    golds = [True, False, True, False]
    assert StatisticalMetrics.cohens_kappa(preds, golds) == 1.0

def test_pearson_correlation_perfect():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert StatisticalMetrics.pearson_correlation(x, y) == 1.0

def test_spearman_rank_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert StatisticalMetrics.spearman_rank_correlation(x, y) == 1.0

def test_mae_rmse():
    preds = [2.0, 4.0]
    golds = [1.0, 5.0]
    assert StatisticalMetrics.mean_absolute_error(preds, golds) == 1.0
    assert StatisticalMetrics.root_mean_squared_error(preds, golds) == 1.0
