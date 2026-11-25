"""Module 2 scoring helpers: MAP@7 and the two-stage ranking rule."""
import numpy as np
import pandas as pd

from features_reco import PRODUCTS


def apk(actual: list, predicted: list, k: int = 7) -> float:
    """Average precision at k for one customer."""
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actuals: list, predicteds: list, k: int = 7) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)]))


def rank_products(stage2_proba: np.ndarray, classes: list, owned_mask: np.ndarray,
                  stage1_proba: np.ndarray | None = None, top_n: int = 7):
    """
    Rank candidate products per customer.

    stage2_proba : (n, n_classes) P(product | customer acquires something)
    owned_mask   : (n, 24) 1 where the customer already holds the product
    stage1_proba : (n,) P(acquires anything) - scales the scores to an absolute
                   propensity; it is a single per-row factor so it does not change
                   the within-customer ordering, only the magnitude.
    """
    prod_idx = {p: i for i, p in enumerate(PRODUCTS)}
    scores = stage2_proba.copy()
    # never recommend something already held
    for j, cls in enumerate(classes):
        scores[:, j] = np.where(owned_mask[:, prod_idx[cls]] == 1, -1.0, scores[:, j])
    if stage1_proba is not None:
        scores = scores * stage1_proba[:, None]

    order = np.argsort(-scores, axis=1)[:, :top_n]
    classes_arr = np.array(classes)
    recs = [[c for c, s in zip(classes_arr[row], scores[i, row]) if s > 0]
            for i, row in enumerate(order)]
    return recs, scores
