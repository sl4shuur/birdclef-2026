import pandas as pd
import pytest

from birdclef.eval.metrics import score


@pytest.fixture
def solution():
    return pd.DataFrame(
        {"row_id": ["a_5", "a_10", "b_5", "b_10"], "bird": [0, 1, 0, 1], "absent": [0] * 4}
    )


@pytest.mark.parametrize("values, expected", [([0.1, 0.8, 0.2, 0.9], 1), ([0.5] * 4, 0.5)])
def test_score_skips_absent_classes_and_preserves_inputs(solution, values, expected):
    predictions = solution.assign(bird=values, absent=0.9)[["row_id", "absent", "bird"]]
    original_solution, original_predictions = solution.copy(), predictions.copy()
    assert score(solution, predictions) == expected
    pd.testing.assert_frame_equal(solution, original_solution)
    pd.testing.assert_frame_equal(predictions, original_predictions)


@pytest.mark.parametrize("bad_ids", [["b_10", "b_5", "a_10", "a_5"], ["a_5"] * 4, [None] * 4])
def test_score_rejects_misaligned_or_invalid_ids(solution, bad_ids):
    with pytest.raises(ValueError, match="row IDs"):
        score(solution, solution.assign(row_id=bad_ids))


def test_score_rejects_matching_duplicate_ids(solution):
    duplicated = solution.assign(row_id="same")
    with pytest.raises(ValueError, match="unique"):
        score(duplicated, duplicated)


def test_score_rejects_class_without_negatives(solution):
    with pytest.raises(ValueError, match="without negative labels.*bird"):
        score(solution.assign(bird=1), solution.assign(bird=0.5))


def test_score_rejects_no_positive_classes(solution):
    with pytest.raises(ValueError, match="at least one class"):
        score(solution.assign(bird=0), solution.assign(bird=0.5))


@pytest.mark.parametrize("invalid", [float("nan"), "not a score"])
def test_score_rejects_invalid_predictions(solution, invalid):
    with pytest.raises(ValueError):
        score(solution, solution.assign(bird=invalid))


def test_score_rejects_missing_species(solution):
    with pytest.raises(ValueError, match="species columns"):
        score(solution, solution.drop(columns="absent"))


def test_evaluation_report_keeps_unobserved_species(solution):
    from birdclef.eval.metrics import evaluate

    predictions = solution.assign(bird=[0.1, 0.9, 0.2, 0.8], absent=0.1)
    auc, report = evaluate(solution, predictions)
    assert auc == 1
    assert report.loc["bird", "auc"] == 1
    assert report.loc["bird", "positive"] == 2
    assert report.loc["bird", "negative"] == 2
    assert pd.isna(report.loc["absent", "auc"])
