import numpy as np
import pandas as pd

from raglab.config import Config
from raglab.data.specificity import Lemmatizer, tokens
from raglab.evaluate import (
    answer_scores,
    cluster_interval,
    contains_answer,
    lemma_match,
    normalize_answer,
    paired_difference,
    retrieval_scores,
)


def test_retrieval_scores_use_evidence_groups() -> None:
    scores = retrieval_scores([7, 3, 9, 1, 5, 2], [{1, 4}, {8}], reference={7, 3})
    assert scores["recall_1"] == 0.0
    assert scores["recall_5"] == 0.5
    assert scores["mrr_10"] == 0.25
    assert scores["all_support_10"] == 0.0
    assert scores["ann_recall_10"] == 1.0


def test_answer_scores_follow_squad_rules() -> None:
    assert answer_scores("The Eiffel Tower.", ["Eiffel Tower"], "en") == (1.0, 1.0)
    assert answer_scores("yes", ["no"], "en") == (0.0, 0.0)
    exact, f1 = answer_scores("у Києві", ["Київ", "в Києві"], "uk")
    assert exact == 0.0
    assert f1 == 0.5
    assert normalize_answer("Ки́їв!", "uk") == "київ"


def test_lemma_match_forgives_case_endings() -> None:
    lemmatize = Lemmatizer("uk")
    assert lemma_match("Києві", ["Київ"], lemmatize, tokens) == 1.0
    assert lemma_match("у Києві", ["Київ"], lemmatize, tokens) == 0.0


def test_answer_search_respects_word_boundaries() -> None:
    assert contains_answer(["у 1990 році"], ["1990"], "uk")
    assert not contains_answer(["код 19901"], ["1990"], "uk")


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    config = Config()
    values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    low, high = cluster_interval(values, np.array(["a", "a", "a", "b", "b", "b"]), config)
    assert (low, high) == (0.0, 1.0)
    assert cluster_interval(values, np.array(["a"] * 6), config) == (None, None)
    independent = cluster_interval(values, np.arange(6), config)
    assert 0.0 <= independent[0] < 0.5 < independent[1] <= 1.0
    assert independent == cluster_interval(values, np.arange(6), config)


def test_paired_difference_keeps_questions_paired() -> None:
    frame = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2", "q3"],
            "method": ["a", "b", "a", "b", "a"],
            "cluster": ["x", "x", "y", "y", "z"],
            "recall_5": [1.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    result = paired_difference(frame, Config(), "a", "b", "recall_5")
    assert result["questions"] == 2
    assert result["difference"] == 0.5
