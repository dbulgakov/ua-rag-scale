import numpy as np
import pandas as pd
import pytest

from raglab.config import LANGUAGES, Config
from raglab.router import (
    FEATURES,
    action_scores,
    fit_heads,
    partitions,
    random_routing,
    route,
    select_threshold,
)


def table(language: str = "uk") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for split, clusters in (("train", 12), ("dev", 20), ("test", 10)):
        for cluster in range(clusters):
            for question in range(12):
                crowded = rng.random()
                fast = float(rng.random() > crowded)
                rows.append(
                    {
                        "order": "random",
                        "seed": 42,
                        "size": 10_000,
                        "question_id": f"{split}-{cluster}-{question}",
                        "split": split,
                        "cluster": f"{split}-{cluster}",
                        "fast": fast,
                        "rerank": max(fast, float(rng.random() < 0.7)),
                        "graph": fast,
                    }
                    | dict.fromkeys(FEATURES, crowded)
                )
    frame = pd.DataFrame(rows)
    return frame if language == "uk" else frame[frame["split"] != "train"]


def test_partitions_share_no_article_clusters() -> None:
    config = Config()
    for language in LANGUAGES:
        fit_split = config.router_fit_split[language]
        parts = partitions(table(language), fit_split)
        clusters = {name: set(part["cluster"]) for name, part in parts.items()}
        assert all(not clusters[a] & clusters[b] for a in clusters for b in clusters if a < b)
        assert set(parts["test"]["split"]) == {"test"}
        assert set(parts["fit"]["split"]) == {fit_split}


def test_partitions_reject_leakage() -> None:
    frame = table()
    frame.loc[frame["split"] == "train", "cluster"] = frame.loc[frame["split"] == "test", "cluster"].iloc[0]
    with pytest.raises(ValueError, match="share"):
        partitions(frame, "train")


def test_threshold_is_chosen_without_test_and_respects_the_budget() -> None:
    config = Config()
    parts = partitions(table(), "train")
    selection = parts["selection"]
    scores = action_scores(config, selection, fit_heads(config, parts), "logistic")
    threshold = select_threshold(config, selection, scores)
    choice, _ = route(config, selection, scores, threshold)
    assert choice.notna().mean() <= config.router_budget
    changed = fit_heads(config, parts | {"test": parts["test"].assign(rerank=0.0)})
    assert (
        select_threshold(config, selection, action_scores(config, selection, changed, "logistic"))
        == threshold
    )


def test_random_routing_is_the_expectation_at_equal_budget() -> None:
    frame = pd.DataFrame(
        {
            "order": "random",
            "seed": 42,
            "size": 10,
            "fast": [0.0, 1.0, 0.0, 1.0],
            "rerank": [1.0, 1.0, 1.0, 0.0],
            "graph": 0.0,
        }
    )
    expected = random_routing(Config(), frame, pd.Series(["rerank", None, None, None]))
    assert np.allclose(expected, frame["fast"] + 0.25 * (frame["rerank"] - frame["fast"]))
