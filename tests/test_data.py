import gzip
import json

import pandas as pd

from raglab.config import LANGUAGES, Config
from raglab.data.corpus import SPLITS

DATA = Config().data


def test_pinned_pools_are_complete() -> None:
    for language in LANGUAGES:
        pool = pd.read_parquet(DATA / f"pool/{language}.parquet")
        assert len(pool) == Config().pool_size
        assert pool["id"].is_unique
        assert set(pool["tier"]) == {0, 1, 2}


def test_benchmark_evidence_is_in_the_pool_and_splits_share_no_articles() -> None:
    with (DATA / "benchmark-v2/questions-primary.jsonl").open() as stream:
        questions = [json.loads(line) for line in stream]
    pool = set(pd.read_parquet(DATA / "pool/uk.parquet", columns=["id"])["id"])
    assert all(
        passage in pool for question in questions for group in question["gold_groups"] for passage in group
    )
    articles = {split: {q["article_id"] for q in questions if q["split"] == split} for split in SPLITS}
    assert not articles["train"] & articles["test"] and not articles["dev"] & articles["test"]
    assert not articles["train"] & articles["dev"]


def test_primary_set_is_the_two_of_three_vote() -> None:
    with gzip.open(DATA / "judges/raw-responses.jsonl.gz", "rt") as stream:
        responses = pd.DataFrame(json.loads(line) for line in stream)
    full = responses.groupby("judge").filter(lambda rows: len(rows) > 1000)
    votes = full.groupby("candidate_id")["accepted"].sum()
    with (DATA / "benchmark-v2/questions-primary.jsonl").open() as stream:
        evidence = {item["candidate_id"] for line in stream for item in json.loads(line)["evidence"]}
    assert full["judge"].nunique() == 3
    assert evidence <= set(votes[votes >= 2].index)
