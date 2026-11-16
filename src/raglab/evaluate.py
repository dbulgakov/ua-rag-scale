import json
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

from raglab.config import LANGUAGES, Config
from raglab.data.corpus import Corpus, atomic, shuffled

Ranking = tuple[list[int], list[float]]

RECALL_AT = (1, 5, 10, 20)
ANN_METHODS = ("hnsw", "adaptive", "ivfpq")
KINDS = ("metrics", "runs")


def normalize_answer(value: str, language: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("́", "").casefold()
    if language == "en":
        value = "".join(character for character in value if character not in string.punctuation)
        value = re.sub(r"\b(a|an|the)\b", " ", value)
    else:
        value = "".join(
            character for character in value if not unicodedata.category(character).startswith("P")
        )
    return " ".join(value.split())


def answer_scores(prediction: str, answers: list[str], language: str) -> tuple[float, float]:
    prediction = normalize_answer(prediction, language)
    best_match = best_f1 = 0.0
    for answer in answers:
        answer = normalize_answer(answer, language)
        best_match = max(best_match, float(prediction == answer))
        closed = {"yes", "no", "noanswer"}
        if language == "en" and (prediction in closed or answer in closed) and prediction != answer:
            continue
        predicted, expected = prediction.split(), answer.split()
        if not predicted or not expected:
            best_f1 = max(best_f1, float(predicted == expected))
        else:
            common = sum((Counter(predicted) & Counter(expected)).values())
            best_f1 = max(best_f1, 2 * common / (len(predicted) + len(expected)))
    return best_match, best_f1


def lemma_match(prediction: str, answers: list[str], lemmatize: Callable, tokens: Callable) -> float:
    predicted = [lemmatize(token) for token in tokens(prediction)]
    for answer in answers:
        expected = [lemmatize(token) for token in tokens(answer)]
        same_length = expected and len(expected) == len(predicted)
        if same_length and all(a & b for a, b in zip(predicted, expected, strict=True)):
            return 1.0
    return 0.0


def normalized_texts(corpus: Corpus, rows: Iterable[int]) -> dict[int, str]:
    rows = sorted(set(rows))
    records = corpus.records(rows)
    return {
        row: normalize_answer(record["text"], corpus.language)
        for row, record in zip(rows, records, strict=True)
    }


def contains_answer(texts: Iterable[str], answers: list[str], language: str) -> bool:
    normalized = (normalize_answer(answer, language) for answer in answers)
    patterns = [re.compile(rf"(?<!\w){re.escape(answer)}(?!\w)") for answer in normalized if answer]
    return any(pattern.search(text) for text in texts for pattern in patterns)


def retrieval_scores(found: list[int], gold: list[set[int]], reference: set[int] | None = None) -> dict:
    relevant = set().union(*gold)
    scores = {
        f"recall_{k}": float(np.mean([bool(group & set(found[:k])) for group in gold])) for k in RECALL_AT
    }
    scores["mrr_10"] = next((1 / rank for rank, row in enumerate(found[:10], 1) if row in relevant), 0.0)
    scores["all_support_10"] = float(all(group & set(found[:10]) for group in gold))
    scores["ann_recall_10"] = len(set(found[:10]) & reference) / len(reference) if reference else None
    return scores


def ratio(sums: np.ndarray, counts: np.ndarray, axis: int = -1) -> np.ndarray:
    return sums.sum(axis=axis) / counts.sum(axis=axis)


def cluster_interval(
    values: np.ndarray, clusters: np.ndarray, config: Config
) -> tuple[float | None, float | None]:
    frame = (
        pd.DataFrame({"value": values, "cluster": clusters}).groupby("cluster")["value"].agg(["sum", "count"])
    )
    if len(frame) < 2:
        return None, None
    result = stats.bootstrap(
        (frame["sum"].to_numpy(), frame["count"].to_numpy()),
        ratio,
        paired=True,
        vectorized=True,
        n_resamples=config.bootstrap_resamples,
        random_state=config.bootstrap_seed,
        confidence_level=0.95,
        method="percentile",
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def with_hits(scored: pd.DataFrame) -> pd.DataFrame:
    recall = {k: scored[f"recall_{k}"] for k in RECALL_AT}
    hits = {f"hit_{k}": (found > 0).astype(float).where(found.notna()) for k, found in recall.items()}
    return scored.assign(**hits, all_support_5=(recall[5] == 1).astype(float).where(recall[5].notna()))


def curves(scored: pd.DataFrame, config: Config, metric: str = "recall_5") -> pd.DataFrame:
    keys = ["order", "method", "size"]
    rows = []
    for values, frame in scored[scored[metric].notna()].groupby(keys):
        question = frame.groupby("question_id").agg(value=(metric, "mean"), cluster=("cluster", "first"))
        low, high = cluster_interval(question["value"].to_numpy(), question["cluster"].to_numpy(), config)
        cell = {metric: question["value"].mean(), "low": low, "high": high}
        rows.append(dict(zip(keys, values, strict=True)) | cell)
    return pd.DataFrame(rows)


def paired_difference(frame: pd.DataFrame, config: Config, method: str, baseline: str, metric: str) -> dict:
    selected = frame[frame["method"].isin([method, baseline])]
    table = selected.groupby(["question_id", "method"])[metric].mean().unstack("method").dropna()
    if method not in table or baseline not in table or table.empty:
        return {"difference": None, "low": None, "high": None, "questions": 0}
    difference = (table[method] - table[baseline]).to_numpy()
    clusters = selected.groupby("question_id")["cluster"].first().loc[table.index].to_numpy()
    low, high = cluster_interval(difference, clusters, config)
    return {"difference": float(difference.mean()), "low": low, "high": high, "questions": len(table)}


def run_path(corpus: Corpus, cell: tuple[str, int, int], kind: str = "runs", suffix: str = "") -> Path:
    order, seed, size = cell
    return corpus.dir / kind / f"{order}-{seed}-{size}{suffix}.parquet"


def probe_features(scores: list[float], articles: np.ndarray) -> dict[str, float]:
    top = np.asarray(scores[:50], dtype=float)
    weights = softmax(top)
    return {
        "gap_1_2": top[0] - top[1],
        "gap_1_10": top[0] - top[9],
        "gap_1_50": top[0] - top[-1],
        "entropy": float(-(weights * np.log(weights)).sum() / np.log(50)),
        "articles_10": len(set(articles[:10])),
        "articles_20": len(set(articles[:20])),
    }


def score_rankings(
    corpus: Corpus, questions: pd.DataFrame, rankings: dict[str, list[Ranking]]
) -> pd.DataFrame:
    texts = normalized_texts(
        corpus, (row for ranked in rankings.values() for rows, _ in ranked for row in rows[:5])
    )
    exact = [set(rows[:10]) for rows, _ in rankings["exact"]] if "exact" in rankings else None
    specific = set(questions.loc[questions["low_specificity"].eq(False).fillna(False), "id"])
    records = []
    for method, ranked in rankings.items():
        for position, (question, (rows, scores)) in enumerate(
            zip(questions.itertuples(), ranked, strict=True)
        ):
            reference = exact[position] if exact and method in ANN_METHODS else None
            record = {"method": method, "question_id": question.id}
            record |= retrieval_scores(rows, corpus.gold[question.id], reference)
            if question.id in specific:
                top = (texts[row] for row in rows[:5])
                record["answer_in_top5"] = float(contains_answer(top, question.answers, corpus.language))
            if method == "hnsw":
                record |= probe_features(scores, corpus.article[rows[:50]])
            records.append(record)
    return pd.DataFrame(records)


def save_run(
    corpus: Corpus,
    cell: tuple[str, int, int],
    questions: pd.DataFrame,
    rankings: dict[str, list[Ranking]],
    suffix: str = "",
) -> None:
    runs = pd.DataFrame(
        {"method": method, "question_id": question, "rows": rows, "scores": scores}
        for method, ranked in rankings.items()
        for question, (rows, scores) in zip(questions["id"], ranked, strict=True)
    )
    for kind, frame in zip(KINDS, (score_rankings(corpus, questions, rankings), runs), strict=True):
        path = run_path(corpus, cell, kind, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic(path) as target:
            frame.to_parquet(target, index=False)


def cell_tables(corpus: Corpus, kind: str, cells: str = "*") -> pd.DataFrame:
    frames = []
    for path in sorted((corpus.dir / kind).glob(f"{cells}.parquet")):
        order, seed, size = path.stem.split("-")[:3]
        frame = pd.read_parquet(path)
        frames.append(frame.assign(language=corpus.language, order=order, seed=int(seed), size=int(size)))
    return pd.concat(frames, ignore_index=True)


def latency_path(corpus: Corpus, kind: str, size: int) -> Path:
    return corpus.dir / "latency" / f"{kind}-{size}.json"


def latency_questions(config: Config, corpus: Corpus) -> list[pd.Series]:
    questions = corpus.split("test").head(config.latency_queries)
    return [questions.iloc[0], *(row for _, row in questions.iterrows())]


def save_latency(path: Path, times: dict[str, list[float]]) -> None:
    medians = {name: float(np.percentile(values[1:], 50)) for name, values in times.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic(path) as target:
        target.write_text(json.dumps(medians, indent=2))


def collect(config: Config) -> None:
    config.results.mkdir(parents=True, exist_ok=True)
    questions, metrics, retrieved, latency = [], [], [], []
    for language in LANGUAGES:
        corpus = Corpus(config, language)
        columns = ["id", "split", "cluster", "kind", "low_specificity"]
        questions.append(corpus.questions[columns].assign(language=language))
        metrics.append(cell_tables(corpus, "metrics"))
        sample = shuffled(corpus.split("test")["id"], "stage-d")[: config.generation_questions]
        order, seed = config.probe_cell
        runs = cell_tables(corpus, "runs", f"{order}-{seed}-*")
        chosen = runs[
            runs["size"].isin(config.rerank_sizes)
            & runs["method"].isin(config.generation_methods)
            & runs["question_id"].isin(sample)
        ]
        passages = [corpus.ids[list(rows[: config.generation_passages])].tolist() for rows in chosen["rows"]]
        retrieved.append(chosen[["language", "size", "method", "question_id"]].assign(passages=passages))
        for path in sorted((corpus.dir / "latency").glob("*.json")):
            size = int(path.stem.split("-")[1])
            latency += [
                {"language": language, "size": size, "method": method, "p50_ms": value}
                for method, value in json.loads(path.read_text()).items()
            ]
    pd.concat(questions).to_parquet(config.results / "questions.parquet", index=False)
    pd.concat(metrics).to_parquet(config.results / "retrieval.parquet", index=False, compression="zstd")
    pd.concat(retrieved).to_parquet(config.results / "retrieved.parquet", index=False)
    pd.DataFrame(latency).to_parquet(config.results / "latency.parquet", index=False)
