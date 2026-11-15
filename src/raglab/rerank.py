import time

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from raglab.config import Config
from raglab.data.corpus import Corpus, atomic
from raglab.evaluate import latency_path, latency_questions, run_path, save_latency, save_run

CHECKPOINT = 20_000


def load_reranker(config: Config) -> CrossEncoder:
    model = CrossEncoder(
        config.reranker,
        revision=config.reranker_revision,
        device=config.device,
        max_length=config.rerank_max_length,
    )
    if model.device.type != "cpu":
        model.model.half()
    return model


def score_pairs(model: CrossEncoder, config: Config, pairs: list[tuple[str, str]]) -> np.ndarray:
    order = np.argsort([len(question) + len(passage) for question, passage in pairs], kind="stable")
    batch = config.rerank_batch * (4 if model.device.type == "cuda" else 1)
    scores = np.empty(len(pairs), dtype=np.float32)
    scores[order] = model.predict([pairs[i] for i in order], batch_size=batch, show_progress_bar=False)
    return scores


def best(rows: list[int], scores: list[float], limit: int) -> tuple[list[int], list[float]]:
    order = sorted(range(len(rows)), key=lambda position: -scores[position])[:limit]
    return [rows[position] for position in order], [scores[position] for position in order]


def rerank(config: Config, language: str) -> None:
    corpus = Corpus(config, language)
    depth = config.rerank_depth[language]
    folder = corpus.dir / "rerank"
    folder.mkdir(exist_ok=True)
    parts = [pd.read_parquet(path) for path in sorted(folder.glob("*.parquet"))]
    known = {(q, row): score for part in parts for q, row, score in part.itertuples(index=False)}
    text_of = dict(zip(corpus.questions["id"], corpus.questions["question"], strict=True))
    model = None
    for cell in config.rerank_cells():
        if run_path(corpus, cell, suffix="-rerank").exists():
            continue
        runs = pd.read_parquet(run_path(corpus, cell))
        runs = runs[runs["method"] == "hnsw"]
        candidates = [[int(row) for row in rows[:depth]] for rows in runs["rows"]]
        wanted = {(q, row) for q, rows in zip(runs["question_id"], candidates, strict=True) for row in rows}
        missing = sorted(wanted - known.keys())
        for start in range(0, len(missing), CHECKPOINT):
            model = model or load_reranker(config)
            chunk = missing[start : start + CHECKPOINT]
            passages = corpus.texts([row for _, row in chunk])
            pairs = [(text_of[q], passage) for (q, _), passage in zip(chunk, passages, strict=True)]
            scores = score_pairs(model, config, pairs)
            frame = pd.DataFrame(
                {"question_id": [q for q, _ in chunk], "row": [r for _, r in chunk], "score": scores}
            )
            with atomic(folder / f"{len(list(folder.glob('*.parquet'))):05d}.parquet") as target:
                frame.to_parquet(target, index=False)
            known.update(zip(chunk, scores.tolist(), strict=True))
            print(f"{language} {cell}: {start + len(chunk)}/{len(missing)} pairs", flush=True)
        rankings: dict[str, list] = {}
        for method, limit in {"rerank": depth, "rerank_top50": 50}.items():
            if method == "rerank" or depth > limit:
                rankings[method] = [
                    best(rows[:limit], [known[q, row] for row in rows[:limit]], config.top_k)
                    for q, rows in zip(runs["question_id"], candidates, strict=True)
                ]
        questions = corpus.questions.set_index("id", drop=False).loc[runs["question_id"]]
        save_run(corpus, cell, questions, rankings, suffix="-rerank")


def measure_latency(config: Config, language: str) -> None:
    corpus = Corpus(config, language)
    depth = config.rerank_depth[language]
    model = load_reranker(config)
    for size in config.rerank_sizes:
        path = latency_path(corpus, "rerank", size)
        if path.exists():
            continue
        runs = pd.read_parquet(run_path(corpus, (*config.probe_cell, size)))
        rows_of = runs[runs["method"] == "hnsw"].set_index("question_id")["rows"]
        times = []
        for question in latency_questions(config, corpus):
            passages = corpus.texts([int(row) for row in rows_of[question["id"]][:depth]])
            started = time.perf_counter()
            score_pairs(model, config, [(question["question"], passage) for passage in passages])
            times.append((time.perf_counter() - started) * 1000)
        save_latency(path, {"rerank_scoring": times})
