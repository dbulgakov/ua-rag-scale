import shutil
import time

import pandas as pd

from raglab.config import Config
from raglab.data.corpus import Corpus, atomic
from raglab.evaluate import (
    KINDS,
    Ranking,
    latency_path,
    latency_questions,
    run_path,
    save_latency,
    save_run,
)
from raglab.index import INDEXES, SEEDED, Index, Lemmas


def cell_questions(config: Config, corpus: Corpus, cell: tuple[str, int, int]) -> pd.DataFrame:
    order, seed, size = cell
    probe = (order, seed) == config.probe_cell and size in config.rerank_sizes
    fit_on_train = config.router_fit_split[corpus.language] == "train"
    return corpus.split("test", "dev", "train") if probe and fit_on_train else corpus.split("test", "dev")


def load_lemmas(config: Config, corpus: Corpus) -> Lemmas | None:
    return Lemmas(corpus) if "hybrid_lemma" in config.methods(corpus.language) else None


def make_index(
    config: Config, corpus: Corpus, cell: tuple[str, int, int], lemmas: Lemmas | None, calibrate: bool = True
) -> Index:
    index = Index(
        config, corpus, corpus.subset(*cell), corpus.vectors(), corpus.vectors("lora"), lemmas, cell[1]
    )
    if calibrate:
        index.calibrate(corpus.queries(corpus.split("dev")["id"]))
    return index


def stored_rankings(corpus: Corpus, cell: tuple[str, int, int], ids: list[str]) -> dict[str, list[Ranking]]:
    runs = pd.read_parquet(run_path(corpus, cell))
    rankings = {}
    for method, part in runs.groupby("method", sort=False):
        part = part.set_index("question_id").loc[ids]
        rankings[method] = [
            (rows.tolist(), scores.tolist())
            for rows, scores in zip(part["rows"], part["scores"], strict=True)
        ]
    return rankings


def run_grid(config: Config, language: str) -> None:
    corpus = Corpus(config, language)
    lemmas = load_lemmas(config, corpus)
    methods = config.methods(language)
    for cell in config.cells():
        order, seed, size = cell
        if run_path(corpus, cell).exists():
            continue
        if size == len(corpus) and order != "random":
            for kind in KINDS:
                with atomic(run_path(corpus, cell, kind)) as target:
                    shutil.copyfile(run_path(corpus, ("random", seed, size), kind), target)
            continue
        questions = cell_questions(config, corpus, cell)
        ids, texts = questions["id"].tolist(), questions["question"].tolist()
        vectors, tuned = corpus.queries(ids), corpus.queries(ids, "lora")
        repeated = size == len(corpus) and seed != config.seeds[0]
        rankings = stored_rankings(corpus, (order, config.seeds[0], size), ids) if repeated else {}
        pending = SEEDED if repeated else methods
        index = make_index(config, corpus, cell, lemmas, calibrate=not repeated)
        for position, method in enumerate(pending):
            rankings[method] = index.search(method, vectors, texts, tuned)
            index.release(method, pending[position + 1 :])
        del index
        save_run(corpus, cell, questions, rankings)
        print(f"{language} {order}-{seed} {size}: done", flush=True)


def measure_latency(config: Config, language: str) -> None:
    corpus = Corpus(config, language)
    lemmas = load_lemmas(config, corpus)
    methods = config.methods(language)
    questions = latency_questions(config, corpus)
    ids = [question["id"] for question in questions]
    vectors, tuned = corpus.queries(ids), corpus.queries(ids, "lora")
    for size in config.rerank_sizes:
        path = latency_path(corpus, "search", size)
        if path.exists():
            continue
        index = make_index(config, corpus, (*config.probe_cell, size), lemmas)
        times: dict[str, list[float]] = {}
        for position, method in enumerate(methods):
            for name in INDEXES[method]:
                getattr(index, name)
            times[method] = []
            for question, base, lora in zip(questions, vectors, tuned, strict=True):
                started = time.perf_counter()
                index.search(method, base[None], [question["question"]], lora[None])
                times[method].append((time.perf_counter() - started) * 1000)
            index.release(method, methods[position + 1 :])
        del index
        save_latency(path, times)
