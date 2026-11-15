import shutil

import numpy as np
import pandas as pd
from datasets import load_dataset

from raglab.config import Config
from raglab.data.chunking import chunk_sentences, load_tokenizer, split_sentences
from raglab.data.corpus import Corpus, atomic
from raglab.data.ukwiki import article_id, load_wiki, passage_id
from raglab.embed import PASSAGE, QUERY, encode, load_encoder
from raglab.evaluate import RECALL_AT


def prepare(config: Config, corpus: Corpus) -> None:
    wiki = load_wiki(config)
    row_of = {title: row for row, title in enumerate(wiki["title"])}
    source = load_dataset(config.anchor_repo, "text-UA", revision=config.anchor_revision, split="dev")
    questions = source.to_pandas()
    questions["row"] = questions["wikititle"].str.replace("_", " ").map(row_of)
    mapped = questions.dropna(subset=["row"]).astype({"row": int})
    tokenizer = load_tokenizer(config.encoder, config.encoder_revision)
    chunks = []
    for record in wiki.select(sorted(set(mapped["row"]))):
        article = article_id(record["id"])
        for chunk in chunk_sentences(
            split_sentences(record["text"]), tokenizer, config.chunk_tokens, config.chunk_overlap
        ):
            identity = passage_id(article, chunk.start, chunk.end)
            text = f"{record['title']}\n{record['text'][chunk.start : chunk.end]}"
            chunks.append({"article": article, "row": corpus.row.get(identity, -1), "text": text})
    article_of = dict(zip(mapped["row"], (article_id(wiki[row]["id"]) for row in mapped["row"]), strict=True))
    mapped = mapped.assign(article=mapped["row"].map(article_of))
    mapped[["id", "question_orig", "article"]].to_parquet(
        corpus.dir / "anchor-questions.parquet", index=False
    )
    pd.DataFrame(chunks).to_parquet(corpus.dir / "anchor-chunks.parquet", index=False)
    model = load_encoder(config)
    with atomic(corpus.dir / "anchor-vectors.npz") as path, path.open("wb") as stream:
        np.savez(
            stream,
            questions=encode(model, mapped["question_orig"].tolist(), QUERY),
            chunks=encode(model, [chunk["text"] for chunk in chunks if chunk["row"] < 0], PASSAGE),
        )


def article_recall(
    queries: np.ndarray, vectors: np.ndarray, articles: np.ndarray, gold: list[set[str]]
) -> pd.DataFrame:
    depth = max(RECALL_AT)
    rows = []
    for query, wanted in zip(queries, gold, strict=True):
        scores = vectors @ query
        top = np.argpartition(-scores, depth)[:depth]
        found = articles[top[np.argsort(-scores[top])]]
        rows.append({f"recall_{k}": float(bool(wanted & set(found[:k]))) for k in RECALL_AT})
    return pd.DataFrame(rows)


def run(config: Config) -> None:
    corpus = Corpus(config, "uk")
    path = corpus.dir / "anchor.parquet"
    if not path.exists():
        with atomic(path) as target:
            evaluate(config, corpus).to_parquet(target, index=False)
    config.results.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, config.results / "anchor.parquet")


def evaluate(config: Config, corpus: Corpus) -> pd.DataFrame:
    if not (corpus.dir / "anchor-vectors.npz").exists():
        prepare(config, corpus)
    pool, names = corpus.vectors(), corpus.article_names[corpus.article]
    order = corpus.order(*config.probe_cell)
    questions = pd.read_parquet(corpus.dir / "anchor-questions.parquet")
    chunks = pd.read_parquet(corpus.dir / "anchor-chunks.parquet")
    stored = np.load(corpus.dir / "anchor-vectors.npz")
    gold_rows = chunks.loc[chunks["row"] >= 0, "row"].to_numpy()
    extra_articles = chunks.loc[chunks["row"] < 0, "article"].to_numpy()
    test = corpus.split("test")
    test_queries = corpus.queries(test["id"])
    test_gold = [{names[row] for group in corpus.gold[question] for row in group} for question in test["id"]]
    frames = []
    for size in config.sizes:
        fill = order[~np.isin(order, gold_rows)][: max(0, size - len(chunks))]
        rows = np.sort(np.concatenate([gold_rows, fill]))
        vectors = np.concatenate([pool[rows], stored["chunks"]])
        articles = np.concatenate([names[rows], extra_articles])
        anchor = article_recall(stored["questions"], vectors, articles, [{a} for a in questions["article"]])
        frames.append(
            anchor.assign(
                source="cus-qa", size=size, question_id=questions["id"], cluster=questions["article"]
            )
        )
        rows = corpus.subset(*config.probe_cell, size)
        benchmark = article_recall(test_queries, np.asarray(pool[rows]), names[rows], test_gold)
        frames.append(
            benchmark.assign(
                source="benchmark",
                size=size,
                question_id=test["id"].to_numpy(),
                cluster=test["cluster"].to_numpy(),
            )
        )
    return pd.concat(frames, ignore_index=True).astype({"question_id": str})
