import json
import os
import unicodedata
from array import array
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from datasets import Dataset, load_dataset
from mwsql import Dump

from raglab.config import Config
from raglab.data.chunking import chunk_sentences, load_tokenizer, split_sentences
from raglab.data.corpus import sha256, write_corpus


def article_id(page: object) -> str:
    return f"ukwiki:{page}"


def passage_id(article: str, start: int, end: int) -> str:
    return f"native-uk:{article.removeprefix('ukwiki:')}:{start}:{end}"


def load_wiki(config: Config) -> Dataset:
    return load_dataset(config.wiki_repo, config.wiki_name, revision=config.wiki_revision, split="train")


def article_spans(
    batch: dict[str, list], name: str, revision: str, max_tokens: int, overlap: int
) -> dict[str, list]:
    tokenizer = load_tokenizer(name, revision)
    rows: dict[str, list] = {"article": [], "start": [], "end": [], "tokens": []}
    for page, text in zip(batch["id"], batch["text"], strict=True):
        for chunk in chunk_sentences(split_sentences(text), tokenizer, max_tokens, overlap):
            rows["article"].append(article_id(page))
            rows["start"].append(chunk.start)
            rows["end"].append(chunk.end)
            rows["tokens"].append(chunk.tokens)
    return rows


def select_pool(spans: pd.DataFrame, mandatory: list[str], size: int) -> pd.DataFrame:
    counts = spans["article"].value_counts(sort=False)
    if missing := set(mandatory) - set(counts.index):
        raise ValueError(f"Benchmark articles absent from the Wikipedia snapshot: {sorted(missing)}")
    rest = sorted(set(counts.index) - set(mandatory), key=lambda article: sha256("uk-pool-v1", article))
    quota: dict[str, int] = {}
    total = 0
    for article in sorted(mandatory) + rest:
        if total >= size:
            break
        quota[article] = min(int(counts[article]), size - total)
        total += quota[article]
    rank = {article: position for position, article in enumerate(quota)}
    pool = spans[spans["article"].isin(quota)].assign(rank=lambda frame: frame["article"].map(rank))
    pool = pool.sort_values(["rank", "start"], kind="stable")
    return pool[pool.groupby("rank").cumcount() < pool["article"].map(quota)].drop(columns="rank")


def attach_texts(pool: pd.DataFrame, wiki: Dataset) -> pd.DataFrame:
    row_of = {article_id(page): row for row, page in enumerate(wiki["id"])}
    articles = pool["article"].drop_duplicates().tolist()
    source: dict[str, tuple[str, str, str]] = {}
    for start in range(0, len(articles), 5000):
        names = articles[start : start + 5000]
        batch = wiki[[row_of[name] for name in names]]
        source.update(zip(names, zip(batch["title"], batch["url"], batch["text"], strict=True), strict=True))
    records = [
        {
            "id": passage_id(article, start, end),
            "article": article,
            "title": source[article][0],
            "url": source[article][1],
            "text": source[article][2][start:end],
            "tokens": tokens,
        }
        for article, start, end, tokens in pool.itertuples(index=False)
    ]
    return pd.DataFrame.from_records(records)


def load_questions(config: Config) -> pd.DataFrame:
    with (config.data / "benchmark-v2/questions-primary.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    return pd.DataFrame(
        {
            "id": [row["id"] for row in rows],
            "question": [row["question"] for row in rows],
            "answers": [row["answers"] for row in rows],
            "gold": [row["gold_groups"] for row in rows],
            "split": [row["split"] for row in rows],
            "component": [row["component_id"] for row in rows],
            "kind": [row["metadata"]["source_split"] for row in rows],
        }
    )


def title_key(title: str) -> str:
    return unicodedata.normalize("NFC", title.replace("_", " "))


def dump_rows(path: Path, *columns: str) -> Iterator[list[str]]:
    dump = Dump.from_file(path)
    positions = [dump.col_names.index(column) for column in columns]
    for row in dump.rows():
        yield [row[position] for position in positions]


def canonical_pages(
    pages: dict[int, tuple[str, bool]], redirects: dict[int, str | None]
) -> dict[int, int | None]:
    by_title = {title: page for page, (title, _) in pages.items()}
    canonical: dict[int, int | None] = {}
    for start in pages:
        chain: list[int] = []
        current: int | None = start
        while current is not None and current not in canonical and current not in chain:
            chain.append(current)
            if not pages[current][1]:
                canonical[current] = current
                break
            current = by_title.get(redirects.get(current))
        outcome = canonical.get(current) if current is not None else None
        canonical.update(dict.fromkeys(chain, outcome))
    return canonical


def link_graph(config: Config, articles: np.ndarray) -> tuple[np.ndarray, set[str], dict[str, float | int]]:
    paths = {
        table: Path(
            pooch.retrieve(
                url, known_hash=f"sha256:{digest}", fname=url.rsplit("/", 1)[1], path=config.work / "sources"
            )
        )
        for table, (url, digest) in config.link_dumps.items()
    }
    pages = {
        int(page): (title_key(title), redirect == "1")
        for page, namespace, title, redirect in dump_rows(
            paths["page"], "page_id", "page_namespace", "page_title", "page_is_redirect"
        )
        if namespace == "0"
    }
    redirects = {
        int(source): title_key(title) if namespace == "0" and interwiki in ("", "NULL") else None
        for source, namespace, title, interwiki in dump_rows(
            paths["redirect"], "rd_from", "rd_namespace", "rd_title", "rd_interwiki"
        )
    }
    canonical = canonical_pages(pages, redirects)
    by_title = {title: page for page, (title, _) in pages.items()}
    pairs = array("q")
    eligible = unresolved = 0
    for source, namespace, title, source_namespace in dump_rows(
        paths["pagelinks"], "pl_from", "pl_namespace", "pl_title", "pl_from_namespace"
    ):
        if namespace != "0" or source_namespace != "0":
            continue
        eligible += 1
        target = canonical.get(by_title.get(title_key(title)))
        if target is None:
            unresolved += 1
            continue
        origin = canonical.get(int(source))
        if origin is not None and origin != target:
            pairs.append(origin << 32 | target)
    edges = np.unique(np.frombuffer(pairs, dtype=np.int64))
    del pairs
    names = np.unique(articles)
    index = pd.Series(np.arange(len(names)), index=[int(name.removeprefix("ukwiki:")) for name in names])
    source, target = edges >> 32, edges & 0xFFFFFFFF
    inside = np.isin(source, index.index) & np.isin(target, index.index)
    links = np.stack([index.loc[source[inside]].to_numpy(), index.loc[target[inside]].to_numpy()], axis=1)
    nodes = np.array([canonical.get(page) == page for page in index.index])
    summary = {
        "namespace0_pages": len(pages),
        "articles": sum(page == target for page, target in canonical.items()),
        "edges": len(edges),
        "unresolved_link_share": unresolved / eligible,
        "pool_edges": len(links),
        "pool_articles_without_node": float(1 - nodes.mean()),
    }
    return links.astype(np.int32), set(names[nodes]), summary


def build(config: Config) -> dict[str, object]:
    wiki = load_wiki(config)
    load_tokenizer(config.encoder, config.encoder_revision)
    spans = wiki.map(
        article_spans,
        batched=True,
        batch_size=500,
        num_proc=os.cpu_count(),
        remove_columns=wiki.column_names,
        fn_kwargs={
            "name": config.encoder,
            "revision": config.encoder_revision,
            "max_tokens": config.chunk_tokens,
            "overlap": config.chunk_overlap,
        },
    ).to_pandas()
    mandatory = json.loads((config.data / "pool/uk-benchmark-articles.json").read_text())
    pool = select_pool(spans, mandatory, config.pool_size)
    links, nodes, graph = link_graph(config, pool["article"].to_numpy())
    passages = attach_texts(pool, wiki)
    questions = load_questions(config)
    write_corpus(config, "uk", passages, questions, links)
    gold = {
        article_id(passage.split(":")[1])
        for groups in questions.loc[questions["split"] != "train", "gold"]
        for group in groups
        for passage in group
    }
    graph |= {"gold_articles": len(gold), "gold_articles_with_node": len(gold & nodes)}
    return {
        "wikipedia_articles": len(wiki),
        "wikipedia_chunks": len(spans),
        "pool_passages": len(passages),
        "pool_articles": int(passages["article"].nunique()),
        "graph": graph,
    }
