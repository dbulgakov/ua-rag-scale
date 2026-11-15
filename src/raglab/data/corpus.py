import hashlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from raglab.config import Config

SPLITS = ("test", "dev", "train")


def sha256(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def shuffled(items: Iterable[str], salt: str, seed: int = 42) -> list[str]:
    return sorted(items, key=lambda item: sha256(salt, seed, item))


@contextmanager
def atomic(path: Path) -> Iterator[Path]:
    partial = path.with_name(f"{path.name}.partial")
    yield partial
    partial.replace(path)


def article_clusters(questions: pd.DataFrame, article_of: dict[str, str]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first = {}
    for split in SPLITS:
        for row in questions[questions["split"] == split].sort_values("id").itertuples():
            keys = sorted({article_of[passage] for group in row.gold for passage in group})
            if row.component:
                keys.append(row.component)
            for key in keys[1:]:
                parent[find(key)] = find(keys[0])
            first[row.id] = keys[0]
    return {question: find(key) for question, key in first.items()}


def write_corpus(
    config: Config, language: str, passages: pd.DataFrame, questions: pd.DataFrame, links: np.ndarray
) -> None:
    pinned = pq.read_table(config.data / f"pool/{language}.parquet", columns=["id"])["id"].to_pylist()
    if passages["id"].tolist() != pinned:
        raise ValueError(f"The rebuilt {language} pool differs from the pinned data/pool/{language}.parquet")
    article_of = dict(zip(passages["id"], passages["article"], strict=True))
    missing = {p for groups in questions["gold"] for group in groups for p in group} - article_of.keys()
    if missing:
        raise ValueError(f"{len(missing)} gold passages are absent from the {language} pool")
    questions = questions.assign(cluster=questions["id"].map(article_clusters(questions, article_of)))
    folder = config.corpus_dir(language)
    folder.mkdir(parents=True, exist_ok=True)
    passages.to_parquet(folder / "passages.parquet", index=False, row_group_size=20_000)
    questions.to_parquet(folder / "questions.parquet", index=False)
    np.save(folder / "links.npy", links)


class Corpus:
    def __init__(self, config: Config, language: str) -> None:
        self.language = language
        self.dir = config.corpus_dir(language)
        table = pq.read_table(self.dir / "passages.parquet", columns=["id", "article", "tokens"])
        self.ids = np.array(table["id"].to_pylist(), dtype=object)
        self.tokens = table["tokens"].to_numpy()
        self.article_names, self.article = np.unique(table["article"].to_pylist(), return_inverse=True)
        self.questions = pd.read_parquet(self.dir / "questions.parquet")
        self.tier = pq.read_table(config.data / f"pool/{language}.parquet", columns=["tier"])[
            "tier"
        ].to_numpy()
        self.orders: dict[tuple[str, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.ids)

    @cached_property
    def row(self) -> dict[str, int]:
        return {identity: row for row, identity in enumerate(self.ids)}

    @cached_property
    def batches(self) -> list[pa.RecordBatch]:
        return pq.read_table(self.dir / "passages.parquet", columns=["title", "text", "url"]).to_batches()

    @cached_property
    def batch_start(self) -> np.ndarray:
        return np.cumsum([0, *(len(batch) for batch in self.batches)])

    @cached_property
    def links(self) -> sparse.csr_matrix:
        edges = np.load(self.dir / "links.npy")
        size = len(self.article_names)
        matrix = sparse.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(size, size))
        matrix.data[:] = 1.0
        return matrix

    @cached_property
    def gold(self) -> dict[str, list[set[int]]]:
        return {
            question: [{self.row[passage] for passage in group} for group in groups]
            for question, groups in zip(self.questions["id"], self.questions["gold"], strict=True)
        }

    def records(self, rows: list[int]) -> list[dict[str, str]]:
        rows = np.asarray(rows, dtype=np.int64)
        batch = np.searchsorted(self.batch_start, rows, side="right") - 1
        records: list[dict[str, str]] = [{}] * len(rows)
        for number in np.unique(batch):
            positions = np.flatnonzero(batch == number)
            taken = self.batches[number].take(pa.array(rows[positions] - self.batch_start[number]))
            for position, record in zip(positions, taken.to_pylist(), strict=True):
                records[position] = record
        return records

    def texts(self, rows: list[int]) -> list[str]:
        return [f"{record['title']}\n{record['text']}" for record in self.records(rows)]

    def split(self, *splits: str) -> pd.DataFrame:
        return self.questions[self.questions["split"].isin(splits)]

    def vectors(self, variant: str = "base") -> np.ndarray:
        return np.load(self.dir / f"vectors-{variant}.npy", mmap_mode="r")

    def queries(self, ids: Iterable[str], variant: str = "base") -> np.ndarray:
        position = {question: row for row, question in enumerate(self.questions["id"])}
        return np.load(self.dir / f"queries-{variant}.npy")[[position[question] for question in ids]]

    def order(self, order: str, seed: int) -> np.ndarray:
        if (order, seed) not in self.orders:
            priority = np.array([int(sha256("growth", seed, identity)[:15], 16) for identity in self.ids])
            tier = self.tier if order == "hard" else np.where(self.tier == 0, 0, 2)
            self.orders[order, seed] = np.lexsort((priority, tier))
        return self.orders[order, seed]

    def subset(self, order: str, seed: int, size: int) -> np.ndarray:
        return np.sort(self.order(order, seed)[:size])
