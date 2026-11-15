import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from functools import cached_property

import bm25s
import faiss
import numpy as np
import pymorphy3
from bm25s.tokenization import Tokenized
from scipy import sparse

from raglab.config import Config
from raglab.data.corpus import Corpus, atomic
from raglab.evaluate import Ranking

WORD = re.compile(r"(?u)\b\w\w+\b")
BLOCK = 20_000
INDEXES = {
    "exact": ("exact",),
    "hnsw": ("hnsw",),
    "ivfpq": ("ivfpq",),
    "adaptive": ("hnsw",),
    "finetuned": ("finetuned",),
    "hybrid": ("hnsw", "bm25"),
    "hybrid_lemma": ("hnsw", "bm25_lemma"),
    "graph": ("hnsw",),
    "graph_no_ppr": ("hnsw",),
    "graph_exact": ("exact",),
}
SEEDED = ("ivfpq",)


def rrf(rankings: list[list[int]], k: int, limit: int) -> Ranking:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1 / (k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    return [item for item, _ in ordered], [score for _, score in ordered]


def personalized_pagerank(
    adjacency: sparse.csr_matrix,
    personalization: np.ndarray,
    restart: float,
    iterations: int = 200,
    tolerance: float = 1e-9,
) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree, dtype=float), where=degree > 0)
    transition = (sparse.diags(inverse) @ adjacency).T
    dangling = degree == 0
    start = personalization / personalization.sum()
    rank = start.copy()
    for _ in range(iterations):
        updated = (1 - restart) * (transition @ rank + rank[dangling].sum() * start) + restart * start
        if np.abs(updated - rank).sum() < tolerance * len(rank):
            return updated
        rank = updated
    return rank


def plain_tokens(texts: Iterable[str]) -> Tokenized:
    return bm25s.tokenize((text.replace("́", "") for text in texts), stopwords=None, show_progress=False)


def blocks(corpus: Corpus, rows: np.ndarray) -> Iterator[str]:
    for start in range(0, len(rows), BLOCK):
        yield from corpus.texts(rows[start : start + BLOCK].tolist())


class LemmaDocuments:
    def __init__(self, ids: np.ndarray, offsets: np.ndarray, rows: np.ndarray) -> None:
        self.ids, self.offsets, self.rows = ids, offsets, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[list[int]]:
        return (self.ids[self.offsets[row] : self.offsets[row + 1]].tolist() for row in self.rows)


class Lemmas:
    def __init__(self, corpus: Corpus) -> None:
        self.analyzer = pymorphy3.MorphAnalyzer(lang="uk")
        self.lemma: dict[str, str] = {}
        path = corpus.dir / "lemmas.npz"
        if not path.exists():
            vocabulary: dict[str, int] = {}
            ids, offsets = [], [0]
            for text in blocks(corpus, np.arange(len(corpus))):
                ids.extend(vocabulary.setdefault(lemma, len(vocabulary)) for lemma in self(text))
                offsets.append(len(ids))
            path.with_suffix(".json").write_text(json.dumps(list(vocabulary), ensure_ascii=False))
            with atomic(path) as target, target.open("wb") as stream:
                np.savez(stream, ids=np.array(ids, dtype=np.int32), offsets=np.array(offsets, dtype=np.int64))
            self.lemma = {}
        stored = np.load(path)
        self.ids, self.offsets = stored["ids"], stored["offsets"]
        self.vocabulary = {
            word: i for i, word in enumerate(json.loads(path.with_suffix(".json").read_text()))
        }

    def __call__(self, text: str) -> list[str]:
        result = []
        for token in WORD.findall(text.replace("́", "").lower()):
            if token not in self.lemma:
                self.lemma[token] = self.analyzer.parse(token)[0].normal_form
            result.append(self.lemma[token])
        return result

    def documents(self, rows: np.ndarray) -> Tokenized:
        return Tokenized(ids=LemmaDocuments(self.ids, self.offsets, rows), vocab=self.vocabulary)


class Index:
    def __init__(
        self,
        config: Config,
        corpus: Corpus,
        rows: np.ndarray,
        vectors: np.ndarray,
        finetuned: np.ndarray,
        lemmas: Lemmas | None,
        seed: int,
    ) -> None:
        self.config = config
        self.corpus = corpus
        self.rows = np.asarray(rows, dtype=np.int64)
        self.vectors = np.ascontiguousarray(vectors[self.rows], dtype=np.float32)
        self.finetuned_vectors = finetuned
        self.lemmas = lemmas
        self.seed = seed
        self.ef = config.hnsw_ef_search
        self.article = corpus.article[self.rows]
        order = np.argsort(self.article, kind="stable")
        self.by_article = order
        self.article_start = np.searchsorted(self.article[order], np.arange(len(corpus.article_names) + 1))
        self.present = np.zeros(len(corpus.article_names), dtype=bool)
        self.present[self.article] = True

    def build_hnsw(self, vectors: np.ndarray) -> faiss.Index:
        index = faiss.IndexHNSWFlat(vectors.shape[1], self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.config.hnsw_ef_construction
        index.add(vectors)
        index.hnsw.efSearch = self.config.hnsw_ef_search
        return index

    @cached_property
    def exact(self) -> faiss.Index:
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        return index

    @cached_property
    def hnsw(self) -> faiss.Index:
        return self.build_hnsw(self.vectors)

    @cached_property
    def finetuned(self) -> faiss.Index:
        return self.build_hnsw(np.ascontiguousarray(self.finetuned_vectors[self.rows], dtype=np.float32))

    @cached_property
    def ivfpq(self) -> faiss.Index:
        config, size = self.config, self.vectors.shape[1]
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatIP(size),
            size,
            config.ivf_lists,
            config.pq_segments,
            config.pq_bits,
            faiss.METRIC_INNER_PRODUCT,
        )
        sample = np.random.default_rng(self.seed).choice(
            len(self.vectors), min(len(self.vectors), config.ivf_training_vectors), replace=False
        )
        index.train(self.vectors[np.sort(sample)])
        index.add(self.vectors)
        index.nprobe = config.ivf_probes
        return index

    def build_bm25(self, documents: Tokenized) -> bm25s.BM25:
        index = bm25s.BM25(k1=self.config.bm25_k1, b=self.config.bm25_b)
        index.index(documents, show_progress=False)
        return index

    @cached_property
    def bm25(self) -> bm25s.BM25:
        return self.build_bm25(plain_tokens(blocks(self.corpus, self.rows)))

    @cached_property
    def bm25_lemma(self) -> bm25s.BM25:
        return self.build_bm25(self.lemmas.documents(self.rows))

    def dense(
        self, index: faiss.Index, queries: np.ndarray, k: int, ef: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if ef is not None:
            index.hnsw.efSearch = ef
        scores, found = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        if ef is not None:
            index.hnsw.efSearch = self.config.hnsw_ef_search
        return found, scores

    def lexical(self, index: bm25s.BM25, queries: Tokenized | list[list[str]]) -> np.ndarray:
        k = min(self.config.candidates, len(self.rows))
        return index.retrieve(queries, k=k, show_progress=False, n_threads=8).documents

    def calibrate(self, queries: np.ndarray) -> None:
        exact, _ = self.dense(self.exact, queries, 10)
        recalls: dict[int, float] = {}
        for ef in self.config.adaptive_ef:
            found, _ = self.dense(self.hnsw, queries, 10, ef)
            recalls[ef] = float(
                np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, exact, strict=True)])
            )
            if recalls[ef] >= self.config.adaptive_target:
                break
        self.ef = max(recalls)

    def release(self, method: str, later: Iterable[str]) -> None:
        needed = {name for other in later for name in INDEXES[other]}
        for name in set(INDEXES[method]) - needed:
            self.__dict__.pop(name, None)

    def article_rows(self, article: int) -> np.ndarray:
        return self.by_article[self.article_start[article] : self.article_start[article + 1]]

    def expand(self, query: np.ndarray, seeds: np.ndarray, use_ppr: bool) -> Ranking:
        config, links = self.config, self.corpus.links
        weights: dict[int, float] = defaultdict(float)
        for rank, article in enumerate(self.article[seeds], 1):
            weights[int(article)] += 1 / (config.rrf_k + rank)
        seed_rows = set(seeds.tolist())
        candidates, frontier, expanded = set(seed_rows), set(weights), set()
        for _ in range(config.graph_hops):
            neighbors = np.unique(links[sorted(frontier)].indices)
            neighbors = neighbors[self.present[neighbors]]
            proposed = candidates | {int(row) for article in neighbors for row in self.article_rows(article)}
            if len(proposed) > config.graph_limit:
                extra = np.array(sorted(proposed - seed_rows))
                best = np.argsort(-(self.vectors[extra] @ query), kind="stable")
                proposed = seed_rows | set(extra[best[: config.graph_limit - len(seed_rows)]].tolist())
            candidates = proposed
            expanded |= frontier
            frontier = {int(article) for article in self.article[list(candidates)]} - expanded
            if not frontier:
                break
        rows = np.array(sorted(candidates))
        dense = self.vectors[rows] @ query
        by_score = np.lexsort((rows, -dense))
        dense_order = rows[by_score]
        if not use_ppr:
            return dense_order[: config.top_k].tolist(), dense[by_score][: config.top_k].tolist()
        articles = np.unique(self.article[rows])
        position = {article: i for i, article in enumerate(articles)}
        personalization = np.zeros(len(articles))
        for article, weight in weights.items():
            personalization[position[article]] = weight
        ppr = personalized_pagerank(links[articles][:, articles], personalization, config.ppr_restart)
        row_ppr = ppr[[position[article] for article in self.article[rows]]]
        graph_order = rows[np.lexsort((rows, -dense, -row_ppr))]
        return rrf([dense_order.tolist(), graph_order.tolist()], config.rrf_k, config.top_k)

    def search(
        self, method: str, queries: np.ndarray, texts: list[str], finetuned_queries: np.ndarray | None = None
    ) -> list[Ranking]:
        config = self.config
        if method in ("hybrid", "hybrid_lemma"):
            dense, _ = self.dense(self.hnsw, queries, config.candidates)
            if method == "hybrid":
                lexical = self.lexical(self.bm25, plain_tokens(texts))
            else:
                lexical = self.lexical(self.bm25_lemma, [self.lemmas(text) for text in texts])
            fused = [
                rrf([d[d >= 0].tolist(), found.tolist()], config.rrf_k, config.top_k)
                for d, found in zip(dense, lexical, strict=True)
            ]
        elif method.startswith("graph"):
            seeds, _ = self.dense(
                self.exact if method == "graph_exact" else self.hnsw, queries, config.graph_seeds
            )
            fused = [
                self.expand(query, seed[seed >= 0], use_ppr=method != "graph_no_ppr")
                for query, seed in zip(queries, seeds, strict=True)
            ]
        else:
            if method == "hnsw":
                found, scores = self.dense(self.hnsw, queries, config.candidates)
            elif method == "adaptive":
                found, scores = self.dense(self.hnsw, queries, config.top_k, self.ef)
            elif method == "finetuned":
                found, scores = self.dense(self.finetuned, finetuned_queries, config.top_k)
            else:
                found, scores = self.dense(getattr(self, method), queries, config.top_k)
            fused = [(f[f >= 0].tolist(), s[f >= 0].tolist()) for f, s in zip(found, scores, strict=True)]
        return [(self.rows[items].tolist(), list(scores)) for items, scores in fused]
