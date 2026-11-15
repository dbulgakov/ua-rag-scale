import heapq
import json
import os
import struct
from functools import cache
from pathlib import Path

import hnswlib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from raglab.config import ROOT, Config
from raglab.data.corpus import Corpus, shuffled
from raglab.embed import QUERY, encode, load_encoder
from raglab.generate import (
    OpenRouter,
    cached,
    generation_cache,
    make_messages,
    oracle_passages,
    parse_answer,
)

HEADER = "<6QiI3QdQ"
TRACE_SIZE = 10_000
EXAMPLES = 5
MAX_COST = 0.5
config = Config(out=Path(os.environ.get("RAGLAB_OUT", ROOT)))
app = FastAPI(title="RAG Scale Lab")


class Query(BaseModel):
    question: str
    size: int = 1_000_000
    retrieval: bool = True
    example_id: str | None = None


@cache
def corpus() -> Corpus:
    return Corpus(config, "uk")


@cache
def index(size: int) -> hnswlib.Index:
    path = corpus().dir / "demo" / f"hnsw-{size}.bin"
    result = hnswlib.Index(space="ip", dim=corpus().vectors().shape[1])
    if path.exists():
        result.load_index(str(path))
    else:
        rows = corpus().subset(*config.probe_cell, size)
        result.init_index(
            len(rows), ef_construction=config.hnsw_ef_construction, M=config.hnsw_m, random_seed=42
        )
        result.add_items(np.asarray(corpus().vectors()[rows]), rows)
        path.parent.mkdir(exist_ok=True)
        result.save_index(str(path))
    result.set_ef(config.hnsw_ef_search)
    return result


class LinkTrace:
    def __init__(self, path: Path, size: int) -> None:
        data = path.read_bytes()
        _, _, count, width, label_at, vector_at, self.levels, self.entry, links, *_ = struct.unpack_from(
            HEADER, data
        )
        base = np.frombuffer(data, np.uint8, count * width, struct.calcsize(HEADER)).reshape(count, width)
        self.vectors = base[:, vector_at : vector_at + 4 * size].copy().view(np.float32)
        self.labels = base[:, label_at : label_at + 8].copy().view(np.uint64).ravel()
        self.neighbors = [[self.block(base[node])] for node in range(count)]
        position = struct.calcsize(HEADER) + count * width
        step = links * 4 + 4
        for node in range(count):
            (length,) = struct.unpack_from("<I", data, position)
            upper = np.frombuffer(data, np.uint8, length, position + 4)
            self.neighbors[node] += [self.block(upper[level * step :]) for level in range(length // step)]
            position += 4 + length

    @staticmethod
    def block(raw: np.ndarray) -> np.ndarray:
        count = int(raw[:4].view(np.uint32)[0]) & 0xFFFF
        return raw[4 : 4 + 4 * count].view(np.uint32)

    def search(self, query: np.ndarray, k: int, ef: int) -> tuple[list[int], list[dict[str, int]]]:
        def distance(node: int) -> float:
            return 1.0 - float(self.vectors[node] @ query)

        trace = []
        current, best = int(self.entry), distance(int(self.entry))
        for level in range(self.levels, 0, -1):
            moved = True
            while moved:
                moved = False
                for neighbor in map(int, self.neighbors[current][level]):
                    trace.append({"level": level, "from": current, "to": neighbor})
                    if distance(neighbor) < best:
                        best, current, moved = distance(neighbor), neighbor, True
        visited, candidates, found = {current}, [(best, current)], [(-best, current)]
        while candidates:
            near, node = heapq.heappop(candidates)
            if near > -found[0][0]:
                break
            for neighbor in map(int, self.neighbors[node][0]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                trace.append({"level": 0, "from": node, "to": neighbor})
                far = distance(neighbor)
                if len(found) < ef or far < -found[0][0]:
                    heapq.heappush(candidates, (far, neighbor))
                    heapq.heappush(found, (-far, neighbor))
                    if len(found) > ef:
                        heapq.heappop(found)
        ranked = sorted((-far, node) for far, node in found)[:k]
        return [int(self.labels[node]) for _, node in ranked], trace


@cache
def link_trace() -> LinkTrace:
    index(TRACE_SIZE)
    return LinkTrace(corpus().dir / "demo" / f"hnsw-{TRACE_SIZE}.bin", corpus().vectors().shape[1])


@cache
def examples() -> dict[str, dict]:
    test = corpus().split("test").set_index("id")
    return {question: test.loc[question].to_dict() for question in shuffled(test.index, "stage-d")[:EXAMPLES]}


@cache
def encoder() -> SentenceTransformer:
    return load_encoder(config)


@cache
def generator() -> OpenRouter:
    return OpenRouter(config, MAX_COST)


@cache
def titles() -> dict[int, str]:
    names = (title for batch in corpus().batches for title in batch.column("title").to_pylist())
    return dict(zip(corpus().article.tolist(), names, strict=True))


def embed(question: str) -> np.ndarray:
    return encode(encoder(), [question], QUERY)[0]


def passages(rows: list[int]) -> list[dict]:
    return [
        record | {"id": corpus().ids[row]} for row, record in zip(rows, corpus().records(rows), strict=True)
    ]


def subgraph(rows: list[int], limit: int = 24) -> dict[str, list]:
    links = corpus().links
    seeds = list(dict.fromkeys(int(article) for article in corpus().article[rows]))
    targets = links[seeds].indices
    extra, counts = np.unique(targets[~np.isin(targets, seeds)], return_counts=True)
    nodes = seeds + extra[np.argsort(-counts, kind="stable")][: limit - len(seeds)].tolist()
    title = titles()
    inside = links[nodes][:, nodes].tocoo()
    return {
        "nodes": [
            {
                "id": node,
                "seed": node in seeds,
                "title": title[node],
                "url": "https://uk.wikipedia.org/wiki/" + title[node].replace(" ", "_"),
            }
            for node in nodes
        ],
        "edges": [
            {"source": nodes[a], "target": nodes[b]} for a, b in zip(inside.row, inside.col, strict=True)
        ],
    }


@app.get("/api/meta")
def meta() -> dict:
    path = config.results / "demo.json"
    return {
        "sizes": config.rerank_sizes,
        "examples": [{"id": key, "question": value["question"]} for key, value in examples().items()],
        "research": json.loads(path.read_text()) if path.exists() else None,
    }


@app.post("/api/search")
def search(query: Query) -> dict:
    if query.size not in config.rerank_sizes or not query.question.strip():
        raise HTTPException(400, "Некоректний запит.")
    hits = []
    if query.retrieval:
        labels, _ = index(query.size).knn_query(embed(query.question)[None], k=config.generation_passages)
        hits = passages(labels[0].tolist())
    try:
        messages = make_messages(query.question, hits, "uk")
        reply = cached(generation_cache(config), generator(), config.generator, messages)
        text, citations, _ = parse_answer(reply)
        answer = {"text": text, "citations": citations, "error": None}
    except Exception as error:
        answer = {"text": "", "citations": [], "error": str(error)}
    check = None
    if query.example_id in examples():
        example = examples()[query.example_id]
        groups = [list(group) for group in example["gold"]]
        gold = [corpus().row[passage] for passage in oracle_passages(groups, config.generation_passages)]
        check = {"answers": list(example["answers"]), "passages": passages(gold)}
    graph = subgraph([corpus().row[hit["id"]] for hit in hits]) if hits else None
    return {
        "question": query.question,
        "size": query.size,
        "hits": hits,
        "answer": answer,
        "check": check,
        "graph": graph,
    }


@app.post("/api/trace")
def trace(query: Query) -> dict:
    vector = embed(query.question)
    labels, steps = link_trace().search(vector, 10, config.hnsw_ef_search)
    expected, _ = index(TRACE_SIZE).knn_query(vector[None], k=10)
    order = list(dict.fromkeys([int(link_trace().entry)] + [step["to"] for step in steps]))[:400]
    points = np.vstack([link_trace().vectors[order], vector])
    centered = points - points.mean(axis=0)
    flat = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
    flat = (flat - flat.min(axis=0)) / np.ptp(flat, axis=0).clip(1e-9)
    shown, found = set(order), set(labels[:5])
    return {
        "index_size": TRACE_SIZE,
        "matches_hnswlib": labels == expected[0].tolist(),
        "levels": int(link_trace().levels),
        "query": flat[-1].tolist(),
        "nodes": [
            {"id": node, "x": float(x), "y": float(y), "found": int(link_trace().labels[node]) in found}
            for node, (x, y) in zip(order, flat[:-1], strict=True)
        ],
        "steps": [step for step in steps if step["from"] in shown and step["to"] in shown][:600],
    }


if (ROOT / "web/dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web/dist", html=True), name="web")
