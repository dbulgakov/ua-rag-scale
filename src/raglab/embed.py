from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from raglab.config import Config
from raglab.data.corpus import Corpus, atomic

BLOCK = 16_384
QUERY, PASSAGE = "query: ", "passage: "


def load_encoder(config: Config, adapter: Path | None = None) -> SentenceTransformer:
    model = SentenceTransformer(config.encoder, revision=config.encoder_revision, device=config.device)
    if adapter:
        model.load_adapter(str(adapter))
    return model


def encode(model: SentenceTransformer, texts: list[str], prefix: str) -> np.ndarray:
    batch = 256 if model.device.type == "cuda" else 64
    vectors = model.encode([prefix + text for text in texts], batch_size=batch, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def embed(config: Config, language: str, variant: str = "base", adapter: Path | None = None) -> None:
    corpus = Corpus(config, language)
    target = corpus.dir / f"vectors-{variant}.npy"
    queries = corpus.dir / f"queries-{variant}.npy"
    if target.exists() and queries.exists():
        return
    model = load_encoder(config, adapter)
    with atomic(queries) as path, path.open("wb") as stream:
        np.save(stream, encode(model, corpus.questions["question"].tolist(), QUERY))
    partial = target.with_suffix(".partial.npy")
    progress = target.with_suffix(".rows")
    done = int(progress.read_text()) if progress.exists() and partial.exists() else 0
    size = model.get_sentence_embedding_dimension()
    vectors = np.lib.format.open_memmap(
        partial, mode="r+" if done else "w+", dtype=np.float32, shape=(len(corpus), size)
    )
    for start in range(done, len(corpus), BLOCK):
        rows = list(range(start, min(start + BLOCK, len(corpus))))
        vectors[rows] = encode(model, corpus.texts(rows), PASSAGE)
        vectors.flush()
        with atomic(progress) as path:
            path.write_text(str(rows[-1] + 1))
        print(f"{language} {variant}: {rows[-1] + 1}/{len(corpus)} passages", flush=True)
    norms = (np.linalg.norm(vectors[start : start + BLOCK], axis=1) for start in range(0, len(corpus), BLOCK))
    if min(block.min() for block in norms) < 0.99:
        raise RuntimeError(f"{partial} holds rows that were not encoded; delete it and rerun")
    del vectors
    partial.replace(target)
    progress.unlink()
