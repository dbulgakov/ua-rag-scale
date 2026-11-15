import heapq
import html
import json
import unicodedata
import urllib.parse
from html.parser import HTMLParser

import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

from raglab.config import Config
from raglab.data.chunking import Chunk, chunk_sentences, load_tokenizer
from raglab.data.corpus import sha256, write_corpus


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalize_title(value: str) -> str:
    value = urllib.parse.unquote(value).split("#", 1)[0].replace("_", " ")
    return normalize_text(html.unescape(value))


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href") or ""
        if tag.casefold() == "a" and href and not href.startswith(("http:", "https:", "//", "mailto:")):
            self.links.add(normalize_title(href))

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def parse_links(sentences: list[str]) -> tuple[str, set[str]]:
    parser = LinkParser()
    parser.feed("".join(sentences))
    parser.close()
    return "".join(parser.parts), parser.links - {""}


def priority(article: str, start: int) -> str:
    return sha256("pool-v2", article)[:48] + f"{start:016x}"


def load_questions(config: Config) -> pd.DataFrame:
    splits = pd.read_parquet(config.data / "pool/en-question-splits.parquet").set_index("id")["split"]
    source = concatenate_datasets(
        [
            load_dataset(config.hotpot_repo, "fullwiki", revision=config.hotpot_revision, split="train"),
            load_dataset(
                config.hotpot_repo, "distractor", revision=config.hotpot_revision, split="validation"
            ),
        ]
    )
    rows = []
    for row in source:
        identity = f"hotpot:{row['id']}"
        if identity not in splits.index:
            continue
        context = dict(zip(row["context"]["title"], row["context"]["sentences"], strict=True))
        facts = zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True)
        support = [
            (title, context[title][position] if position < len(context.get(title, ())) else None)
            for title, position in facts
        ]
        rows.append(
            {
                "id": identity,
                "question": row["question"],
                "answers": [row["answer"]],
                "support": support,
                "split": splits[identity],
                "component": None,
                "kind": row["type"],
            }
        )
    if len(rows) != len(splits):
        raise ValueError("Pinned HotpotQA question ids are missing from the source dataset")
    return pd.DataFrame(rows)


def read_abstracts(
    config: Config, needed: set[str]
) -> tuple[dict[str, dict], dict[str, tuple], dict[str, set[str]]]:
    path = hf_hub_download(
        config.abstracts_repo, config.abstracts_file, repo_type="dataset", revision=config.abstracts_revision
    )
    tokenizer = load_tokenizer(config.encoder, config.encoder_revision)
    chunks: dict[str, dict] = {}
    evidence: dict[str, tuple[str, list[str], list[tuple[str, Chunk]]]] = {}
    links: dict[str, set[str]] = {}
    heap: list[tuple[int, str]] = []
    with open(path, "rb") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                article = f"en:{record['id']}"
                key = normalize_title(record["title"])
                is_evidence = key in needed
                if (
                    not is_evidence
                    and len(heap) >= config.pool_size
                    and -int(priority(article, 0), 16) < heap[0][0]
                ):
                    continue
                sentences = record["text"]
                if not isinstance(sentences, list) or not all(isinstance(item, str) for item in sentences):
                    continue
                clean, outgoing = parse_links(record["text_with_links"])
                text = "".join(sentences)
                if normalize_text(clean) != normalize_text(text):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            spans = chunk_sentences(sentences, tokenizer, config.chunk_tokens, config.chunk_overlap)
            added = False
            for span in spans:
                identity = f"{article}:{span.start}:{span.end}"
                if identity in chunks:
                    continue
                rank = priority(article, span.start)
                if not is_evidence:
                    if len(heap) < config.pool_size:
                        heapq.heappush(heap, (-int(rank, 16), identity))
                    elif -int(rank, 16) > heap[0][0]:
                        del chunks[heapq.heapreplace(heap, (-int(rank, 16), identity))[1]]
                    else:
                        continue
                chunks[identity] = {
                    "id": identity,
                    "article": article,
                    "title": record["title"],
                    "url": record["url"],
                    "text": text[span.start : span.end],
                    "tokens": span.tokens,
                    "priority": rank,
                    "title_key": key,
                }
                added = True
            if added:
                links.setdefault(article, set()).update(outgoing)
            if is_evidence and key not in evidence:
                evidence[key] = (article, sentences, [(f"{article}:{s.start}:{s.end}", s) for s in spans])
    return chunks, evidence, links


def align(support: list[tuple[str, str | None]], evidence: dict[str, tuple]) -> list[list[str]]:
    groups = []
    for title, sentence in support:
        _, sentences, spans = evidence[normalize_title(title)]
        wanted = {
            i for i, item in enumerate(sentences) if normalize_text(item) == normalize_text(sentence or "")
        }
        groups.append(sorted({identity for identity, span in spans if wanted & set(span.sentences)}))
        if not groups[-1]:
            raise ValueError(f"Supporting sentence of {title!r} is not aligned to a passage")
    return groups


def article_links(passages: pd.DataFrame, links: dict[str, set[str]]) -> np.ndarray:
    titles = passages.drop_duplicates("article").groupby("title_key")["article"]
    target_of = titles.first()[titles.size() == 1].to_dict()
    names = np.unique(passages["article"])
    index = {name: position for position, name in enumerate(names)}
    edges = {
        (index[source], index[target])
        for source in names
        for title in links.get(source, ())
        if (target := target_of.get(title)) and target != source
    }
    return np.array(sorted(edges), dtype=np.int32)


def build(config: Config) -> dict[str, object]:
    questions = load_questions(config)
    needed = {normalize_title(title) for support in questions["support"] for title, _ in support}
    chunks, evidence, links = read_abstracts(config, needed)
    questions["gold"] = [align(support, evidence) for support in questions["support"]]
    gold = {passage for groups in questions["gold"] for group in groups for passage in group}
    rest = sorted((chunk["priority"], identity) for identity, chunk in chunks.items() if identity not in gold)
    selected = gold | {identity for _, identity in rest[: config.pool_size - len(gold)]}
    passages = pd.DataFrame([chunks[identity] for identity in sorted(selected)])
    edges = article_links(passages, links)
    write_corpus(
        config,
        "en",
        passages.drop(columns=["priority", "title_key"]),
        questions.drop(columns="support"),
        edges,
    )
    return {
        "pool_passages": len(passages),
        "pool_articles": int(passages["article"].nunique()),
        "graph": {"pool_edges": len(edges)},
    }
