import re
from dataclasses import dataclass
from functools import cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

SENTENCE = re.compile(r".+?(?:[.!?](?=\s|$)\s*|\n+|$)", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    tokens: int
    sentences: tuple[int, ...]


@cache
def load_tokenizer(name: str, revision: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    tokenizer.model_max_length = 10**9
    return tokenizer


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE.findall(text)
    return parts if parts and "".join(parts) == text else [text]


def chunk_sentences(
    sentences: list[str], tokenizer: PreTrainedTokenizerBase, max_tokens: int, overlap: int
) -> list[Chunk]:
    text = "".join(sentences)
    if not text.strip():
        return []
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        return []
    spans, cursor = [], 0
    for sentence in sentences:
        spans.append((cursor + len(sentence) - len(sentence.lstrip()), cursor + len(sentence.rstrip())))
        cursor += len(sentence)
    boundaries = {end for _, end in spans}
    chunks, start = [], 0
    while True:
        stop = min(start + max_tokens, len(offsets))
        if stop < len(offsets):
            breaks = [i for i in range(start + 1, stop + 1) if offsets[i - 1][1] in boundaries]
            if breaks and breaks[-1] - start > overlap:
                stop = breaks[-1]
        left, right = offsets[start][0], offsets[stop - 1][1]
        inside = tuple(i for i, (a, b) in enumerate(spans) if a >= left and b <= right)
        chunks.append(Chunk(left, right, stop - start, inside))
        if stop == len(offsets):
            return chunks
        start = max(start + 1, stop - overlap)
