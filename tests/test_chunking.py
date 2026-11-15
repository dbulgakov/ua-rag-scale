import json
from pathlib import Path

from raglab.config import Config
from raglab.data.chunking import chunk_sentences, load_tokenizer, split_sentences
from raglab.data.ukwiki import article_id, passage_id

ARTICLE = json.loads((Path(__file__).parent / "fixtures/article.json").read_text())


def test_sentences_rebuild_the_text() -> None:
    text = "Перше речення. Друге речення!\n\nТретє без крапки"
    parts = split_sentences(text)
    assert "".join(parts) == text
    assert len(parts) == 3


def test_passage_ids_match_the_pinned_pool() -> None:
    config = Config()
    tokenizer = load_tokenizer(config.encoder, config.encoder_revision)
    chunks = chunk_sentences(
        split_sentences(ARTICLE["text"]), tokenizer, config.chunk_tokens, config.chunk_overlap
    )
    article = article_id(ARTICLE["id"])
    assert [passage_id(article, chunk.start, chunk.end) for chunk in chunks] == ARTICLE["passages"]
    assert all(chunk.tokens <= config.chunk_tokens for chunk in chunks)
    assert chunks[1].start < chunks[0].end
