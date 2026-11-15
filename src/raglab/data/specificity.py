import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import nltk
import pandas as pd
import pyarrow.parquet as pq
import pymorphy3

from raglab.config import Config
from raglab.data.corpus import Corpus

TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
APOSTROPHES = str.maketrans(dict.fromkeys("’ʼ‘`", "'"))
STOPWORDS = {
    "uk": (
        "а або аби але без би б був була були було бути в ви від він вона вони воно все всі де для до є "
        "за з і із й її їх його коли на над не ні о об от по при про та так те ти то той тут у це цей ця "
        "ці чи що щоб як я ми нас нам вас вам мене мені тебе тобі себе собі"
    ),
    "en": (
        "a an and are as at be been being but by can could did do does for from had has have he her hers "
        "him his how i if in into is it its may me might my no not of on or our ours she should so some "
        "than that the their theirs them then there these they this those to too us was we were what when "
        "where which who whom why will with would you your yours yes"
    ),
}


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).replace("́", "").casefold()
    return TOKEN.findall(text.translate(APOSTROPHES))


class Lemmatizer:
    def __init__(self, language: str) -> None:
        self.cache: dict[str, frozenset[str]] = {}
        if language == "uk":
            analyzer = pymorphy3.MorphAnalyzer(lang="uk")
            self.lemmas = lambda token: {parse.normal_form for parse in analyzer.parse(token)}
        else:
            nltk.download("wordnet", quiet=True)
            wordnet = nltk.corpus.wordnet
            self.lemmas = lambda token: {lemma for pos in "nvars" for lemma in wordnet._morphy(token, pos)}

    def __call__(self, token: str) -> frozenset[str]:
        if token not in self.cache:
            self.cache[token] = frozenset(self.lemmas(token) or {token})
        return self.cache[token]


def scan(arguments: tuple[str, str, int, dict[str, set[str]]]) -> set[tuple[str, str]]:
    path, language, group, targets = arguments
    lemmatize = Lemmatizer(language)
    table = pq.ParquetFile(path).read_row_group(group, columns=["article", "text"])
    found = set()
    for article, text in zip(table["article"].to_pylist(), table["text"].to_pylist(), strict=True):
        for token in set(tokens(text)):
            for lemma in lemmatize(token):
                found.update((article, key) for key in targets.get(lemma, ()))
    return found


def annotate(config: Config, language: str, workers: int = 8) -> dict[str, int]:
    corpus = Corpus(config, language)
    lemmatize = Lemmatizer(language)
    heldout = corpus.questions["split"].isin(["test", "dev"])
    keys = {
        words[0]
        for answers in corpus.questions.loc[heldout, "answers"]
        for a in answers
        if len(words := tokens(a)) == 1
    }
    targets: dict[str, set[str]] = defaultdict(set)
    for key in keys:
        for lemma in lemmatize(key):
            targets[lemma].add(key)
    path = str(corpus.dir / "passages.parquet")
    jobs = [(path, language, group, dict(targets)) for group in range(pq.ParquetFile(path).num_row_groups)]
    with ProcessPoolExecutor(workers) as pool:
        pairs = set().union(*pool.map(scan, jobs))
    frequency = pd.Series([key for _, key in pairs]).value_counts().to_dict()
    limit = config.specificity_threshold * len(corpus.article_names)
    stop_lemmas = frozenset().union(*(lemmatize(word) for word in STOPWORDS[language].split()))

    def low(answers: list[str]) -> bool:
        singles = [words[0] for answer in answers if len(words := tokens(answer)) == 1]
        return any(lemmatize(word) & stop_lemmas or frequency.get(word, 0) >= limit for word in singles)

    flags = [
        low(answers) if held else None
        for answers, held in zip(corpus.questions["answers"], heldout, strict=True)
    ]
    corpus.questions.assign(low_specificity=pd.array(flags, dtype="boolean")).to_parquet(
        corpus.dir / "questions.parquet", index=False
    )
    return {"questions": int(heldout.sum()), "low_specificity": int(sum(bool(flag) for flag in flags))}
