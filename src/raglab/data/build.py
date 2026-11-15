import json

from raglab.config import Config
from raglab.data import hotpot, specificity, ukwiki
from raglab.data.corpus import Corpus, atomic


def build(config: Config) -> None:
    summary = {}
    for language, builder in (("uk", ukwiki), ("en", hotpot)):
        path = config.corpus_dir(language) / "summary.json"
        if not path.exists():
            stats = builder.build(config)
            stats["specificity"] = specificity.annotate(config, language)
            corpus = Corpus(config, language)
            stats["tokens"] = {
                size: int(corpus.tokens[corpus.subset(*config.probe_cell, size)].sum())
                for size in config.sizes
            }
            with atomic(path) as target:
                target.write_text(json.dumps(stats, indent=2))
        summary[language] = json.loads(path.read_text())
    config.results.mkdir(parents=True, exist_ok=True)
    (config.results / "corpus.json").write_text(json.dumps(summary, indent=2))
