from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("uk", "en")
ARCHIVE = "https://archive.org/download/ukwiki-20220501/ukwiki-20220501-{table}.sql.gz"


@dataclass(frozen=True)
class Config:
    out: Path = ROOT
    device: str | None = None

    wiki_repo: str = "wikimedia/wikipedia"
    wiki_name: str = "20231101.uk"
    wiki_revision: str = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
    hotpot_repo: str = "hotpotqa/hotpot_qa"
    hotpot_revision: str = "1908d6afbbead072334abe2965f91bd2709910ab"
    abstracts_repo: str = "LovinYou/enwiki-20171001-pages-meta-current-withlinks-abstracts"
    abstracts_file: str = "enwiki-20171001-pages-meta-current-withlinks-abstracts.tsv"
    abstracts_revision: str = "dd8699459339f9a9c9fd4e4d47d181fb70392ce6"
    anchor_repo: str = "ufal/cus-qa"
    anchor_revision: str = "0b45809367c4597a49a7ef77cb74f8ef0f307f35"
    link_dumps: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "page": (
                ARCHIVE.format(table="page"),
                "378545350b220f02e4ae3ec0ce4d8bd641845b3dad588372d1208244c4d1fbc3",
            ),
            "redirect": (
                ARCHIVE.format(table="redirect"),
                "7820d459765351b52f7113be62d839678a8844d99330a1da2048290bf9d8ae34",
            ),
            "pagelinks": (
                ARCHIVE.format(table="pagelinks"),
                "ad56e89e9e1af8a13b209457a18bf3cb6600789ac1e2597f98d7890b650d81d5",
            ),
        }
    )
    pool_size: int = 1_000_000
    chunk_tokens: int = 384
    chunk_overlap: int = 32
    specificity_threshold: float = 0.01

    encoder: str = "intfloat/multilingual-e5-small"
    encoder_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    generator: str = "qwen/qwen3-235b-a22b-2507"
    generator_provider: str = "deepinfra/fp8"
    local_generator: str = "mlx-community/Qwen3-4B-4bit"

    sizes: tuple[int, ...] = (10_000, 30_000, 100_000, 300_000, 1_000_000)
    seeds: tuple[int, ...] = (42, 43, 44)
    orders: tuple[str, ...] = ("random", "hard")
    top_k: int = 50
    candidates: int = 100
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    adaptive_ef: tuple[int, ...] = (128, 256, 512, 1024)
    adaptive_target: float = 0.95
    ivf_lists: int = 256
    pq_segments: int = 48
    pq_bits: int = 8
    ivf_probes: int = 16
    ivf_training_vectors: int = 50_000
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    graph_seeds: int = 20
    graph_hops: int = 2
    graph_limit: int = 2000
    ppr_restart: float = 0.15

    rerank_sizes: tuple[int, ...] = (10_000, 100_000, 1_000_000)
    rerank_depth: dict[str, int] = field(default_factory=lambda: {"uk": 100, "en": 50})
    rerank_batch: int = 32
    rerank_max_length: int = 512
    probe_cell: tuple[str, int] = ("random", 42)
    latency_queries: int = 200

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_learning_rate: float = 1e-4
    lora_batch: int = 16
    lora_accumulation: int = 4
    lora_epochs: int = 2
    lora_temperature: float = 0.05
    lora_warmup: float = 0.1
    lora_seed: int = 42
    lora_english_questions: int = 20_000
    lora_negatives: int = 4
    lora_negative_depth: int = 100

    generation_questions: int = 200
    generation_methods: tuple[str, ...] = (
        "exact",
        "hnsw",
        "ivfpq",
        "hybrid",
        "rerank",
        "adaptive",
        "finetuned",
        "graph",
    )
    generation_passages: int = 5
    generation_max_tokens: int = 128
    generation_workers: int = 8

    router_actions: tuple[str, ...] = ("rerank", "graph")
    router_fit_split: dict[str, str] = field(default_factory=lambda: {"uk": "train", "en": "dev"})
    router_budget: float = 0.25
    router_min_gain: float = 0.01

    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 271828

    @property
    def work(self) -> Path:
        return self.out / "work"

    @property
    def results(self) -> Path:
        return self.out / "results"

    @property
    def reports(self) -> Path:
        return self.out / "reports"

    @property
    def data(self) -> Path:
        return ROOT / "data"

    def corpus_dir(self, language: str) -> Path:
        return self.work / language

    def cells(self) -> list[tuple[str, int, int]]:
        return [(order, seed, size) for order in self.orders for seed in self.seeds for size in self.sizes]

    def rerank_cells(self) -> list[tuple[str, int, int]]:
        return [
            (order, seed, size)
            for order, seed, size in self.cells()
            if size in self.rerank_sizes and (order == "random" or seed == self.probe_cell[1])
        ]

    def methods(self, language: str) -> tuple[str, ...]:
        base = ("exact", "hnsw", "ivfpq", "hybrid", "adaptive", "graph", "graph_no_ppr", "graph_exact")
        return (*base, "hybrid_lemma", "finetuned") if language == "uk" else (*base, "finetuned")
