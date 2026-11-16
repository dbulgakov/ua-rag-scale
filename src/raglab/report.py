import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from raglab import law, router
from raglab.config import LANGUAGES, Config
from raglab.data.corpus import SPLITS
from raglab.evaluate import cluster_interval, curves, paired_difference

TEMPLATES = Path(__file__).parent / "templates"

SIZE = {10_000: "10 тис.", 30_000: "30 тис.", 100_000: "100 тис.", 300_000: "300 тис.", 1_000_000: "1 млн"}
KEY = {10_000: "10k", 30_000: "30k", 100_000: "100k", 300_000: "300k", 1_000_000: "1m"}
LANGUAGE = {"uk": "UA", "en": "EN"}
METHOD = {
    "exact": "Точний пошук",
    "hnsw": "HNSW",
    "adaptive": "Адаптивний HNSW",
    "ivfpq": "IVF-PQ",
    "hybrid": "Гібрид BM25 + вектори",
    "hybrid_lemma": "Гібрид з лематизованим BM25",
    "finetuned": "LoRA",
    "graph": "Граф посилань (PPR)",
    "graph_no_ppr": "Граф без PPR",
    "graph_exact": "Граф, точні seed",
    "rerank": "Переранжування",
    "rerank_top50": "Переранжування, top-50",
    "rerank_scoring": "Оцінювання переранжувальником",
}
METRIC = {
    "recall_1": "Recall@1",
    "recall_5": "Recall@5",
    "recall_10": "Recall@10",
    "recall_20": "Recall@20",
    "mrr_10": "MRR@10",
    "all_support_10": "Усі докази в топ-10",
    "answer_in_top5": "Відповідь у топ-5",
}
ANSWER_METRIC = {"em": "EM", "f1": "F1", "em_lemma": "EM за лемами"}
PART = {"fit": "навчання", "calibration": "калібрування", "selection": "вибір порога", "test": "оцінка"}
POLICY = {
    "fast": "завжди HNSW",
    "router": "маршрутизатор",
    "random": "випадково, той самий бюджет",
    "rerank": "завжди переранжування",
    "graph": "завжди граф",
    "oracle": "ідеальний вибір",
}
JUDGES = {
    "qwen/qwen3-235b-a22b-2507": "Qwen",
    "google/gemini-2.5-flash": "Flash",
    "openai/gpt-4.1-mini": "mini",
}


@dataclass
class Results:
    config: Config
    retrieval: pd.DataFrame
    generation: pd.DataFrame
    anchor: pd.DataFrame
    latency: pd.DataFrame
    questions: pd.DataFrame
    corpus: dict
    finetune: dict
    cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def test(self, language: str, order: str | None = "random") -> pd.DataFrame:
        frame = self.retrieval[(self.retrieval["language"] == language) & (self.retrieval["split"] == "test")]
        return frame if order is None else frame[frame["order"] == order]

    def curve(self, language: str, order: str | None = "random") -> pd.DataFrame:
        if language not in self.cache:
            self.cache[language] = curves(self.test(language, None), self.config)
        table = self.cache[language]
        return table if order is None else table[table["order"] == order]

    def answers(self, language: str, local: bool = False) -> pd.DataFrame:
        model = self.config.local_generator if local else self.config.generator
        return self.generation[
            (self.generation["language"] == language) & (self.generation["model"] == model)
        ]


def load(config: Config) -> Results:
    folder = config.results
    questions = pd.read_parquet(folder / "questions.parquet").rename(columns={"id": "question_id"})
    keys = ["language", "question_id"]
    return Results(
        config,
        pd.read_parquet(folder / "retrieval.parquet").merge(questions, on=keys),
        pd.concat(pd.read_parquet(path) for path in sorted(folder.glob("generation*.parquet"))).merge(
            questions[[*keys, "cluster"]], on=keys
        ),
        pd.read_parquet(folder / "anchor.parquet"),
        pd.read_parquet(folder / "latency.parquet"),
        questions,
        json.loads((folder / "corpus.json").read_text()),
        json.loads((folder / "finetune.json").read_text()),
    )


def number(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:{'+' if signed else ''}.{digits}f}".replace(".", ",").replace("-", "−")


def interval(value: float, low: float | None, high: float | None, signed: bool = False) -> str:
    if low is None or pd.isna(low):
        return number(value, signed=signed)
    text = f"{number(value, signed=signed)} [{number(low, signed=signed)}; {number(high, signed=signed)}]"
    return text + (" ↑" if signed and low > 0 else " ↓" if signed and high < 0 else "")


def markdown(frame: pd.DataFrame) -> str:
    return frame.fillna("—").to_markdown(index=False, disable_numparse=True)


def intervals(frame: pd.DataFrame, value: str, signed: bool = False) -> list[str]:
    return [
        interval(v, low, high, signed)
        for v, low, high in zip(frame[value], frame["low"], frame["high"], strict=True)
    ]


def method_table(frame: pd.DataFrame, cells: list[str], methods: Iterable[str] = METHOD) -> str:
    table = frame.assign(cell=cells).pivot(index="method", columns="size", values="cell")
    table = table.reindex(list(methods)).dropna(how="all").rename(index=METHOD, columns=SIZE)
    return markdown(table.rename_axis("Метод").reset_index())


def curve_table(results: Results, language: str, order: str) -> str:
    curve = results.curve(language, order)
    return method_table(curve, intervals(curve, "recall_5"))


def metric_table(results: Results, language: str) -> str:
    test = results.test(language)
    table = test[test["method"] == "exact"].groupby("size")[list(METRIC)].mean().rename(columns=METRIC)
    for method in ("hnsw", "ivfpq"):
        approximate = test[test["method"] == method].groupby("size")["ann_recall_10"].mean()
        table[f"ANN Recall@10, {METHOD[method]}"] = approximate
    return markdown(table.map(number).rename(index=SIZE).rename_axis("Розмір").reset_index())


def differences(
    frame: pd.DataFrame, config: Config, methods: list[str], baseline: str, metric: str = "recall_5"
) -> pd.DataFrame:
    rows = [
        {"method": method, "size": int(size)} | paired_difference(part, config, method, baseline, metric)
        for size, part in frame.groupby("size")
        for method in methods
    ]
    return pd.DataFrame(rows).dropna(subset=["difference"])


def difference_table(frame: pd.DataFrame, config: Config, methods: list[str], baseline: str) -> str:
    table = differences(frame, config, methods, baseline)
    return method_table(table, intervals(table, "difference", signed=True), methods)


def mean_interval(frame: pd.DataFrame, config: Config, metric: str) -> str:
    low, high = cluster_interval(frame[metric].to_numpy(dtype=float), frame["cluster"].to_numpy(), config)
    return interval(frame[metric].mean(), low, high)


def generation_table(results: Results, language: str, local: bool = False) -> str:
    answers = results.answers(language, local)
    rows = []
    for (condition, method, size), part in answers.groupby(["condition", "method", "size"], dropna=False):
        label = {"no_rag": "без пошуку", "oracle": "еталонні уривки"}.get(condition) or METHOD[method]
        rows.append(
            {"Умова": label, "Розмір": SIZE.get(size, "—")}
            | {name: mean_interval(part, results.config, metric) for metric, name in ANSWER_METRIC.items()}
            | {"Некоректний формат": int((~part["valid"]).sum())}
        )
    return markdown(pd.DataFrame(rows)) if rows else "Не вимірювалося."


def generation_differences(results: Results, language: str) -> pd.DataFrame:
    rag = results.answers(language)
    rag = rag[rag["condition"] == "rag"]
    return pd.concat(
        differences(rag, results.config, ["rerank"], "hnsw", metric).assign(metric=metric)
        for metric in ANSWER_METRIC
    )


def generation_pairs(pairs: dict[str, pd.DataFrame]) -> str:
    frames = []
    for language, table in pairs.items():
        cells = table.assign(cell=intervals(table, "difference", signed=True))
        wide = cells.pivot(index="size", columns="metric", values="cell")[list(ANSWER_METRIC)]
        wide.columns = [f"Δ {name}" for name in ANSWER_METRIC.values()]
        wide.insert(0, "Розмір", wide.index.map(SIZE))
        wide.insert(0, "Мова", LANGUAGE[language])
        frames.append(wide)
    return markdown(pd.concat(frames))


def errors(results: Results, language: str) -> pd.DataFrame:
    rag = results.answers(language)
    rag = rag[(rag["condition"] == "rag") & (rag["method"] == "hnsw")]
    rows = []
    for size, part in rag.groupby("size"):
        wrong = part[part["em"] == 0]
        found = part[part["evidence"] == "all"]
        rows.append(
            {
                "size": int(size),
                "wrong": len(wrong),
                "search": int((wrong["evidence"] == "none").sum()),
                "partial": int((wrong["evidence"] == "partial").sum()),
                "reading": int((wrong["evidence"] == "all").sum()),
                "reading_rate": float((found["em"] == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def error_table(results: Results) -> str:
    frames = [errors(results, language).assign(language=LANGUAGE[language]) for language in LANGUAGES]
    table = pd.concat(frames).assign(
        size=lambda frame: frame["size"].map(SIZE), reading_rate=lambda f: f["reading_rate"].map(number)
    )
    columns = {
        "language": "Мова",
        "size": "Розмір",
        "wrong": "Неправильних",
        "search": "Помилка пошуку",
        "partial": "Часткові докази",
        "reading": "Помилка читання",
        "reading_rate": "Частка помилок за наявного еталона",
    }
    return markdown(table[list(columns)].rename(columns=columns))


def routers(results: Results) -> dict[str, dict]:
    retrieval = results.retrieval
    return {
        language: router.run(results.config, language, retrieval[retrieval["language"] == language])
        for language in LANGUAGES
    }


def router_curve(reports: dict[str, dict]) -> pd.DataFrame:
    curve = pd.DataFrame(reports["uk"]["random"]["curves"])
    return curve[curve["family"] == "logistic"].sort_values("expensive_share")


def router_latency(
    results: Results, language: str, size: int, shares: dict[str, float]
) -> tuple[float, float]:
    times = results.latency[(results.latency["language"] == language) & (results.latency["size"] == size)]
    times = times.set_index("method")["p50_ms"]
    routed = times["hnsw"] + shares["rerank"] * times["rerank_scoring"] + shares["graph"] * times["graph"]
    return float(routed), float(times["hnsw"] + times["rerank_scoring"])


def router_tables(results: Results, reports: dict[str, dict]) -> tuple[str, str, str, str]:
    policies, sizes, quality, parts = [], [], [], []
    for language, report in reports.items():
        for order, entry in report.items():
            where = {"Мова": LANGUAGE[language], "Порядок": order}
            parts.append(
                where
                | {
                    PART[name]: f"{part['questions']} / {part['clusters']}"
                    for name, part in entry["partitions"].items()
                }
            )
            for family, result in entry["routers"].items():
                overall = result["overall"]
                row = where | {
                    "Модель": family,
                    "Дорогих запитів": number(100 * overall["expensive_share"], 1) + "%",
                }
                row |= {POLICY[name]: number(overall[name]["mean"]) for name in POLICY}
                diff = overall["router_minus_random"]
                row["Δ до випадкового"] = interval(diff["mean"], diff["low"], diff["high"], signed=True)
                row["Частка виграшу"] = (
                    number(100 * overall["gain_share"], 0) + "%" if overall["gain_share"] else "—"
                )
                policies.append(row)
                for size, value in result["by_size"].items():
                    routed, always = router_latency(results, language, size, value["shares"])
                    diff = value["router_minus_random"]
                    sizes.append(
                        where
                        | {
                            "Модель": family,
                            "Розмір": SIZE[size],
                            "Маршрутизатор": number(value["router"]["mean"]),
                            "Δ до випадкового": interval(
                                diff["mean"], diff["low"], diff["high"], signed=True
                            ),
                            "Затримка, мс": number(routed, 0),
                            "Завжди переранжування, мс": number(always, 0),
                        }
                    )
            quality += [where | row for row in entry["prediction"]]
    quality = pd.DataFrame(quality).pivot(index=["Мова", "Порядок", "family"], columns="head", values="auc")
    quality = quality.map(number).reset_index().rename(columns={"family": "Модель"})
    tables = (pd.DataFrame(policies), pd.DataFrame(sizes), quality, pd.DataFrame(parts))
    return tuple(markdown(table) for table in tables)


def law_tables(results: Results) -> tuple[str, str, pd.DataFrame]:
    checks = pd.concat(
        [
            law.check(results.curve(language, None), results.config, language).assign(language=language)
            for language in LANGUAGES
        ]
    )
    rows = []
    for row in checks.itertuples():
        cells = {"Мова": LANGUAGE[row.language], "Порядок": row.order, "Метод": METHOD[row.method]}
        cells["Факт 1 млн"] = interval(row.observed, row.low, row.high)
        for form, label in (("log_linear", "Лог-лінійна"), ("power", "Степенева")):
            hit = "так" if getattr(row, f"{form}_hit") else "ні"
            cells[label] = (
                f"{number(getattr(row, form))}, {hit}, {number(getattr(row, f'{form}_error'), signed=True)}"
            )
        cells["Last-value"] = number(row.last_value_error, signed=True)
        cells["Втрата на 10×"] = number(row.loss_per_decade)
        rows.append(cells)
    errors_ = checks.assign(
        **{c: checks[c].abs() for c in ("log_linear_error", "power_error", "last_value_error")}
    )
    mae = errors_.groupby(["language", "order"])[
        ["log_linear_error", "power_error", "last_value_error"]
    ].mean()
    table = mae.map(number).reset_index()
    table["language"] = table["language"].map(LANGUAGE)
    table.columns = ["Мова", "Порядок", "MAE лог-лінійної", "MAE степеневої", "MAE last-value"]
    return markdown(pd.DataFrame(rows)), markdown(table), checks


def anchor_table(results: Results) -> str:
    rows = []
    for size, part in results.anchor.groupby("size"):
        row = {"Розмір": SIZE[int(size)]}
        for source, label in (("cus-qa", "CUS-QA"), ("benchmark", "Бенчмарк")):
            frame = part[part["source"] == source]
            row[f"{label} R@5"] = mean_interval(frame, results.config, "recall_5")
        rows.append(row)
    return markdown(pd.DataFrame(rows))


def latency_table(results: Results, language: str) -> str:
    table = results.latency[results.latency["language"] == language]
    return method_table(table, [number(value, 1) for value in table["p50_ms"]])


def judge_agreement(config: Config) -> dict[str, float]:
    with gzip.open(config.data / "judges/raw-responses.jsonl.gz", "rt") as stream:
        frame = pd.DataFrame(json.loads(line) for line in stream)
    frame = frame[frame["judge"].isin(JUDGES)]
    votes = frame.groupby(["question_id", "judge"])["accepted"].any().unstack("judge")
    result = {}
    for left, right in combinations(JUDGES, 2):
        pair = f"{JUDGES[left]}_{JUDGES[right]}".lower()
        result[f"agree_{pair}"] = float((votes[left] == votes[right]).mean())
        result[f"kappa_{pair}"] = float(cohen_kappa_score(votes[left], votes[right]))
    return result


def headline(
    results: Results, reports: dict[str, dict], checks: pd.DataFrame, pairs: dict[str, pd.DataFrame]
) -> dict[str, float]:
    config = results.config
    values: dict[str, float] = judge_agreement(config)
    for language in LANGUAGES:
        test = results.test(language)
        curve = results.curve(language).set_index(["method", "size"])["recall_5"]
        for (method, size), value in curve.items():
            values[f"{language}_{method}_{KEY[size]}"] = value
        values[f"{language}_exact_drop"] = curve["exact", 10_000] - curve["exact", 1_000_000]
        compared = [
            ("graph", "hnsw"),
            ("rerank", "hnsw"),
            ("finetuned", "hnsw"),
            ("hybrid", "exact"),
            ("hnsw", "exact"),
            ("hybrid_lemma", "hybrid"),
            ("hybrid_lemma", "exact"),
        ]
        for method, baseline in compared:
            for row in differences(test, config, [method], baseline).itertuples():
                key = f"d_{language}_{method}_{baseline}_{KEY[row.size]}"
                values |= {key: row.difference, f"{key}_low": row.low, f"{key}_high": row.high}
        rag = results.answers(language)
        for (condition, method, size), part in rag.groupby(["condition", "method", "size"], dropna=False):
            name = condition if condition != "rag" else f"{method}_{KEY[int(size)]}"
            values |= {
                f"{language}_em_{name}": part["em"].mean(),
                f"{language}_lemma_{name}": part["em_lemma"].mean(),
            }
        for row in pairs[language].itertuples():
            key = f"d_{language}_{row.metric}_{KEY[row.size]}"
            values |= {key: row.difference, f"{key}_low": row.low, f"{key}_high": row.high}
        for row in errors(results, language).itertuples():
            key = KEY[row.size]
            values |= {f"{language}_read_{key}": row.reading_rate, f"{language}_wrong_{key}_n": row.wrong}
            values[f"{language}_search_{key}_n"] = row.search
        logistic = reports[language]["random"]["routers"]["logistic"]
        overall = logistic["overall"]
        for name in POLICY:
            values[f"{language}_route_{name}"] = overall[name]["mean"]
        diff = overall["router_minus_random"]
        values |= {
            f"d_{language}_route": diff["mean"],
            f"d_{language}_route_low": diff["low"],
            f"d_{language}_route_high": diff["high"],
        }
        values |= {
            f"{language}_route_share_pct": 100 * overall["expensive_share"],
            f"{language}_route_gain_pct": 100 * (overall["gain_share"] or 0),
        }
        times = [
            router_latency(results, language, size, value["shares"])
            for size, value in logistic["by_size"].items()
        ]
        values[f"{language}_route_time_low_pct"] = 100 * min(routed / always for routed, always in times)
        values[f"{language}_route_time_high_pct"] = 100 * max(routed / always for routed, always in times)
        for row in reports[language]["random"]["prediction"]:
            if row["family"] == "logistic":
                values[f"{language}_auc_{row['head']}"] = row["auc"]
        history = results.finetune["history"]
        best = max(history, key=lambda epoch: epoch["tune_recall_10"]["mean"])["tune_recall_10"][language]
        values |= {
            f"{language}_lora_before": results.finetune["baseline_tune_recall_10"][language],
            f"{language}_lora_after": best,
        }
        values[f"{language}_negatives_pct1"] = (
            100 * results.finetune["hard_negatives"][language]["negatives_with_answer"]
        )
        stats = results.corpus[language]
        values |= {
            f"{language}_low_n": stats["specificity"]["low_specificity"],
            f"{language}_heldout_n": stats["specificity"]["questions"],
        }
        values[f"{language}_tokens_mln"] = stats["tokens"][str(config.sizes[-1])] / 1e6
        times = results.latency[results.latency["language"] == language]
        for row in times.itertuples():
            values[f"{language}_ms_{row.method}_{KEY[row.size]}"] = row.p50_ms
        part = checks[checks["language"] == language]
        for order, group in part.groupby("order"):
            for column in ("log_linear", "power", "last_value"):
                values[f"{language}_{order}_mae_{column}"] = group[f"{column}_error"].abs().mean()
        counts = results.questions[results.questions["language"] == language]["split"].value_counts()
        values |= {f"{language}_{split}_n": int(counts[split]) for split in SPLITS}
        values[f"{language}_clusters_n"] = test["cluster"].nunique()
    local = results.answers("uk", local=True)
    for (condition, _, size), part in local.groupby(["condition", "method", "size"], dropna=False):
        values[f"local_em_{condition if condition != 'rag' else KEY[int(size)]}"] = part["em"].mean()
    values |= {
        "law_log_hits_n": int(checks["log_linear_hit"].sum()),
        "law_power_hits_n": int(checks["power_hit"].sum()),
        "law_rows_n": len(checks),
        "law_refit_gap": checks[["log_linear_refit_gap", "power_refit_gap"]].to_numpy().max(),
    }
    anchor = results.anchor.groupby(["source", "size"])["recall_5"].mean()
    for (source, size), value in anchor.items():
        values[f"{'anchor' if source == 'cus-qa' else 'articles'}_{KEY[size]}"] = value
    uk = results.corpus["uk"]
    return values | {
        "wiki_articles_n": uk["wikipedia_articles"],
        "wiki_chunks_n": uk["wikipedia_chunks"],
        "pool_articles_n": uk["pool_articles"],
        "graph_articles_mln": uk["graph"]["articles"] / 1e6,
        "graph_edges_mln": uk["graph"]["edges"] / 1e6,
        "unresolved_pct1": 100 * uk["graph"]["unresolved_link_share"],
        "no_node_pct": 100 * uk["graph"]["pool_articles_without_node"],
        "gold_articles_n": uk["graph"]["gold_articles"],
        "gold_with_node_n": uk["graph"]["gold_articles_with_node"],
    }


def formatted(values: dict[str, float]) -> dict[str, str]:
    result = {}
    for key, value in values.items():
        if key.endswith("_n"):
            result[key] = f"{int(value):,}".replace(",", " ")
        elif key.endswith(("_pct", "_ms")) or "_ms_" in key:
            result[key] = number(value, 0 if key.endswith("_pct") or value >= 100 else 1)
        elif key.endswith(("_pct1", "_mln")):
            result[key] = number(value, 1 if value < 100 else 0)
        elif key.endswith("_gap"):
            result[key] = number(value, 4)
        else:
            result[key] = number(value, signed=key.startswith("d_"))
    return result


def figures(results: Results, reports: dict[str, dict]) -> None:
    folder = results.config.reports / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "raglab"
    shown = ["exact", "hnsw", "hybrid", "graph", "rerank"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    tokens, axis_tokens = plt.subplots(figsize=(7, 4))
    for axis, language in zip(axes, LANGUAGES, strict=True):
        curve = results.curve(language)
        for method in shown:
            part = curve[curve["method"] == method].sort_values("size")
            axis.plot(part["size"], part["recall_5"], marker="o", label=METHOD[method])
        axis.set(xscale="log", title=LANGUAGE[language], xlabel="уривків у корпусі")
        axis.grid(alpha=0.3)
        counts = results.corpus[language]["tokens"]
        for method, style in (("exact", "-"), ("hnsw", "--")):
            part = curve[curve["method"] == method].sort_values("size")
            axis_tokens.plot(
                [counts[str(size)] for size in part["size"]],
                part["recall_5"],
                style,
                marker="o",
                label=f"{LANGUAGE[language]}, {METHOD[method]}",
            )
    axes[0].set_ylabel("Recall@5, test")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(folder / "degradation.svg", metadata={"Date": None})
    axis_tokens.set(xscale="log", xlabel="токенів e5 у корпусі", ylabel="Recall@5, test")
    axis_tokens.grid(alpha=0.3)
    axis_tokens.legend()
    tokens.tight_layout()
    tokens.savefig(folder / "degradation_tokens.svg", metadata={"Date": None})
    curve = router_curve(reports)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(curve["expensive_share"], curve["recall_5"], marker="o", label="маршрутизатор")
    axis.plot(curve["expensive_share"], curve["random"], "--", label="випадково, той самий бюджет")
    axis.set(xlabel="частка запитів на дорогий метод", ylabel="Recall@5, UA test")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "router.svg", metadata={"Date": None})
    plt.close("all")


def demo_data(results: Results, reports: dict[str, dict], values: dict[str, str]) -> dict:
    def series(language: str, methods: list[str]) -> list[dict]:
        curve = results.curve(language)
        return [
            {
                "method": method,
                "label": METHOD[method],
                "points": [
                    {"size": int(row.size), "recall": row.recall_5}
                    for row in curve[curve["method"] == method].sort_values("size").itertuples()
                ],
            }
            for method in methods
        ]

    template = json.loads((TEMPLATES / "recommendations_uk.json").read_text())
    return {
        "curves": series("uk", ["exact", "hnsw", "hybrid", "graph", "rerank", "hybrid_lemma"]),
        "english": series("en", ["exact"]),
        "tokens": {language: results.corpus[language]["tokens"] for language in LANGUAGES},
        "router": [
            {"share": row.expensive_share, "recall": row.recall_5, "random": row.random}
            for row in router_curve(reports).itertuples()
        ],
        "recommendations": [{key: text.format_map(values) for key, text in row.items()} for row in template],
    }


def write_report(config: Config) -> None:
    results = load(config)
    reports = routers(results)
    law_table, mae_table, checks = law_tables(results)
    pairs = {language: generation_differences(results, language) for language in LANGUAGES}
    numbers = headline(results, reports, checks, pairs)
    policies, sizes, quality, parts = router_tables(results, reports)
    values = formatted(numbers) | {
        "law": law_table,
        "law_mae": mae_table,
        "router_policies": policies,
        "router_sizes": sizes,
        "router_auc": quality,
        "router_parts": parts,
        "generation_pairs": generation_pairs(pairs),
        "errors": error_table(results),
        "anchor": anchor_table(results),
        "generation_local": generation_table(results, "uk", local=True),
    }
    graph = ["graph", "graph_exact", "graph_no_ppr"]
    for language in LANGUAGES:
        test = results.test(language, None)
        random = test[test["order"] == "random"]
        methods = [
            method
            for method in METHOD
            if method not in ("hnsw", "rerank_top50") and method in set(random["method"])
        ]
        values |= {
            f"curves_{language}": curve_table(results, language, "random"),
            f"curves_hard_{language}": curve_table(results, language, "hard"),
            f"metrics_{language}": metric_table(results, language),
            f"methods_{language}": difference_table(random, config, methods, "hnsw"),
            f"graph_hard_{language}": difference_table(test[test["order"] == "hard"], config, graph, "hnsw"),
            f"generation_{language}": generation_table(results, language),
            f"latency_{language}": latency_table(results, language),
        }
    english = results.test("en")
    for kind in ("bridge", "comparison"):
        values[f"graph_{kind}"] = difference_table(english[english["kind"] == kind], config, graph, "hnsw")
    demo = demo_data(results, reports, values)
    values["recommendations"] = markdown(
        pd.DataFrame(demo["recommendations"]).rename(
            columns={"size": "Розмір бази", "use": "Що використати", "avoid": "Чого уникати"}
        )
    )
    template = (TEMPLATES / "report_uk.md").read_text()
    config.reports.mkdir(parents=True, exist_ok=True)
    (config.reports / "report_uk.md").write_text(template.format_map(values))
    (config.results / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=1))
    (config.results / "headline.json").write_text(json.dumps(numbers, indent=1, default=float))
    figures(results, reports)
