import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from raglab.config import Config
from raglab.data.corpus import shuffled
from raglab.evaluate import cluster_interval

FEATURES = ["gap_1_2", "gap_1_10", "gap_1_50", "entropy", "articles_10", "articles_20", "log_size"]
THRESHOLDS = [round(value, 2) for value in np.arange(0, 1.0001, 0.05)]
Heads = dict[tuple[str, str], ClassifierMixin]


def families() -> dict[str, ClassifierMixin]:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=5, random_state=42
        ),
    }


def routing_table(config: Config, retrieval: pd.DataFrame) -> pd.DataFrame:
    cells = ["order", "seed", "size", "question_id"]
    selected = retrieval[retrieval["size"].isin(config.rerank_sizes)]
    quality = selected[selected["method"].isin(["hnsw", *config.router_actions])]
    quality = quality.pivot_table(index=cells, columns="method", values="recall_5")
    quality = quality.reindex(columns=["hnsw", *config.router_actions]).rename(columns={"hnsw": "fast"})
    probe = selected[selected["method"] == "hnsw"].set_index(cells)[[*FEATURES[:-1], "split", "cluster"]]
    table = probe.join(quality).reset_index()
    return table.assign(log_size=np.log10(table["size"]))


def partitions(frame: pd.DataFrame, fit_split: str) -> dict[str, pd.DataFrame]:
    tune = shuffled(frame.loc[frame["split"] == "dev", "cluster"].unique(), "router")
    if fit_split == "train":
        cut = len(tune) // 2
        parts = {"fit": set(frame.loc[frame["split"] == "train", "cluster"]), "calibration": tune[:cut]}
    else:
        first, cut = round(len(tune) * 0.6), round(len(tune) * 0.8)
        parts = {"fit": tune[:first], "calibration": tune[first:cut]}
    parts |= {"selection": tune[cut:], "test": set(frame.loc[frame["split"] == "test", "cluster"])}
    sets = {name: set(clusters) for name, clusters in parts.items()}
    if any(sets[a] & sets[b] for a in sets for b in sets if a < b):
        raise ValueError("Router partitions share article clusters")
    if min(len(sets["calibration"]), len(sets["selection"])) < 2:
        raise ValueError("Router calibration and selection need at least two clusters each")
    return {name: frame[frame["cluster"].isin(clusters)] for name, clusters in sets.items()}


def gain(frame: pd.DataFrame, action: str, config: Config) -> pd.Series:
    return (frame[action] - frame["fast"] >= config.router_min_gain).astype(int)


def fit_heads(config: Config, parts: dict[str, pd.DataFrame]) -> Heads:
    heads: Heads = {}
    for family in families():
        for action in config.router_actions:
            fit, calibration = (parts[name].dropna(subset=[action]) for name in ("fit", "calibration"))
            labels, held = gain(fit, action, config), gain(calibration, action, config)
            if labels.nunique() == 2 and held.nunique() == 2:
                model = FrozenEstimator(families()[family].fit(fit[FEATURES], labels))
                heads[family, action] = CalibratedClassifierCV(model, method="sigmoid").fit(
                    calibration[FEATURES], held
                )
    return heads


def action_scores(config: Config, frame: pd.DataFrame, heads: Heads, family: str) -> pd.DataFrame:
    scores = pd.DataFrame(-1.0, index=frame.index, columns=list(config.router_actions))
    for action in config.router_actions:
        if (family, action) in heads:
            known = frame[action].notna()
            scores.loc[known, action] = heads[family, action].predict_proba(frame.loc[known, FEATURES])[:, 1]
    return scores


def route(
    config: Config, frame: pd.DataFrame, scores: pd.DataFrame, threshold: float
) -> tuple[pd.Series, pd.Series]:
    choice = scores.idxmax(axis=1).where(scores.max(axis=1) >= threshold)
    quality = frame["fast"].copy()
    for action in config.router_actions:
        quality[choice == action] = frame.loc[choice == action, action]
    return choice, quality


def random_routing(config: Config, frame: pd.DataFrame, choice: pd.Series) -> pd.Series:
    cells = [frame[key] for key in ("order", "seed", "size")]
    expected = frame["fast"].copy()
    for action in config.router_actions:
        share = (choice == action).groupby(cells).transform("mean")
        expected += share * (frame[action].fillna(frame["fast"]) - frame["fast"])
    return expected


def select_threshold(config: Config, selection: pd.DataFrame, scores: pd.DataFrame) -> float | None:
    baseline = selection["fast"].mean()
    candidates = []
    for threshold in THRESHOLDS:
        choice, quality = route(config, selection, scores, threshold)
        share = choice.notna().mean()
        if share <= config.router_budget and quality.mean() >= baseline:
            candidates.append((quality.mean(), -share, threshold))
    return max(candidates)[2] if candidates else None


def summarize(config: Config, test: pd.DataFrame, scores: pd.DataFrame, threshold: float | None) -> dict:
    if threshold is None:
        choice, quality = pd.Series(np.nan, index=test.index), test["fast"]
    else:
        choice, quality = route(config, test, scores, threshold)
    clusters = test.groupby("question_id")["cluster"].first().to_numpy()

    def per_question(values: pd.Series) -> np.ndarray:
        return values.groupby(test["question_id"]).mean().to_numpy()

    def interval(values: np.ndarray) -> dict[str, float | None]:
        low, high = cluster_interval(values, clusters, config)
        return {"mean": float(values.mean()), "low": low, "high": high}

    policies = {
        "router": per_question(quality),
        "fast": per_question(test["fast"]),
        "random": per_question(random_routing(config, test, choice)),
        "oracle": per_question(test[["fast", *config.router_actions]].max(axis=1)),
    } | {action: per_question(test[action].fillna(test["fast"])) for action in config.router_actions}
    result = {name: interval(values) for name, values in policies.items()}
    result["router_minus_random"] = interval(policies["router"] - policies["random"])
    best = max(result[action]["mean"] for action in config.router_actions) - result["fast"]["mean"]
    return result | {
        "expensive_share": float(choice.notna().mean()),
        "shares": {action: float((choice == action).mean()) for action in config.router_actions},
        "gain_share": (result["router"]["mean"] - result["fast"]["mean"]) / best if best > 0 else None,
    }


def auc(labels: pd.Series, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if labels.nunique() == 2 else None


def prediction_quality(config: Config, parts: dict[str, pd.DataFrame], heads: Heads) -> list[dict]:
    rows = []
    test = parts["test"]
    for (family, action), head in heads.items():
        known = test.dropna(subset=[action])
        scores = head.predict_proba(known[FEATURES])[:, 1]
        rows.append(
            {"family": family, "head": f"{action}_gain", "auc": auc(gain(known, action, config), scores)}
        )
    for family, model in families().items():
        model.fit(parts["fit"][FEATURES], (parts["fit"]["fast"] == 0).astype(int))
        scores = model.predict_proba(test[FEATURES])[:, 1]
        rows.append({"family": family, "head": "fast_miss", "auc": auc(test["fast"] == 0, scores)})
    return rows


def trade_off(config: Config, test: pd.DataFrame, scores: pd.DataFrame, family: str) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        choice, quality = route(config, test, scores, threshold)
        random = random_routing(config, test, choice)
        rows.append(
            {
                "family": family,
                "expensive_share": float(choice.notna().mean()),
                "recall_5": float(quality.groupby(test["question_id"]).mean().mean()),
                "random": float(random.groupby(test["question_id"]).mean().mean()),
            }
        )
    return rows


def run(config: Config, language: str, retrieval: pd.DataFrame) -> dict[str, dict]:
    table = routing_table(config, retrieval)
    probe = (table["order"] == config.probe_cell[0]) & (table["seed"] == config.probe_cell[1])
    train = table[(table["split"] == "train") & probe]
    report = {}
    for order in sorted(table["order"].unique()):
        heldout = table[(table["order"] == order) & (table["split"] != "train")]
        parts = partitions(
            pd.concat([heldout.dropna(subset=list(config.router_actions)), train]),
            config.router_fit_split[language],
        )
        heads = fit_heads(config, parts)
        entry: dict = {
            "partitions": {
                name: {"questions": part["question_id"].nunique(), "clusters": part["cluster"].nunique()}
                for name, part in parts.items()
            },
            "prediction": prediction_quality(config, parts, heads),
            "routers": {},
            "curves": [],
        }
        for family in families():
            if any(key[0] == family for key in heads):
                test = parts["test"]
                scores = action_scores(config, test, heads, family)
                selection = action_scores(config, parts["selection"], heads, family)
                threshold = select_threshold(config, parts["selection"], selection)
                entry["routers"][family] = {
                    "overall": summarize(config, test, scores, threshold),
                    "by_size": {
                        int(size): summarize(config, part, scores.loc[part.index], threshold)
                        for size, part in test.groupby("size")
                    },
                }
                entry["curves"] += trade_off(config, test, scores, family)
        report[order] = entry
    return report
