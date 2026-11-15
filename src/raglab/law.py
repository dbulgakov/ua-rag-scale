import re

import numpy as np
import pandas as pd

from raglab.config import LANGUAGES, ROOT, Config

PREREGISTRATION = ROOT / "reports/degradation_preregistration.md"
FORMS = ("log_linear", "power")
FORECAST = re.compile(r"(-?[\d.]+) \[(-?[\d.]+); (-?[\d.]+)\]")


def fit(sizes: np.ndarray, recalls: np.ndarray) -> dict[str, tuple[float, float]]:
    slope, intercept = np.polyfit(np.log10(sizes), recalls, 1)
    exponent, scale = np.polyfit(np.log(sizes), np.log(np.clip(1 - recalls, 1e-6, None)), 1)
    return {"log_linear": (float(intercept), float(-slope)), "power": (float(np.exp(scale)), float(exponent))}


def predict(forms: dict[str, tuple[float, float]], size: int) -> dict[str, float]:
    (a, b), (c, d) = forms["log_linear"], forms["power"]
    return {"log_linear": a - b * np.log10(size), "power": 1 - c * size**d}


def registered() -> pd.DataFrame:
    rows = []
    for line in PREREGISTRATION.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 8 and cells[0] in LANGUAGES:
            row = {"language": cells[0], "order": cells[1], "method": cells[2]}
            for form, cell in zip(FORMS, cells[6:], strict=True):
                point, low, high = map(float, FORECAST.fullmatch(cell).groups())
                row |= {form: point, f"{form}_low": low, f"{form}_high": high}
            rows.append(row)
    return pd.DataFrame(rows)


def check(curve: pd.DataFrame, config: Config, language: str) -> pd.DataFrame:
    sizes, target = list(config.sizes[:-1]), config.sizes[-1]
    forecast = registered()
    forecast = forecast[forecast["language"] == language].set_index(["order", "method"])
    rows = []
    for (order, method), frame in curve.groupby(["order", "method"]):
        if (order, method) not in forecast.index:
            continue
        recall = frame.set_index("size")["recall_5"]
        observed = frame.set_index("size").loc[target]
        forms = fit(np.array(sizes), recall[sizes].to_numpy())
        refit = predict(forms, target)
        row = {
            "order": order,
            "method": method,
            "observed": observed["recall_5"],
            "low": observed["low"],
            "high": observed["high"],
            "last_value_error": recall[sizes[-1]] - recall[target],
            "loss_per_decade": forms["log_linear"][1],
        }
        for form in FORMS:
            point, low, high = forecast.loc[(order, method), [form, f"{form}_low", f"{form}_high"]]
            row |= {
                form: point,
                f"{form}_hit": low <= recall[target] <= high,
                f"{form}_error": point - recall[target],
                f"{form}_refit_gap": abs(refit[form] - point),
            }
        rows.append(row)
    return pd.DataFrame(rows)
