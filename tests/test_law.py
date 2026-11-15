import hashlib

import numpy as np

from raglab.law import PREREGISTRATION, fit, predict, registered


def test_fit_recovers_both_forms() -> None:
    sizes = np.array([10_000, 30_000, 100_000, 300_000])
    forms = fit(sizes, 0.9 - 0.1 * np.log10(sizes))
    assert np.allclose(forms["log_linear"], (0.9, 0.1))
    assert np.isclose(predict(forms, 1_000_000)["log_linear"], 0.3)
    forms = fit(sizes, 1 - 0.05 * sizes**0.2)
    assert np.allclose(forms["power"], (0.05, 0.2))


def test_preregistered_forecasts_are_unchanged_and_readable() -> None:
    digest = hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    assert digest == "e9ee653fb270dd3ef762eace612b0162225f8508dbe7c71cbc63119c8c1d650e"
    forecasts = registered()
    assert len(forecasts) == 36
    row = forecasts.set_index(["language", "order", "method"]).loc[("uk", "random", "exact")]
    assert (row["log_linear_low"], row["log_linear"], row["log_linear_high"]) == (0.4004, 0.5015, 0.5736)
