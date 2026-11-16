import pytest
import typer

from raglab.cli import configure
from raglab.config import ROOT, Config


def test_sizes_restrict_the_grid_and_reranking() -> None:
    config = configure(ROOT, sizes=[100_000, 10_000])
    assert config.sizes == (10_000, 100_000)
    assert config.rerank_sizes == (10_000, 100_000)
    assert {size for _, _, size in config.cells()} == {10_000, 100_000}
    assert configure(ROOT, sizes=[30_000]).rerank_cells() == []


def test_default_sizes_are_kept_and_unknown_sizes_are_rejected() -> None:
    assert configure(ROOT) == Config()
    with pytest.raises(typer.BadParameter):
        configure(ROOT, sizes=[20_000])
