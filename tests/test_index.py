import networkx as nx
import numpy as np
from scipy import sparse

from raglab.index import personalized_pagerank, rrf


def test_personalized_pagerank_matches_networkx() -> None:
    rng = np.random.default_rng(7)
    dense = (rng.random((40, 40)) < 0.12).astype(float)
    np.fill_diagonal(dense, 0)
    dense[5] = 0
    personalization = np.zeros(40)
    personalization[[1, 9, 20]] = [0.5, 0.3, 0.2]
    ours = personalized_pagerank(sparse.csr_matrix(dense), personalization, restart=0.15)
    graph = nx.from_numpy_array(dense, create_using=nx.DiGraph)
    weights = dict(enumerate(personalization))
    theirs = nx.pagerank(
        graph, alpha=0.85, personalization=weights, dangling=weights, tol=1e-12, max_iter=500
    )
    assert np.allclose(ours, [theirs[node] for node in range(40)], atol=1e-8)


def test_rrf_rewards_agreement() -> None:
    items, scores = rrf([[1, 2, 3], [3, 1, 4]], k=60, limit=3)
    assert items == [1, 3, 2]
    assert scores[0] == 1 / 61 + 1 / 62
