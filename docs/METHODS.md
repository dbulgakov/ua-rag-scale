# Methods

This document describes the final protocol behind the numbers in `reports/report_uk.md`. The parameters live in `src/raglab/config.py`.

## Question

How does retrieval quality in a RAG system change as the corpus grows from 10 thousand to 1 million passages, and which methods compensate for the loss? The main experiment uses Ukrainian Wikipedia; English HotpotQA is a control.

## Data

### Ukrainian corpus

- **Source.** `wikimedia/wikipedia`, configuration `20231101.uk`, pinned revision. It is the only searchable source.
- **Passages.** Each article is split at sentence boundaries into passages of at most 384 tokens of the encoder tokenizer with a 32-token overlap. A passage id `native-uk:<page>:<start>:<end>` is a character range of the article text, so ids are stable and verifiable.
- **Pool.** The corpus is a fixed pool of 1,000,000 passages: all benchmark articles first, then whole articles in the order of a salted SHA-256 hash of the article id.
- **Check.** `raglab data` rebuilds the pool from the raw dump and fails unless the passage ids equal the pinned list in `data/pool/uk.parquet`.

### Ukrainian benchmark

- **Questions and answers** come from UA-SQuAD, a Ukrainian translation of SQuAD. Translated contexts are never indexed and never used as evidence.
- **Evidence.** Each question was aligned automatically to passages of native Ukrainian Wikipedia that contain the answer. Every candidate question–passage pair was then judged by three LLMs from different families (Qwen3-235B, Gemini 2.5 Flash, GPT-4.1-mini).
- **Vote.** A pair is accepted when at least two judges accept it; a question is kept when it has an accepted pair. The raw judge responses are in `data/judges/`.
- **Splits.** Articles are assigned to train, tune and test once; no article is shared between splits. The result is 739 test, 148 tune and 735 train questions (`data/benchmark-v2/questions-primary.jsonl`). The single-judge and pre-filter sets in the same folder are robustness sets.

### Link graph

- **Source.** Wikipedia SQL dumps `ukwiki-20220501` (`page`, `redirect`, `pagelinks`) from the Internet Archive; URLs and SHA-256 hashes are pinned. No later dump with page links was available, so the graph is 549 days older than the text.
- **Construction.** Namespace-0 pages are nodes. Redirects are resolved to their target article, links to missing pages are dropped, self-loops are removed and duplicate pairs are merged.
- **English.** The graph comes from the hyperlinks inside the same 2017 abstracts that form the corpus.

### English control

- **Corpus.** Introductory paragraphs of the 2017 English Wikipedia dump used by HotpotQA, chunked by the same rule. The pool holds the evidence passages of the selected questions and then passages in hash order, 1,000,000 in total.
- **Questions.** 1,000 test and 500 tune questions from the HotpotQA validation set and 20,000 training questions from its training set. Splits share no supporting article. The question ids and splits are pinned in `data/pool/en-question-splits.parquet`.
- **Evidence.** Each supporting fact is one evidence group: the passages that contain that sentence.

### Growth orders

The corpus sizes are nested: every smaller corpus is a subset of the larger one, and evidence passages are always present.

- **Random growth.** Passages are added in the order of a per-seed hash.
- **Hard growth.** Passages most likely to be confused with the evidence come first: the articles of the evidence, their link neighbours, and the 100 nearest passages by vector for every evidence passage. The membership of this tier is pinned in `data/pool/*.parquet` because nearest neighbours vary slightly between devices.
- **Seeds.** 42, 43 and 44. At 1 million passages both orders give the same corpus.

### Low-specificity answers

An answer is low-specificity if it is a single token and either its lemma occurs in at least 1% of the pool articles or it is a stop word (pymorphy3 for Ukrainian, WordNet for English). Such questions are excluded from the "answer string in a retrieved passage" metric only.

## Retrieval methods

All dense methods use `intfloat/multilingual-e5-small` (384 dimensions) with normalized vectors.

| Method | Definition |
|---|---|
| Exact | FAISS flat inner product |
| HNSW | M = 32, efConstruction = 200, efSearch = 128 |
| Adaptive HNSW | The smallest ef in {128, 256, 512, 1024} that reaches ANN Recall@10 ≥ 0.95 on tune |
| IVF-PQ | 256 lists, 48 sub-vectors of 8 bits, 16 probes |
| Hybrid | HNSW top-100 and BM25 top-100 (k1 = 1.2, b = 0.75) fused by RRF (k = 60) |
| Hybrid with lemmas | The same with BM25 over pymorphy3 lemmas (Ukrainian only) |
| Rerank | HNSW candidates rescored by `BAAI/bge-reranker-v2-m3`: top-100 for Ukrainian, top-50 for English |
| LoRA | HNSW over vectors of the encoder with a LoRA adapter |
| Link graph | Articles of the HNSW top-20 are seeds; up to two hops along outgoing links, at most 2,000 candidates; personalized PageRank (restart 0.15) fused with the dense order by RRF |

- **Graph ablations.** Expansion without PageRank, and expansion from exact seeds.
- **Rerank depth.** Top-50 is used only where it differs from top-100 by less than 0.01 Recall@5 on tune; this held for English and not for Ukrainian.
- **Rerank grid.** Reranking is measured at 10 thousand, 100 thousand and 1 million passages; for hard growth only with seed 42.
- **LoRA.** Rank 8, alpha 16, dropout 0.05, learning rate 1e-4, batches of 16 with gradient accumulation over 4, two epochs, temperature 0.05. Training uses 20,000 English and all Ukrainian training questions, four hard negatives per question from the top-100 of the base encoder, and a multi-positive loss that masks other passages of the evidence articles. The epoch with the best tune Recall@10 is kept.
- **Hybrid with lemmas** was added after the forecasts were registered, so it is not part of the degradation-law check.

## Metrics and statistics

- **Retrieval.** Recall@k is the share of the question's evidence groups that have a passage in the top k. A Ukrainian question has one group. An English question has one group per supporting fact, that is per supporting sentence of HotpotQA; the count for every question is the column `n_groups` of `results/questions.parquet`, 2 to 7 in the test set. A paragraph with several supporting sentences therefore weighs more. The two languages are compared on Recall@k, the chance of finding one given piece of evidence. Two bounds are reported next to it: Hit@k (at least one evidence passage in the top k) and "all evidence in the top k"; both are derived from the stored Recall@k (above zero, equal to one), and all three coincide for Ukrainian. MRR@10 is the reciprocal rank of the first evidence passage, so Hit@1 ≤ MRR@10 ≤ Hit@10.
- **Other retrieval metrics.** Answer string in the top five passages, and ANN Recall@10 against exact search.
- **Answers.** Exact match and token F1 after SQuAD-style normalization, plus exact match over lemmas, which forgives case endings.
- **Uncertainty.** All intervals are 95% percentile intervals from a paired cluster bootstrap with 1,000 resamples and seed 271828. A cluster is a connected component of questions that share an evidence article, so questions about one article are resampled together. Method differences are computed on the same questions.
- **Seeds** are not independent samples: per-question values are averaged over seeds before resampling.

## Degradation law

Two forms are fitted to Recall@5 (for English the share of evidence groups found, not Hit@5) at 10, 30, 100 and 300 thousand passages:

- log-linear: `Recall@5 = a − b·log10(N)`;
- power: `1 − Recall@5 = c·N^d`.

The forecasts for 1 million passages with bootstrap intervals were written to `reports/degradation_preregistration.md` and committed before any measurement at 1 million (commit `3a64ddf` in the original repository, whose history is kept as a bundle). The report compares these fixed forecasts with the observation and with a naive last-value forecast (the value at 300 thousand).

## Generation

- **Model.** `qwen/qwen3-235b-a22b-2507` through OpenRouter, provider pinned to DeepInfra `deepinfra/fp8` without fallback, temperature 0, at most 128 new tokens.
- **Questions.** 200 test questions per language, chosen by hash before any generation.
- **Conditions.** No retrieval, the top five passages of each of eight methods at 10 thousand, 100 thousand and 1 million passages (random growth, seed 42), and the evidence passages (oracle).
- **Error analysis.** A wrong answer is a retrieval error when no evidence group is in the prompt and a reading error when all groups are.
- **Control.** A local `mlx-community/Qwen3-4B-4bit` answers the same Ukrainian prompts.

## Query router

The router sends a query to an expensive method (reranking or graph) only when a cheap probe suggests it will help.

- **Features.** From the HNSW top-50: score gaps between ranks 1–2, 1–10 and 1–50, entropy of the scores, number of distinct articles in the top 10 and top 20, and the logarithm of the corpus size.
- **Label.** The expensive method improves Recall@5 by at least 0.01. Labels and policy quality use Recall@5, which for English is the share of evidence groups found.
- **Models.** Logistic regression and gradient boosting, each with Platt calibration.
- **Splits without leakage.** Ukrainian: fit on train, calibrate and choose the threshold on two disjoint halves of the tune articles, evaluate on test. English: fit, calibration and selection are disjoint parts of tune. No article cluster is shared between parts.
- **Threshold.** The highest-quality threshold on the selection part that sends at most 25% of queries to an expensive method.
- **Baselines.** Always fast, always expensive, the exact expectation of random routing with the same share of every action in every corpus cell, and an oracle that knows the outcome.

## External anchor

`ufal/cus-qa` (text-UA, dev) contains questions written by native speakers with a reference Wikipedia article. All passages of the reference articles are added to the corpus, and article-level Recall@k of exact search is measured at every corpus size.

## Related work

- Reimers and Gurevych (ACL 2021) showed that dense retrieval degrades faster than BM25 as the index grows, and faster in low dimensions. This work adds native Ukrainian text, a hard growth order, a registered forecast and compensation methods.
- Wang et al., "BM25 Wins at Scale" (arXiv 2607.26497), fit power laws to cost only; here a law is fitted to Recall@5 and its extrapolation is tested.
- Genc et al., "Adaptive Re-Ranking" (ICTIR 2026), and Jeong et al., "Adaptive-RAG" (NAACL 2024), route by the query text. The router here uses score statistics of a cheap probe and the corpus size, and is compared with random routing at equal budget.
- Asai et al. (ICLR 2020) and HippoRAG (NeurIPS 2024) use trained or LLM-built graphs. Here the existing hyperlink graph is used without training, and its effect is tracked against corpus size.
- CUS-QA (TACL 2026) provides the native-speaker questions used as the external anchor.

## Limitations

- Questions are translations written from a source paragraph, and evidence was selected by LLM judges, not by people.
- The encoder was trained on SQuAD and HotpotQA, and part of the Ukrainian test questions are translations of SQuAD training questions.
- The Ukrainian test set has 41 article clusters, so intervals are wide.
- The Ukrainian and English results compare two benchmarks, not two languages: passage length, question type, the number of evidence pieces and encoder exposure all differ. The Ukrainian curve is steeper by Recall@5 and Hit@5 but not by "all evidence in the top five".
- English evidence is counted per supporting sentence. A paragraph-level recall cannot be computed without new rankings: `results/` keeps the share of evidence found, not the full lists of retrieved passages.
- Latency depends on hardware.
- The provider's generator does not repeat its answers exactly at temperature 0, so the significance of borderline answer-metric differences on 200 questions is unstable between runs.
- IVF-PQ differs slightly between platforms, by less than its spread over seeds. On one platform a second full run gave identical retrieval tables.
