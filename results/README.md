# Results

Small tables written by the pipeline. `raglab report` reads them and writes `reports/report_uk.md`, the figures, `demo.json` and `headline.json`; nothing else is needed to rebuild the report.

| File | Content | Written by |
|---|---|---|
| `questions.parquet` | Question id, split, article cluster, question type, low-specificity flag | `raglab grid` |
| `retrieval.parquet` | One row per question, method, corpus size, growth order and seed: Recall@1/5/10/20, MRR@10, answer in top five, ANN recall, and the router features of the HNSW probe | `raglab grid`, `raglab rerank` |
| `retrieved.parquet` | Top passages of each method for the 200 generation questions | `raglab grid`, `raglab rerank` |
| `latency.parquet` | Median single-query latency per method and size | `raglab grid --latency`, `raglab rerank --latency` |
| `generation.parquet`, `generation-local.parquet` | One row per answer: condition, method, size, answer, EM, F1, lemma EM, evidence in the prompt; the second file is the local MLX control | `raglab generate`, `raglab generate --local` |
| `anchor.parquet` | Article-level recall on CUS-QA and on the benchmark | `raglab embed` |
| `corpus.json` | Corpus, link-graph, specificity and token counts | `raglab data` |
| `finetune.json` | LoRA training log and hard-negative diagnostics | `raglab finetune` |
| `headline.json` | Every number quoted in the report, with intervals | `raglab report` |
| `demo.json` | Curves and recommendations shown by the demo | `raglab report` |

The tables in this folder come from the original run of the project, converted to this layout (`retrieved.parquet` was not kept). They are replaced when the pipeline is rerun.
