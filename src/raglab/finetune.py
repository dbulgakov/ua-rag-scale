import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

from raglab.config import LANGUAGES, Config
from raglab.data.corpus import Corpus, atomic, sha256
from raglab.embed import PASSAGE, QUERY, embed, encode, load_encoder
from raglab.evaluate import contains_answer, normalized_texts, retrieval_scores

Question = dict[str, object]


def training_questions(config: Config, corpus: Corpus) -> list[Question]:
    train = corpus.split("train").itertuples()
    rows = sorted(train, key=lambda row: sha256(json.dumps([config.lora_seed, row.id], ensure_ascii=False)))
    if corpus.language == "en":
        rows = rows[: config.lora_english_questions]
    return [
        {"id": row.id, "question": row.question, "answers": list(row.answers), "language": corpus.language}
        for row in rows
    ]


def mine_negatives(
    config: Config, corpus: Corpus, questions: list[Question], device: str
) -> dict[str, list[int]]:
    pool = torch.from_numpy(np.asarray(corpus.vectors())).to(device)
    queries = torch.from_numpy(corpus.queries([question["id"] for question in questions])).to(device)
    negatives = {}
    for start in range(0, len(questions), 256):
        batch = questions[start : start + 256]
        scores = queries[start : start + 256] @ pool.T
        found = torch.topk(scores, config.lora_negative_depth).indices.cpu().numpy()
        for question, rows in zip(batch, found, strict=True):
            positives = set().union(*corpus.gold[question["id"]])
            articles = {corpus.article[row] for row in positives}
            allowed = [
                int(row) for row in rows if row not in positives and corpus.article[row] not in articles
            ]
            negatives[question["id"]] = allowed[: config.lora_negatives]
    return negatives


def contamination(
    corpus: Corpus, questions: list[Question], negatives: dict[str, list[int]]
) -> dict[str, float]:
    texts = normalized_texts(corpus, (row for values in negatives.values() for row in values))
    hits = [
        [
            contains_answer([texts[row]], question["answers"], corpus.language)
            for row in negatives[question["id"]]
        ]
        for question in questions
    ]
    return {
        "questions": len(questions),
        "negatives": sum(map(len, hits)),
        "negatives_with_answer": sum(map(sum, hits)) / sum(map(len, hits)),
        "questions_with_such_negative": float(np.mean([any(flags) for flags in hits])),
    }


def epoch_batches(config: Config, questions: list[Question], epoch: int) -> list[list[Question]]:
    rng = np.random.default_rng(config.lora_seed + epoch)
    batches = []
    for language in sorted(LANGUAGES):
        group = [question for question in questions if question["language"] == language]
        order = rng.permutation(len(group))
        batches += [
            [group[i] for i in order[s : s + config.lora_batch]]
            for s in range(0, len(group), config.lora_batch)
        ]
    return [batches[i] for i in rng.permutation(len(batches))]


def multi_positive_loss(scores: torch.Tensor, positives: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~allowed, -1e4)
    targets = positives.float() / positives.sum(dim=1, keepdim=True)
    return -(targets * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()


def batch_passages(
    corpus: Corpus, batch: list[Question], negatives: dict[str, list[int]]
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    gold = [set().union(*corpus.gold[question["id"]]) for question in batch]
    rows = sorted(
        set().union(*gold, *(negatives[question["id"]] for question in batch)), key=lambda r: corpus.ids[r]
    )
    articles = corpus.article[rows]
    positives = torch.tensor([[row in wanted for row in rows] for wanted in gold])
    own = [{corpus.article[row] for row in wanted} for wanted in gold]
    same_article = torch.tensor([[article in group for article in articles] for group in own])
    return rows, positives, positives | ~same_article


def validation(config: Config, model: SentenceTransformer, corpora: dict[str, Corpus]) -> dict[str, float]:
    result = {}
    for language, corpus in corpora.items():
        rows = corpus.subset(*config.probe_cell, config.sizes[0])
        passages = encode(model, corpus.texts(rows.tolist()), PASSAGE)
        tune = corpus.split("dev")
        found = np.argsort(-(encode(model, tune["question"].tolist(), QUERY) @ passages.T), axis=1)[:, :10]
        recall = [
            retrieval_scores(rows[hits].tolist(), corpus.gold[question])["recall_10"]
            for question, hits in zip(tune["id"], found, strict=True)
        ]
        result[language] = float(np.mean(recall))
    return result | {"mean": float(np.mean(list(result.values())))}


def finetune(config: Config) -> None:
    target = config.work / "lora"
    report = target / "report.json"
    if not report.exists():
        corpora = {language: Corpus(config, language) for language in LANGUAGES}
        summary = train(config, corpora, target)
        with atomic(report) as path:
            path.write_text(json.dumps(summary, indent=2))
        del corpora
    config.results.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report, config.results / "finetune.json")
    for language in LANGUAGES:
        embed(config, language, "lora", target)


def train(config: Config, corpora: dict[str, Corpus], target: Path) -> dict[str, object]:
    torch.manual_seed(config.lora_seed)
    model = load_encoder(config)
    device = model.device.type
    questions, negatives, leaked = [], {}, {}
    for language, corpus in corpora.items():
        selected = training_questions(config, corpus)
        mined = mine_negatives(config, corpus, selected, device)
        leaked[language] = contamination(corpus, selected, mined)
        questions += selected
        negatives |= mined
    baseline = validation(config, model, corpora)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.add_adapter(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "value"],
        )
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lora_learning_rate)
    steps = (
        math.ceil(len(epoch_batches(config, questions, 0)) / config.lora_accumulation) * config.lora_epochs
    )
    warmup = max(1, int(steps * config.lora_warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / (steps - warmup)),
    )

    def vectors(texts: list[str]) -> torch.Tensor:
        chunks = [texts[start : start + 32] for start in range(0, len(texts), 32)]
        outputs = [model(batch_to_device(model.preprocess(chunk), model.device)) for chunk in chunks]
        return torch.cat([output["sentence_embedding"] for output in outputs])

    exposures = dict.fromkeys(corpora, 0)
    history, best = [], None
    for epoch in range(config.lora_epochs):
        model.train()
        batches = epoch_batches(config, questions, epoch)
        losses = []
        for number, batch in enumerate(batches, 1):
            corpus = corpora[batch[0]["language"]]
            rows, positives, allowed = batch_passages(corpus, batch, negatives)
            queries = vectors([QUERY + question["question"] for question in batch])
            passages = vectors([PASSAGE + text for text in corpus.texts(rows)])
            scores = queries @ passages.T / config.lora_temperature
            loss = multi_positive_loss(scores, positives.to(model.device), allowed.to(model.device))
            (loss / config.lora_accumulation).backward()
            losses.append(float(loss.detach()))
            exposures[corpus.language] += len(batch)
            if number % config.lora_accumulation == 0 or number == len(batches):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if device == "mps":
                torch.mps.empty_cache()
            if number % 100 == 0:
                print(
                    f"epoch {epoch + 1}: {number}/{len(batches)} loss {np.mean(losses[-100:]):.4f}",
                    flush=True,
                )
        model.eval()
        score = validation(config, model, corpora)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "tune_recall_10": score})
        if best is None or score["mean"] > best:
            best = score["mean"]
            model[0].auto_model.save_pretrained(target)
    return {
        "questions": {language: sum(q["language"] == language for q in questions) for language in corpora},
        "exposures": exposures,
        "baseline_tune_recall_10": baseline,
        "history": history,
        "hard_negatives": leaked,
    }
