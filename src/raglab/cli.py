import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from raglab.config import LANGUAGES, ROOT, Config

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
run = typer.Typer(no_args_is_help=True)
app.add_typer(run, name="run", help="Run several stages in order.")

Out = Annotated[Path, typer.Option(help="Directory that receives work/ and results/.")]
Device = Annotated[str | None, typer.Option(help="cuda, mps or cpu; detected when omitted.")]


@app.command(help="Build both 1M passage pools, questions and link graphs from the raw sources.")
def data(out: Out = ROOT) -> None:
    from raglab.data.build import build

    build(Config(out=out))


@app.command(help="Encode passages and questions with the base encoder; evaluate the CUS-QA anchor.")
def embed(out: Out = ROOT, device: Device = None) -> None:
    from raglab import anchor
    from raglab.embed import embed as encode

    config = Config(out=out, device=device)
    for language in LANGUAGES:
        encode(config, language)
    anchor.run(config)


@app.command(help="Train the LoRA adapter on mined hard negatives and encode with it.")
def finetune(out: Out = ROOT, device: Device = None) -> None:
    from raglab.finetune import finetune as train

    train(Config(out=out, device=device))


@app.command(help="Run every retrieval method on every corpus size, growth order and seed.")
def grid(out: Out = ROOT, latency: bool = False) -> None:
    from raglab.evaluate import collect
    from raglab.grid import measure_latency, run_grid

    config = Config(out=out)
    for language in LANGUAGES:
        (measure_latency if latency else run_grid)(config, language)
    collect(config)


@app.command(help="Rerank HNSW candidates with the cross-encoder.")
def rerank(out: Out = ROOT, device: Device = None, latency: bool = False) -> None:
    from raglab import rerank as stage
    from raglab.evaluate import collect

    config = Config(out=out, device=device)
    for language in LANGUAGES:
        (stage.measure_latency if latency else stage.rerank)(config, language)
    collect(config)


@app.command(
    help="Answer 200 questions per language without retrieval, with retrieval and with gold passages."
)
def generate(out: Out = ROOT, max_cost: float = 3.0, local: bool = False) -> None:
    from raglab.generate import generate as answer

    answer(Config(out=out), max_cost, local)


@app.command(help="Rebuild the report and figures from results/.")
def report(out: Out = ROOT) -> None:
    from raglab.report import write_report

    write_report(Config(out=out))


@app.command(help="Start the demo.")
def serve(out: Out = ROOT, port: int = 8010) -> None:
    import uvicorn

    os.environ["RAGLAB_OUT"] = str(out)
    uvicorn.run("raglab.api:app", port=port)


@run.command("all", help="Run the whole pipeline; finished stages and cells are skipped.")
def run_all(out: Out = ROOT, device: Device = None, max_cost: float = 3.0) -> None:
    hardware = ["--device", device] if device else []
    stages = [
        ["data"],
        ["embed", *hardware],
        ["finetune", *hardware],
        ["grid"],
        ["rerank", *hardware],
        ["grid", "--latency"],
        ["rerank", "--latency", *hardware],
    ]
    if os.environ.get("OPENROUTER_API_KEY"):
        stages.append(["generate", "--max-cost", str(max_cost)])
    else:
        typer.echo("OPENROUTER_API_KEY is not set: the generation stage is skipped.")
    for stage in stages:
        subprocess.run([sys.executable, "-m", "raglab.cli", *stage, "--out", str(out)], check=True)
    if (Config(out=out).results / "generation.parquet").exists():
        report(out)


if __name__ == "__main__":
    app()
