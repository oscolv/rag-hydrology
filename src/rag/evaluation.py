"""Evaluation pipeline using RAGAS metrics with rate-limit handling."""

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from rich import box
from rich.console import Console
from rich.table import Table

from rag.config import Settings
from rag.factories import get_embeddings

console = Console()

# Max requests per minute for rate limiting (conservative for free tier)
_REQUEST_DELAY = 1.5  # seconds between evaluation queries


def _build_eval_llm(settings: Settings) -> ChatOpenAI:
    """Build the LLM for RAGAS evaluation.

    Always uses OpenAI directly (gpt-4o-mini by default) because RAGAS
    requires strict JSON output parsing that is only reliable with OpenAI models.
    """
    return ChatOpenAI(
        model=settings.evaluation.eval_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
        max_retries=5,
        request_timeout=120,
    )


def load_documents_from_chroma(settings: Settings) -> list:
    """Load all documents from ChromaDB for test set generation."""
    embeddings = get_embeddings(settings)
    vectorstore = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
        collection_name=settings.domain.collection_name,
    )
    results = vectorstore.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    docs = []
    for text, meta in zip(results["documents"], results["metadatas"], strict=False):
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def generate_testset(settings: Settings, output_path: Path | None = None, testset_size: int | None = None) -> pd.DataFrame:
    """Generate a synthetic test set using RAGAS TestsetGenerator.

    Always uses gpt-4o-mini (OpenAI) for generation — RAGAS requires strict
    JSON parsing that is only reliable with OpenAI models.
    """
    console.print("[bold]Generando test set sintetico...[/bold]")
    console.print(f"  Modelo RAGAS: [cyan]{settings.evaluation.eval_model}[/cyan] (OpenAI)")

    llm = LangchainLLMWrapper(_build_eval_llm(settings))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings(settings))

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    docs = load_documents_from_chroma(settings)
    console.print(f"  {len(docs)} chunks cargados de ChromaDB")

    # Sample docs to reduce rate limit pressure (use a representative subset)
    max_docs = min(len(docs), 150)
    if len(docs) > max_docs:
        import random
        random.seed(42)
        # Reproducible sampling for evaluation — not a security-sensitive use.
        docs = random.sample(docs, max_docs)  # nosec B311
        console.print(f"  Usando muestra de {max_docs} chunks para generacion")

    test_size = testset_size or settings.evaluation.test_set_size
    console.print(f"  Generando {test_size} preguntas (esto puede tardar varios minutos)...")

    # RunConfig: limit concurrency to avoid rate limits on free tier
    run_cfg = RunConfig(max_workers=4, max_retries=15, max_wait=90, timeout=300)

    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
        run_config=run_cfg,
    )

    df = testset.to_pandas()

    if output_path is None:
        output_path = settings.project_root / "data" / "testset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Test set guardado en {output_path}[/green] ({len(df)} preguntas)")
    return df


def run_evaluation(
    chain_with_sources: Callable,
    settings: Settings,
    testset_path: Path | None = None,
) -> dict:
    """Run RAGAS evaluation on the RAG chain with rate-limit handling.

    The RAG chain uses the configured LLM (e.g. Trinity via OpenRouter) to
    generate answers, while RAGAS metrics are computed with gpt-4o-mini (OpenAI).

    Args:
        chain_with_sources: Callable that takes a question string and returns
            {"answer": str, "source_documents": list[Document], "question": str}
        settings: Application settings
        testset_path: Path to CSV with test questions and ground truths
    """
    if testset_path is None:
        testset_path = settings.project_root / "data" / "testset.csv"

    if not testset_path.exists():
        console.print("[red]No se encontro test set.[/red]")
        console.print("  Ejecuta: [cyan]rag evaluate --generate[/cyan]")
        return {}

    df = pd.read_csv(testset_path)
    total = len(df)
    console.print(f"[bold]Ejecutando evaluacion en {total} preguntas...[/bold]")
    console.print(f"  RAG pipeline: [cyan]{settings.llm.model}[/cyan]")
    console.print(f"  Metricas RAGAS: [cyan]{settings.evaluation.eval_model}[/cyan] (OpenAI)")
    console.print(f"  Delay entre queries: [cyan]{_REQUEST_DELAY}s[/cyan] (rate limit)\n")

    results = []
    errors = 0

    for idx, row in df.iterrows():
        question = row["user_input"]
        console.print(f"  [{idx + 1}/{total}] {question[:70]}...")

        try:
            response = chain_with_sources(question)
            results.append({
                "user_input": question,
                "response": response["answer"],
                "retrieved_contexts": [
                    d.page_content for d in response["source_documents"]
                ],
                "reference": row.get("reference_answer", row.get("reference", "")),
            })
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                console.print("    [yellow]Rate limit, esperando 30s...[/yellow]")
                time.sleep(30)
                # Retry once
                try:
                    response = chain_with_sources(question)
                    results.append({
                        "user_input": question,
                        "response": response["answer"],
                        "retrieved_contexts": [
                            d.page_content for d in response["source_documents"]
                        ],
                        "reference": row.get("reference_answer", row.get("reference", "")),
                    })
                    continue
                except Exception as retry_err:
                    console.print(
                        f"    [red]Retry failed: {str(retry_err)[:80]}[/red]"
                    )

            console.print(f"    [red]Error: {error_msg[:80]}[/red]")
            errors += 1
            results.append({
                "user_input": question,
                "response": f"Error: {error_msg}",
                "retrieved_contexts": [],
                "reference": row.get("reference_answer", row.get("reference", "")),
            })

        # Rate limit delay between queries
        if idx < total - 1:
            time.sleep(_REQUEST_DELAY)

    if errors:
        console.print(f"\n[yellow]{errors} pregunta(s) con error.[/yellow]")

    dataset = Dataset.from_list(results)

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    eval_llm = LangchainLLMWrapper(_build_eval_llm(settings))
    eval_embeddings = LangchainEmbeddingsWrapper(get_embeddings(settings))

    console.print("\n[bold]Calculando metricas RAGAS...[/bold]")
    console.print("  [dim]Esto puede tardar varios minutos con rate limiting[/dim]")

    run_cfg = RunConfig(max_workers=4, max_retries=15, max_wait=90, timeout=300)

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_cfg,
    )

    # Display results
    display_scores(scores)

    # Save detailed results
    results_path = settings.project_root / "data" / "eval_results.csv"
    scores_df = scores.to_pandas()
    scores_df.to_csv(results_path, index=False)
    console.print(f"\n[green]Resultados detallados guardados en {results_path}[/green]")

    return scores


def display_scores(scores) -> None:
    """Display evaluation scores in a rich table.

    Works with RAGAS 0.4 EvaluationResult (uses .to_pandas() to extract averages).
    """
    table = Table(title="Resultados RAGAS", show_edge=False, box=box.SIMPLE)
    table.add_column("Metrica", style="cyan")
    table.add_column("Score", style="green", justify="right")
    table.add_column("", style="dim")

    descriptions = {
        "faithfulness": "Fidelidad al contexto (no alucina)",
        "answer_relevancy": "Relevancia de la respuesta",
        "response_relevancy": "Relevancia de la respuesta",
        "context_precision": "Precision del contexto recuperado",
        "llm_context_precision_without_reference": "Precision del contexto (sin referencia)",
        "context_recall": "Cobertura del contexto necesario",
        "llm_context_recall": "Cobertura del contexto (LLM)",
    }

    # Extract per-metric averages from EvaluationResult
    df = scores.to_pandas()
    for col in sorted(df.columns):
        if col in ("user_input", "response", "retrieved_contexts", "reference"):
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        avg = float(values.mean())
        desc = descriptions.get(col, "")
        if avg >= 0.8:
            score_str = f"[bold green]{avg:.4f}[/bold green]"
        elif avg >= 0.5:
            score_str = f"[yellow]{avg:.4f}[/yellow]"
        else:
            score_str = f"[red]{avg:.4f}[/red]"
        table.add_row(col, score_str, desc)

    console.print(table)
