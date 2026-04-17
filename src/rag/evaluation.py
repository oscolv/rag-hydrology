"""Evaluation pipeline using RAGAS metrics."""

import json
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.testset import TestsetGenerator
from rich.console import Console
from rich.table import Table

from rag.config import Settings

console = Console()


def load_documents_from_chroma(settings: Settings) -> list:
    """Load all documents from ChromaDB for test set generation."""
    embeddings = OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    vectorstore = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
        collection_name="hydrology_docs",
    )
    results = vectorstore.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    docs = []
    for text, meta in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def generate_testset(settings: Settings, output_path: Path | None = None) -> pd.DataFrame:
    """Generate a synthetic test set using RAGAS TestsetGenerator."""
    console.print("[bold]Generating synthetic test set...[/bold]")

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=settings.llm.model,
        openai_api_key=settings.openai_api_key,
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key=settings.openai_api_key,
    ))

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    docs = load_documents_from_chroma(settings)
    console.print(f"  Loaded {len(docs)} chunks from ChromaDB")

    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=settings.evaluation.test_set_size,
    )

    df = testset.to_pandas()

    if output_path is None:
        output_path = settings.project_root / "data" / "testset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Test set saved to {output_path}[/green] ({len(df)} questions)")
    return df


def run_evaluation(
    chain_with_sources: Callable,
    settings: Settings,
    testset_path: Path | None = None,
) -> dict:
    """Run RAGAS evaluation on the RAG chain.

    Args:
        chain_with_sources: Callable that takes a question string and returns
            {"answer": str, "source_documents": list[Document], "question": str}
        settings: Application settings
        testset_path: Path to CSV with test questions and ground truths
    """
    if testset_path is None:
        testset_path = settings.project_root / "data" / "testset.csv"

    if not testset_path.exists():
        console.print("[red]No test set found. Run with --generate first.[/red]")
        return {}

    df = pd.read_csv(testset_path)
    console.print(f"[bold]Running evaluation on {len(df)} questions...[/bold]")

    results = []
    for idx, row in df.iterrows():
        question = row["user_input"]
        console.print(f"  [{idx + 1}/{len(df)}] {question[:80]}...")

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
            console.print(f"    [red]Error: {e}[/red]")
            results.append({
                "user_input": question,
                "response": f"Error: {e}",
                "retrieved_contexts": [],
                "reference": row.get("reference_answer", row.get("reference", "")),
            })

    dataset = Dataset.from_list(results)

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=settings.llm.model,
        openai_api_key=settings.openai_api_key,
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key=settings.openai_api_key,
    ))

    console.print("[bold]Computing RAGAS metrics...[/bold]")
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    # Display results
    display_scores(scores)

    # Save detailed results
    results_path = settings.project_root / "data" / "eval_results.csv"
    scores_df = scores.to_pandas()
    scores_df.to_csv(results_path, index=False)
    console.print(f"\n[green]Detailed results saved to {results_path}[/green]")

    return scores


def display_scores(scores: dict) -> None:
    """Display evaluation scores in a rich table."""
    table = Table(title="RAGAS Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green", justify="right")

    score_dict = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
    for metric, score in sorted(score_dict.items()):
        table.add_row(metric, f"{score:.4f}")

    console.print(table)
