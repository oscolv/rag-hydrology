"""CLI interface for the RAG system."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from rag.config import get_settings

app = typer.Typer(
    name="rag",
    help="Hydrology RAG System - Query research papers with AI",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    docs_dir: str = typer.Option(None, help="Override docs directory path"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion of all documents"),
    project_root: str = typer.Option(".", help="Project root directory"),
) -> None:
    """Parse PDFs, chunk, embed, and index into ChromaDB + BM25."""
    settings = get_settings(project_root)
    if docs_dir:
        settings.docs_dir = docs_dir

    from rag.ingest import ingest_documents

    ingest_documents(settings, force=force)


@app.command()
def query(
    question: str = typer.Argument(help="Question to ask"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show source documents"),
    project_root: str = typer.Option(".", help="Project root directory"),
) -> None:
    """Ask a question against the indexed documents."""
    settings = get_settings(project_root)

    from rag.retrieval import build_retriever
    from rag.generation import build_rag_chain_with_sources

    console.print("[bold]Building retrieval pipeline...[/bold]")
    retriever = build_retriever(settings)

    console.print("[bold]Querying...[/bold]\n")
    chain = build_rag_chain_with_sources(retriever, settings)
    result = chain(question)

    console.print(Panel(result["answer"], title="Answer", border_style="green"))

    if verbose and result["source_documents"]:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            console.print(f"  {i}. {source} (page {page})")
            if verbose:
                preview = doc.page_content[:200].replace("\n", " ")
                console.print(f"     [dim]{preview}...[/dim]")


@app.command(name="evaluate")
def evaluate_cmd(
    generate: bool = typer.Option(False, "--generate", "-g", help="Generate synthetic test set first"),
    testset: str = typer.Option(None, help="Path to existing test set CSV"),
    project_root: str = typer.Option(".", help="Project root directory"),
) -> None:
    """Run RAGAS evaluation on the RAG pipeline."""
    settings = get_settings(project_root)

    from rag.evaluation import generate_testset, run_evaluation
    from rag.retrieval import build_retriever
    from rag.generation import build_rag_chain_with_sources

    testset_path = Path(testset) if testset else None

    if generate:
        generate_testset(settings, output_path=testset_path)

    console.print("[bold]Building RAG pipeline for evaluation...[/bold]")
    retriever = build_retriever(settings)
    chain = build_rag_chain_with_sources(retriever, settings)

    run_evaluation(chain, settings, testset_path=testset_path)


@app.command()
def info(
    project_root: str = typer.Option(".", help="Project root directory"),
) -> None:
    """Show index statistics and configuration."""
    settings = get_settings(project_root)

    from rich.table import Table

    # Check ChromaDB
    chroma_path = settings.chroma_path
    if chroma_path.exists():
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.llm.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings,
            collection_name="hydrology_docs",
        )
        collection = vectorstore._collection
        count = collection.count()

        # Get unique sources
        results = vectorstore.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])

        table = Table(title="Index Statistics")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("ChromaDB path", str(chroma_path))
        table.add_row("Total chunks", str(count))
        table.add_row("Unique documents", str(len(sources)))
        table.add_row("Embedding model", settings.llm.embedding_model)
        console.print(table)

        if sources:
            console.print("\n[bold]Indexed documents:[/bold]")
            for s in sorted(sources):
                console.print(f"  • {s}")
    else:
        console.print("[yellow]No ChromaDB index found. Run 'rag ingest' first.[/yellow]")

    # Check BM25
    bm25_path = settings.bm25_full_path
    if bm25_path.exists():
        import os
        size_mb = os.path.getsize(bm25_path) / (1024 * 1024)
        console.print(f"\n[bold]BM25 index:[/bold] {bm25_path} ({size_mb:.1f} MB)")
    else:
        console.print("[yellow]No BM25 index found.[/yellow]")

    # Config summary
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  LLM: {settings.llm.model}")
    console.print(f"  Chunk size: {settings.chunking.chunk_size}")
    console.print(f"  Chunk overlap: {settings.chunking.chunk_overlap}")
    console.print(f"  Dense k: {settings.retrieval.dense_k}")
    console.print(f"  BM25 k: {settings.retrieval.bm25_k}")
    console.print(f"  Rerank top k: {settings.retrieval.rerank_top_k}")
    console.print(f"  Multi-query: {settings.retrieval.multi_query}")


if __name__ == "__main__":
    app()
