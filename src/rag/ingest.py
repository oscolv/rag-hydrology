"""Ingestion pipeline: PDF parsing, chunking, embedding, and dual indexing."""

import hashlib
import pickle
import re
from pathlib import Path

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rag.config import Settings

console = Console()


def _file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_year(filename: str, text: str) -> str:
    """Try to extract publication year from filename or content."""
    # From filename patterns like "2024", "2012", "gleeson2015"
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return match.group(0)
    # From first page content
    match = re.search(r"(19|20)\d{2}", text[:2000])
    if match:
        return match.group(0)
    return "unknown"


def _detect_language(text: str) -> str:
    """Simple heuristic language detection."""
    spanish_words = {"de", "el", "la", "en", "los", "las", "del", "por", "con", "una"}
    words = set(text[:3000].lower().split())
    spanish_count = len(words & spanish_words)
    return "es" if spanish_count >= 5 else "en"


def _extract_section_header(text: str) -> str:
    """Extract the nearest section header from chunk text."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:80]
    return ""


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Parse a PDF into per-page Markdown chunks with metadata."""
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return pages


def build_chunks(
    pages: list[dict],
    source_name: str,
    file_hash: str,
    settings: Settings,
) -> list[Document]:
    """Split parsed pages into chunks with contextual headers and metadata."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        length_function=len,
    )

    full_text = "\n\n".join(p.get("text", "") for p in pages)
    year = _extract_year(source_name, full_text)
    language = _detect_language(full_text)

    # Extract title from first page
    first_text = pages[0].get("text", "") if pages else ""
    title_lines = [l.strip() for l in first_text.split("\n") if l.strip()]
    title = title_lines[0][:120] if title_lines else source_name

    all_chunks = []
    for page_data in pages:
        text = page_data.get("text", "")
        if not text.strip():
            continue

        page_meta = page_data.get("metadata", {})
        page_num = page_meta.get("page", 0) + 1  # 0-indexed to 1-indexed

        page_docs = splitter.create_documents([text])

        for doc in page_docs:
            section = _extract_section_header(doc.page_content)

            # Contextual header (Anthropic Contextual Retrieval technique)
            header = f"[From: {title} | Page: {page_num}"
            if section:
                header += f" | Section: {section}"
            header += "]"

            doc.page_content = f"{header}\n{doc.page_content}"
            doc.metadata = {
                "source": source_name,
                "file_hash": file_hash,
                "page": page_num,
                "title": title,
                "section": section,
                "year": year,
                "language": language,
            }
            all_chunks.append(doc)

    return all_chunks


def ingest_documents(settings: Settings, force: bool = False) -> dict:
    """Full ingestion pipeline: parse, chunk, embed, and index all PDFs.

    Returns statistics dict with counts.
    """
    docs_path = settings.docs_path
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[red]No PDF files found in {docs_path}[/red]")
        return {"pdfs": 0, "chunks": 0, "skipped": 0}

    # Deduplication: compute hashes and skip duplicates
    seen_hashes: dict[str, str] = {}
    unique_files: list[tuple[Path, str]] = []
    skipped = 0

    for pdf in pdf_files:
        h = _file_hash(pdf)
        if h in seen_hashes:
            console.print(
                f"[yellow]Skipping duplicate:[/yellow] {pdf.name} "
                f"(same as {seen_hashes[h]})"
            )
            skipped += 1
        else:
            seen_hashes[h] = pdf.name
            unique_files.append((pdf, h))

    # Parse and chunk all PDFs
    all_chunks: list[Document] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing PDFs...", total=len(unique_files))

        for pdf_path, file_hash in unique_files:
            progress.update(task, description=f"Parsing {pdf_path.name}...")
            try:
                pages = parse_pdf(pdf_path)
                chunks = build_chunks(pages, pdf_path.name, file_hash, settings)
                all_chunks.extend(chunks)
                console.print(
                    f"  [green]✓[/green] {pdf_path.name}: "
                    f"{len(pages)} pages → {len(chunks)} chunks"
                )
            except Exception as e:
                console.print(f"  [red]✗[/red] {pdf_path.name}: {e}")
            progress.advance(task)

    if not all_chunks:
        console.print("[red]No chunks generated.[/red]")
        return {"pdfs": len(unique_files), "chunks": 0, "skipped": skipped}

    # Clear existing index if force
    chroma_path = settings.chroma_path
    if force and chroma_path.exists():
        import shutil
        shutil.rmtree(chroma_path)
        console.print("[yellow]Cleared existing ChromaDB index.[/yellow]")

    bm25_full_path = settings.bm25_full_path
    if force and bm25_full_path.exists():
        bm25_full_path.unlink()

    # Create data directory
    chroma_path.parent.mkdir(parents=True, exist_ok=True)

    # Index into ChromaDB
    console.print("\n[bold]Indexing into ChromaDB...[/bold]")
    embeddings = OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    # Batch add to avoid memory issues with large corpora
    batch_size = 100
    vectorstore = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
        collection_name="hydrology_docs",
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(all_chunks))
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            vectorstore.add_documents(batch)
            progress.advance(task, advance=len(batch))

    # Build BM25 index
    console.print("[bold]Building BM25 index...[/bold]")
    corpus_tokens = [doc.page_content.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(corpus_tokens)

    bm25_full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_full_path, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": all_chunks}, f)

    stats = {
        "pdfs": len(unique_files),
        "chunks": len(all_chunks),
        "skipped": skipped,
    }
    console.print(
        f"\n[bold green]Ingestion complete:[/bold green] "
        f"{stats['pdfs']} PDFs → {stats['chunks']} chunks "
        f"({stats['skipped']} duplicates skipped)"
    )
    return stats
