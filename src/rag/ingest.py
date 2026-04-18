"""Ingestion pipeline: PDF parsing, chunking, embedding, and dual indexing.

Supports three chunking strategies:
- Fixed-size (RecursiveCharacterTextSplitter) — default
- Semantic chunking — splits at embedding similarity breakpoints
- Contextual Retrieval — LLM generates situational context per chunk (Anthropic 2024)
"""

import hashlib
import pickle
import re
from pathlib import Path

import numpy as np
import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rag.config import Settings

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_year(filename: str, text: str) -> str:
    """Try to extract publication year from filename or content."""
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return match.group(0)
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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex heuristics for scientific text."""
    # Split on period/question/exclamation followed by space and uppercase,
    # but avoid splitting on abbreviations like "Dr.", "Fig.", "et al.", numbers like "3.5"
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z\[\(])',
        text.strip(),
    )
    # Filter empty and merge very short fragments
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if result and len(result[-1]) < 50:
            result[-1] = result[-1] + " " + s
        else:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Semantic Chunking
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def semantic_chunk(
    text: str,
    embeddings: OpenAIEmbeddings,
    similarity_threshold: float = 0.82,
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    window_size: int = 3,
) -> list[str]:
    """Split text into semantically coherent chunks using embedding similarity.

    Algorithm:
    1. Split text into sentences
    2. Create overlapping windows of sentences
    3. Embed each window
    4. Compute cosine similarity between consecutive windows
    5. Split at breakpoints where similarity drops below threshold
    6. Enforce min/max chunk size constraints
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    # Build sentence windows for more stable embeddings
    windows = []
    for i in range(len(sentences)):
        start = max(0, i - window_size // 2)
        end = min(len(sentences), i + window_size // 2 + 1)
        windows.append(" ".join(sentences[start:end]))

    # Batch embed all windows
    window_embeddings = embeddings.embed_documents(windows)
    emb_array = np.array(window_embeddings)

    # Compute similarities between consecutive windows
    similarities = []
    for i in range(len(emb_array) - 1):
        sim = _cosine_similarity(emb_array[i], emb_array[i + 1])
        similarities.append(sim)

    # Find breakpoints where similarity drops below threshold
    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            breakpoints.append(i + 1)  # Split AFTER sentence i

    # Build chunks from sentence groups between breakpoints
    chunks = []
    start = 0
    for bp in breakpoints:
        chunk_text = " ".join(sentences[start:bp]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start = bp

    # Don't forget the last segment
    if start < len(sentences):
        chunk_text = " ".join(sentences[start:]).strip()
        if chunk_text:
            chunks.append(chunk_text)

    # Enforce size constraints: merge small chunks, split large ones
    merged = _enforce_chunk_sizes(chunks, min_chunk_size, max_chunk_size)
    return merged


def _enforce_chunk_sizes(
    chunks: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """Merge chunks below min_size, split chunks above max_size."""
    # Merge small chunks with neighbors
    merged = []
    buffer = ""
    for chunk in chunks:
        if buffer and len(buffer) + len(chunk) + 1 <= max_size:
            buffer = buffer + " " + chunk
        elif buffer and len(buffer) < min_size:
            buffer = buffer + " " + chunk
        else:
            if buffer:
                merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)

    # Split oversized chunks with RecursiveCharacterTextSplitter
    final = []
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=max_size,
        chunk_overlap=100,
    )
    for chunk in merged:
        if len(chunk) > max_size:
            sub_docs = splitter.create_documents([chunk])
            final.extend(d.page_content for d in sub_docs)
        else:
            final.append(chunk)

    return final


# ---------------------------------------------------------------------------
# Contextual Retrieval (Anthropic 2024)
# ---------------------------------------------------------------------------

_CONTEXT_PROMPT = """<document>
{document_excerpt}
</document>
Here is the chunk we want to situate within the overall document:
<chunk>
{chunk_content}
</chunk>
Give a short, succinct context (2-3 sentences) to situate this chunk within \
the overall document for the purposes of improving search retrieval of the chunk. \
Mention the document title, topic area, and how this chunk relates to the broader content. \
Answer ONLY with the contextual summary, nothing else."""


def _build_context_llm(settings: Settings) -> ChatOpenAI:
    """Build the LLM for contextual retrieval generation."""
    model = settings.chunking.context_model or settings.llm.model
    kwargs = {
        "model": model,
        "temperature": 0,
        "openai_api_key": settings.llm_api_key,
        "max_retries": 5,
        "request_timeout": 60,
    }
    if settings.llm_base_url:
        kwargs["openai_api_base"] = settings.llm_base_url
    return ChatOpenAI(**kwargs)


def generate_chunk_contexts(
    chunks: list[Document],
    document_text: str,
    settings: Settings,
) -> list[Document]:
    """Add LLM-generated contextual summaries to each chunk.

    For each chunk, the LLM sees the full document excerpt + the chunk,
    and generates a short context that situates the chunk within the document.
    This context is prepended to the chunk content.
    """
    llm = _build_context_llm(settings)

    # Truncate document to fit in context window (~8000 chars ≈ 2000 tokens)
    doc_excerpt = document_text[:8000]
    if len(document_text) > 8000:
        doc_excerpt += "\n[... document continues ...]"

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("human", _CONTEXT_PROMPT),
    ])

    chain = prompt | llm

    contextualized = []
    for chunk in chunks:
        try:
            result = chain.invoke({
                "document_excerpt": doc_excerpt,
                "chunk_content": chunk.page_content,
            })
            context = result.content.strip()
            # Prepend context to chunk
            chunk.page_content = f"<context>\n{context}\n</context>\n{chunk.page_content}"
            chunk.metadata["has_context"] = True
        except Exception as e:
            # If context generation fails, keep the chunk as-is
            console.print(f"    [yellow]Context generation failed: {str(e)[:60]}[/yellow]")
            chunk.metadata["has_context"] = False
        contextualized.append(chunk)

    return contextualized


# ---------------------------------------------------------------------------
# PDF Parsing & Chunk Building
# ---------------------------------------------------------------------------


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Parse a PDF into per-page Markdown chunks with metadata."""
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return pages


def build_chunks(
    pages: list[dict],
    source_name: str,
    file_hash: str,
    settings: Settings,
    embeddings: OpenAIEmbeddings | None = None,
) -> list[Document]:
    """Split parsed pages into chunks with contextual headers and metadata.

    Supports two modes:
    - Fixed-size: RecursiveCharacterTextSplitter (default)
    - Semantic: embedding-based similarity breakpoints (when settings.chunking.semantic=True)
    """
    use_semantic = settings.chunking.semantic and embeddings is not None

    full_text = "\n\n".join(p.get("text", "") for p in pages)
    year = _extract_year(source_name, full_text)
    language = _detect_language(full_text)

    # Extract title from first page
    first_text = pages[0].get("text", "") if pages else ""
    title_lines = [l.strip() for l in first_text.split("\n") if l.strip()]
    title = title_lines[0][:120] if title_lines else source_name

    if use_semantic:
        return _build_chunks_semantic(
            pages, source_name, file_hash, title, year, language, settings, embeddings,
        )
    else:
        return _build_chunks_fixed(
            pages, source_name, file_hash, title, year, language, settings,
        )


def _build_chunks_fixed(
    pages: list[dict],
    source_name: str,
    file_hash: str,
    title: str,
    year: str,
    language: str,
    settings: Settings,
) -> list[Document]:
    """Fixed-size chunking with RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        length_function=len,
    )

    all_chunks = []
    for page_data in pages:
        text = page_data.get("text", "")
        if not text.strip():
            continue

        page_meta = page_data.get("metadata", {})
        page_num = page_meta.get("page", 0) + 1

        page_docs = splitter.create_documents([text])

        for doc in page_docs:
            section = _extract_section_header(doc.page_content)
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
                "chunking": "fixed",
            }
            all_chunks.append(doc)

    return all_chunks


def _build_chunks_semantic(
    pages: list[dict],
    source_name: str,
    file_hash: str,
    title: str,
    year: str,
    language: str,
    settings: Settings,
    embeddings: OpenAIEmbeddings,
) -> list[Document]:
    """Semantic chunking: splits at embedding similarity breakpoints."""
    all_chunks = []

    for page_data in pages:
        text = page_data.get("text", "")
        if not text.strip():
            continue

        page_meta = page_data.get("metadata", {})
        page_num = page_meta.get("page", 0) + 1

        chunk_texts = semantic_chunk(
            text=text,
            embeddings=embeddings,
            similarity_threshold=settings.chunking.similarity_threshold,
            max_chunk_size=settings.chunking.chunk_size + 500,  # Allow slightly larger semantic chunks
            min_chunk_size=max(100, settings.chunking.chunk_size // 5),
        )

        for chunk_text in chunk_texts:
            section = _extract_section_header(chunk_text)
            header = f"[From: {title} | Page: {page_num}"
            if section:
                header += f" | Section: {section}"
            header += "]"

            doc = Document(
                page_content=f"{header}\n{chunk_text}",
                metadata={
                    "source": source_name,
                    "file_hash": file_hash,
                    "page": page_num,
                    "title": title,
                    "section": section,
                    "year": year,
                    "language": language,
                    "chunking": "semantic",
                },
            )
            all_chunks.append(doc)

    return all_chunks


# ---------------------------------------------------------------------------
# Full Ingestion Pipeline
# ---------------------------------------------------------------------------


def ingest_documents(settings: Settings, force: bool = False) -> dict:
    """Full ingestion pipeline: parse, chunk, embed, and index all PDFs.

    Returns statistics dict with counts.
    """
    docs_path = settings.docs_path
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[red]No PDF files found in {docs_path}[/red]")
        return {"pdfs": 0, "chunks": 0, "skipped": 0}

    use_semantic = settings.chunking.semantic
    use_contextual = settings.chunking.contextual_retrieval

    if use_semantic:
        console.print("[bold cyan]Semantic Chunking[/bold cyan] activado")
        console.print(f"  Umbral de similitud: {settings.chunking.similarity_threshold}")
    if use_contextual:
        ctx_model = settings.chunking.context_model or settings.llm.model
        console.print("[bold cyan]Contextual Retrieval[/bold cyan] activado")
        console.print(f"  Modelo de contexto: {ctx_model}")

    # Build embeddings (needed for both semantic chunking and ChromaDB)
    embeddings = OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

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
                chunks = build_chunks(
                    pages, pdf_path.name, file_hash, settings,
                    embeddings=embeddings if use_semantic else None,
                )

                # Contextual Retrieval: generate LLM context per chunk
                if use_contextual and chunks:
                    progress.update(task, description=f"Contextualizing {pdf_path.name}...")
                    full_text = "\n\n".join(p.get("text", "") for p in pages)
                    chunks = generate_chunk_contexts(chunks, full_text, settings)
                    ctx_count = sum(1 for c in chunks if c.metadata.get("has_context"))
                    console.print(
                        f"  [green]✓[/green] {pdf_path.name}: "
                        f"{len(pages)} pages → {len(chunks)} chunks "
                        f"({ctx_count} contextualized)"
                    )
                else:
                    console.print(
                        f"  [green]✓[/green] {pdf_path.name}: "
                        f"{len(pages)} pages → {len(chunks)} chunks"
                    )

                all_chunks.extend(chunks)
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

    # Batch add to avoid memory issues with large corpora
    batch_size = 100
    vectorstore = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
        collection_name=settings.domain.collection_name,
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

    chunking_mode = "semantic" if use_semantic else "fixed"
    stats = {
        "pdfs": len(unique_files),
        "chunks": len(all_chunks),
        "skipped": skipped,
        "chunking": chunking_mode,
        "contextual": use_contextual,
    }
    console.print(
        f"\n[bold green]Ingestion complete:[/bold green] "
        f"{stats['pdfs']} PDFs → {stats['chunks']} chunks "
        f"({stats['skipped']} duplicates skipped)"
    )
    console.print(
        f"  Chunking: [cyan]{chunking_mode}[/cyan] | "
        f"Contextual Retrieval: [cyan]{'ON' if use_contextual else 'OFF'}[/cyan]"
    )
    return stats
