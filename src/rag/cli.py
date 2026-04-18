"""Professional CLI interface for the RAG system."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from rag.collections import CollectionManager, valid_collection_name
from rag.config import Settings, get_settings

console = Console()

LOGO = r"""[bold blue]
  ____      _    ____   _   _           _           _
 |  _ \    / \  / ___| | | | |_   _  __| |_ __ ___ | | ___   __ _ _   _
 | |_) |  / _ \| |  _  | |_| | | | |/ _` | '__/ _ \| |/ _ \ / _` | | | |
 |  _ <  / ___ \ |_| | |  _  | |_| | (_| | | | (_) | | (_) | (_| | |_| |
 |_| \_\/_/   \_\____| |_| |_|\__, |\__,_|_|  \___/|_|\___/ \__, |\__, |
                                |___/                         |___/ |___/
[/bold blue]"""

# ============================================================================
# Health check & diagnostics
# ============================================================================

class SystemStatus:
    """Check system readiness and report issues with solutions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.issues: list[tuple[str, str]] = []  # (problem, solution)

    def check_api_keys(self) -> bool:
        ok = True
        if not self.settings.openai_api_key:
            self.issues.append((
                "OPENAI_API_KEY no configurada",
                "Agrega OPENAI_API_KEY=sk-... a tu archivo .env",
            ))
            ok = False
        if not self.settings.cohere_api_key:
            self.issues.append((
                "COHERE_API_KEY no configurada",
                "Agrega COHERE_API_KEY=... a tu archivo .env\n"
                "         Obtenla gratis en dashboard.cohere.com > API Keys",
            ))
            ok = False
        # Check OpenRouter if using an external base_url
        uses_openrouter = self.settings.llm_base_url and "openrouter" in self.settings.llm_base_url
        if uses_openrouter and not self.settings.openrouter_api_key:
            self.issues.append((
                "OPENROUTER_API_KEY no configurada (requerida por el modelo actual)",
                "Agrega OPENROUTER_API_KEY=sk-or-v1-... a tu archivo .env\n"
                "         Obtenla en openrouter.ai/keys",
            ))
            ok = False
        return ok

    def check_index(self) -> bool:
        if not self.settings.chroma_path.exists():
            self.issues.append((
                "No hay indice creado",
                "Ejecuta: [bold cyan]rag ingest[/bold cyan]",
            ))
            return False
        if not self.settings.bm25_full_path.exists():
            self.issues.append((
                "Falta el indice BM25",
                "Ejecuta: [bold cyan]rag ingest --force[/bold cyan]",
            ))
            return False
        return True

    def check_docs(self) -> bool:
        docs = list(self.settings.docs_path.glob("*.pdf"))
        if not docs:
            self.issues.append((
                f"No hay PDFs en {self.settings.docs_path}",
                "Agrega tus PDFs:\n"
                "         [bold cyan]rag docs add archivo.pdf[/bold cyan]\n"
                "         o copia PDFs directamente a ./docs/",
            ))
            return False
        return True

    def check_all(self) -> bool:
        self.check_docs()
        self.check_api_keys()
        self.check_index()
        return len(self.issues) == 0

    def print_report(self) -> None:
        if not self.issues:
            console.print("[bold green]Todo listo.[/bold green] El sistema esta configurado correctamente.\n")
            return

        console.print(f"[bold yellow]Se encontraron {len(self.issues)} problema(s):[/bold yellow]\n")
        for i, (problem, solution) in enumerate(self.issues, 1):
            console.print(f"  [bold red]{i}.[/bold red] {problem}")
            console.print(f"         {solution}\n")


def _require_ready(settings: Settings, need_index: bool = True) -> bool:
    """Verify the system is ready. Print friendly diagnostics on failure."""
    status = SystemStatus(settings)
    status.check_api_keys()
    if need_index:
        status.check_index()
    if status.issues:
        console.print()
        status.print_report()
        return False
    return True


# ============================================================================
# Main app
# ============================================================================

app = typer.Typer(
    name="rag",
    help="RAG — Consulta documentos PDF con IA (dominio configurable en config.yaml)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
docs_app = typer.Typer(help="Gestionar documentos PDF")
config_app = typer.Typer(help="Ver y modificar configuracion")
collection_app = typer.Typer(help="Gestionar colecciones (corpus independientes)")
app.add_typer(docs_app, name="docs")
app.add_typer(config_app, name="config")
app.add_typer(collection_app, name="collection")


def _apply_collection(collection: str | None) -> None:
    """Set RAG_COLLECTION env var so get_settings() picks it up.

    Must be called BEFORE any call to get_settings() in the command body,
    since settings are lru_cached by project_root only.
    """
    if collection:
        os.environ["RAG_COLLECTION"] = collection
        get_settings.cache_clear()


# ============================================================================
# rag status  (health check)
# ============================================================================

@app.command()
def status(
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz del proyecto"),
) -> None:
    """Diagnostico completo del sistema: API keys, documentos, indices."""
    settings = get_settings(project_root)

    console.print(LOGO)
    console.print("[dim]Verificando el sistema...[/dim]\n")

    checker = SystemStatus(settings)

    # API Keys
    openai_ok = bool(settings.openai_api_key)
    cohere_ok = bool(settings.cohere_api_key)
    _print_check("OPENAI_API_KEY", openai_ok)
    _print_check("COHERE_API_KEY", cohere_ok)

    # OpenRouter (only show if using it)
    uses_openrouter = settings.llm_base_url and "openrouter" in settings.llm_base_url
    if uses_openrouter:
        or_ok = bool(settings.openrouter_api_key)
        _print_check("OPENROUTER_API_KEY", or_ok, hint="Agrega OPENROUTER_API_KEY=sk-or-v1-... a .env")

    # Model info
    model_label = f"Modelo LLM: {settings.llm.model}"
    if uses_openrouter:
        model_label += " (via OpenRouter)"
    _print_check(model_label, True)

    # Advanced features
    _print_check(
        f"Semantic Chunking: {'ON' if settings.chunking.semantic else 'OFF'}",
        True,
    )
    _print_check(
        f"Contextual Retrieval: {'ON' if settings.chunking.contextual_retrieval else 'OFF'}",
        True,
    )
    _print_check(
        f"Self-RAG: {'ON' if settings.retrieval.self_rag else 'OFF'}",
        True,
    )

    # Docs
    pdfs = list(settings.docs_path.glob("*.pdf"))
    _print_check(f"Documentos PDF ({len(pdfs)} encontrados)", len(pdfs) > 0)

    # Index
    chroma_ok = settings.chroma_path.exists()
    bm25_ok = settings.bm25_full_path.exists()
    _print_check("Indice ChromaDB", chroma_ok)
    _print_check("Indice BM25", bm25_ok)

    # Tesseract
    import shutil
    tess_ok = shutil.which("tesseract") is not None
    _print_check("Tesseract OCR", tess_ok, hint="sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa")

    console.print()

    checker.check_all()
    if checker.issues:
        checker.print_report()
        _print_next_steps(settings)
    else:
        console.print("[bold green]El sistema esta listo para consultas.[/bold green]")
        console.print("\n  Prueba: [bold cyan]rag chat[/bold cyan]")


def _print_check(label: str, ok: bool, hint: str | None = None) -> None:
    icon = "[bold green]OK[/bold green]" if ok else "[bold red]--[/bold red]"
    console.print(f"  {icon}  {label}")
    if not ok and hint:
        console.print(f"       [dim]Fix: {hint}[/dim]")


def _print_next_steps(settings: Settings) -> None:
    """Suggest the logical next step based on current state."""
    console.print("[bold]Siguiente paso:[/bold]")

    if not settings.openai_api_key:
        console.print("  1. Crea el archivo .env con tu OPENAI_API_KEY")
        console.print("     [dim]cp .env.example .env && nano .env[/dim]")
        return

    pdfs = list(settings.docs_path.glob("*.pdf"))
    if not pdfs:
        console.print("  1. Agrega PDFs a ./docs/")
        console.print("     [dim]rag docs add tus_archivos.pdf[/dim]")
        return

    if not settings.chroma_path.exists():
        console.print(f"  1. Indexa los {len(pdfs)} documentos")
        console.print("     [dim]rag ingest[/dim]")
        return

    console.print("  1. Inicia una sesion de consulta")
    console.print("     [dim]rag chat[/dim]")


# ============================================================================
# rag ingest
# ============================================================================

@app.command()
def ingest(
    docs_dir: str = typer.Option(None, help="Directorio alternativo de PDFs"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-indexar todo desde cero"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Indexar PDFs: parsear, dividir en chunks, crear embeddings y construir indices.

    \b
    Que hace:
      1. Parsea cada PDF preservando tablas y estructura (pymupdf4llm)
      2. Divide el texto en chunks inteligentes
      3. Genera embeddings y los almacena en ChromaDB
      4. Construye un indice BM25 para busqueda lexica

    \b
    Tecnicas avanzadas (activar en config.yaml):
      Semantic Chunking         Divide por similitud semantica, no por tamano fijo
                                chunking.semantic: true
      Contextual Retrieval      Un LLM genera contexto situacional por cada chunk
                                chunking.contextual_retrieval: true
                                (Tecnica Anthropic 2024, reduce errores de retrieval ~49%)

    \b
    Ejemplos:
      rag ingest                Indexar nuevos PDFs
      rag ingest --force        Re-indexar todo desde cero
      rag ingest -c papers      Indexar en la coleccion 'papers'
      rag config set chunking.semantic true     Activar semantic chunking
      rag config set chunking.contextual_retrieval true  Activar contextual retrieval
    """
    _apply_collection(collection)
    settings = get_settings(project_root)
    if docs_dir:
        settings.docs_dir = docs_dir

    if not _require_ready(settings, need_index=False):
        return

    pdfs = list(settings.docs_path.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No hay PDFs en {settings.docs_path}[/yellow]")
        console.print("  Agrega documentos: [cyan]rag docs add archivo.pdf[/cyan]")
        raise typer.Exit(1)

    if force:
        console.print("[yellow]Modo --force: se re-indexaran todos los documentos.[/yellow]\n")

    from rag.ingest import ingest_documents
    stats = ingest_documents(settings, force=force)

    # Next step suggestion
    if stats.get("chunks", 0) > 0:
        console.print("\n[bold]Que puedes hacer ahora:[/bold]")
        console.print("  [cyan]rag chat[/cyan]          Iniciar sesion interactiva")
        console.print("  [cyan]rag query \"...\"[/cyan]   Hacer una consulta rapida")
        console.print("  [cyan]rag info[/cyan]          Ver estadisticas del indice")


# ============================================================================
# rag query
# ============================================================================

@app.command()
def query(
    question: str = typer.Argument(help="Pregunta a realizar"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mostrar fuentes con preview"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Streaming de tokens"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Hacer una consulta rapida contra los documentos indexados.

    \b
    El pipeline busca con retrieval hibrido (dense + BM25), aplica reranking,
    y genera una respuesta con citas. Si Self-RAG esta activo, verifica
    la respuesta antes de mostrarla (grading + hallucination check).

    \b
    Ejemplos:
      rag query "What is GRACE?"
      rag query "Que contiene el atlas del agua?" -v
      rag query "TWS estimation methods" --verbose
      rag query "..." --collection papers --no-stream
    """
    _apply_collection(collection)
    settings = get_settings(project_root)
    if not _require_ready(settings):
        raise typer.Exit(1)

    from rag.retrieval import build_retriever

    with console.status("[bold]Cargando pipeline...[/bold]"):
        retriever = build_retriever(settings)

    if stream:
        result = _run_streaming_query(retriever, settings, question)
    else:
        from rag.generation import build_rag_chain_with_sources
        chain = build_rag_chain_with_sources(retriever, settings)
        with console.status("[bold]Buscando y generando respuesta...[/bold]"):
            result = chain(question)
        _display_answer(result, verbose=verbose)

    if verbose and result.get("source_documents"):
        _display_sources(result["source_documents"], detailed=True)

    console.print("\n[dim]Tip: Usa [cyan]rag chat[/cyan] para una sesion interactiva con historial.[/dim]")


# ============================================================================
# rag chat
# ============================================================================

@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mostrar fuentes siempre"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Streaming de tokens"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Iniciar sesion interactiva de consulta con tus documentos.

    \b
    Escribe tu pregunta en lenguaje natural. El sistema busca en todos
    los documentos indexados y genera una respuesta con citas.

    \b
    Si Self-RAG esta activado (retrieval.self_rag: true), cada respuesta
    pasa por verificacion automatica:
      1. Grading    - filtra documentos irrelevantes
      2. Reformula  - si no hay suficiente contexto, reformula y reintenta
      3. Genera     - produce la respuesta con citas
      4. Verifica   - detecta alucinaciones y regenera si es necesario

    \b
    Comandos disponibles dentro del chat:
      /sources     Ver fuentes de la ultima respuesta
      /reflection  Ver log de verificacion Self-RAG
      /history     Ver historial de la conversacion
      /export      Exportar sesion a archivo Markdown
      /verbose     Alternar modo detallado (fuentes + Self-RAG)
      /help        Ver todos los comandos
      /quit        Salir
    """
    _apply_collection(collection)
    settings = get_settings(project_root)
    if not _require_ready(settings):
        raise typer.Exit(1)

    from rag.generation import build_rag_chain_with_sources
    from rag.retrieval import build_retriever

    # Welcome
    console.print(LOGO)
    welcome_text = (
        f"[bold]Sesion interactiva — {settings.domain.name}[/bold]\n\n"
        "Pregunta lo que necesites sobre tus documentos indexados.\n"
        "El sistema busca en todo el corpus y genera una respuesta\n"
        "con citas a las fuentes originales.\n"
    )
    console.print(Panel(welcome_text, border_style="blue", padding=(1, 2)))

    # Quick-start guide
    console.print("[bold]Como funciona:[/bold]")
    console.print("  1. Escribe tu pregunta en lenguaje natural (espanol o ingles)")
    console.print("  2. El sistema busca en los documentos usando retrieval hibrido (dense + BM25)")
    console.print("  3. Reranking selecciona los fragmentos mas relevantes")
    console.print("  4. El LLM genera una respuesta con citas [Source: archivo, Page: N]")
    if settings.retrieval.self_rag:
        console.print("  5. [cyan]Self-RAG[/cyan] verifica la respuesta: grading, hallucination check")
    console.print()

    # Active features
    features = []
    if settings.chunking.semantic:
        features.append("[cyan]Semantic Chunking[/cyan]")
    if settings.chunking.contextual_retrieval:
        features.append("[cyan]Contextual Retrieval[/cyan]")
    if settings.retrieval.multi_query:
        features.append("[cyan]Multi-Query[/cyan]")
    if settings.retrieval.self_rag:
        features.append("[cyan]Self-RAG[/cyan]")
    if features:
        console.print(f"[bold]Tecnicas activas:[/bold] {' + '.join(features)}\n")

    if settings.domain.example_queries:
        console.print("[bold]Ejemplos de preguntas:[/bold]")
        for q in settings.domain.example_queries:
            console.print(f'  [dim]"{q}"[/dim]')
        console.print()

    console.print("[bold]Tips:[/bold]")
    console.print("  - Escribe [cyan]/[/cyan] + [cyan]Tab[/cyan] para ver los comandos disponibles")
    console.print("  - Usa [cyan]/sources[/cyan] para ver las fuentes en detalle")
    if settings.retrieval.self_rag:
        console.print("  - Usa [cyan]/reflection[/cyan] para ver el log de verificacion Self-RAG")
        console.print("  - Activa [cyan]/verbose[/cyan] para ver fuentes + Self-RAG automaticamente")
    console.print("  - Usa [cyan]/export[/cyan] para guardar la sesion como archivo")
    console.print(f"  - Modelo activo: [cyan]{settings.llm.model}[/cyan]")
    console.print()

    with console.status("[bold]Cargando pipeline (esto puede tardar unos segundos)...[/bold]"):
        retriever = build_retriever(settings)
        chain = None if stream else build_rag_chain_with_sources(retriever, settings)

    console.print("[green]Listo.[/green] Escribe tu primera pregunta.\n")

    # Tab completion for slash commands
    chat_prompt_session = _build_chat_prompt()

    history: list[dict] = []
    last_result: dict | None = None
    show_sources = verbose
    query_count = 0

    while True:
        try:
            question = chat_prompt_session.prompt("Pregunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            _chat_goodbye(history)
            break

        if not question:
            console.print("[dim]Escribe una pregunta o /help para ver comandos.[/dim]")
            continue

        # Slash commands
        if question.startswith("/"):
            cmd = question.lower().split()[0]
            args = question.split(maxsplit=1)[1] if " " in question else None

            # Just "/" shows quick command menu
            if question.strip() == "/":
                _chat_quick_commands()
                continue

            if cmd in ("/quit", "/q", "/exit"):
                _chat_goodbye(history)
                break

            elif cmd == "/help":
                _chat_help()

            elif cmd in ("/sources", "/s"):
                if last_result and last_result.get("source_documents"):
                    _display_sources(last_result["source_documents"], detailed=True)
                else:
                    console.print("[yellow]Aun no hay fuentes. Haz una pregunta primero.[/yellow]")

            elif cmd in ("/history", "/h"):
                _display_history(history)

            elif cmd in ("/export", "/e"):
                _export_session(history, args)

            elif cmd == "/clear":
                history.clear()
                last_result = None
                query_count = 0
                console.print("[green]Historial limpiado.[/green]")

            elif cmd in ("/verbose", "/v"):
                show_sources = not show_sources
                state = "activado" if show_sources else "desactivado"
                console.print(f"[green]Modo detallado {state}.[/green]")

            elif cmd in ("/info", "/i"):
                _chat_quick_info(settings)

            elif cmd in ("/model", "/m"):
                console.print(f"  [cyan]Modelo:[/cyan]     {settings.llm.model}")
                if settings.llm_base_url:
                    console.print(f"  [cyan]Proveedor:[/cyan]  {settings.llm_base_url}")

            elif cmd in ("/reflection", "/r"):
                if last_result and last_result.get("reflection"):
                    _display_reflection(last_result["reflection"])
                elif settings.retrieval.self_rag:
                    console.print("[yellow]Aun no hay reflection. Haz una pregunta primero.[/yellow]")
                else:
                    console.print("[yellow]Self-RAG no esta activado. Activa con: rag config set retrieval.self_rag true[/yellow]")

            else:
                console.print(f"[yellow]Comando desconocido: {cmd}[/yellow]")
                _chat_quick_commands()

            continue

        # Run query
        query_count += 1
        try:
            if stream:
                result = _run_streaming_query(retriever, settings, question)
            else:
                with console.status("[bold]Buscando en los documentos y generando respuesta...[/bold]"):
                    result = chain(question)
        except Exception as e:
            console.print(f"\n[bold red]Error al procesar la consulta:[/bold red] {e}")
            console.print("[dim]Intenta reformular tu pregunta o verifica tu conexion.[/dim]")
            continue

        last_result = result
        history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": result["answer"],
            "sources": [
                {"source": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?")}
                for d in result.get("source_documents", [])
            ],
        })

        if not stream:
            _display_answer(result, verbose=show_sources)
        elif show_sources:
            _display_sources(result.get("source_documents", []), detailed=True)

        # Show Self-RAG reflection summary in verbose mode
        if show_sources and result.get("reflection"):
            _display_reflection(result["reflection"])

        # Contextual tips for new users
        if query_count == 1:
            console.print("[dim]Tip: Escribe /sources para ver las fuentes en detalle.[/dim]")
        elif query_count == 3:
            console.print("[dim]Tip: Usa /export para guardar esta sesion como archivo.[/dim]")

        console.print()


# ============================================================================
# rag search
# ============================================================================

@app.command()
def search(
    query_text: str = typer.Argument(help="Texto de busqueda"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Numero de resultados"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Buscar documentos sin usar el LLM (solo retrieval).

    \b
    Util para:
      - Verificar que el indice encuentra documentos relevantes
      - Depurar problemas de relevancia
      - Buscar fragmentos especificos sin generar respuesta
    """
    _apply_collection(collection)
    settings = get_settings(project_root)
    if not _require_ready(settings):
        raise typer.Exit(1)

    from rag.retrieval import build_retriever

    with console.status("[bold]Buscando...[/bold]"):
        retriever = build_retriever(settings)
        docs = retriever.invoke(query_text)

    docs = docs[:top_k]

    if not docs:
        console.print("[yellow]No se encontraron resultados.[/yellow]")
        console.print("[dim]Sugerencias:[/dim]")
        console.print("  - Intenta con terminos mas generales")
        console.print("  - Verifica que los documentos esten indexados: [cyan]rag docs list[/cyan]")
        return

    console.print(f"\n[bold]Top {len(docs)} resultados para:[/bold] \"{query_text}\"\n")

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        year = doc.metadata.get("year", "")
        section = doc.metadata.get("section", "")

        # Header
        header = f"[bold cyan]#{i}[/bold cyan]  {source}"
        meta_parts = []
        if page != "?":
            meta_parts.append(f"p.{page}")
        if year and year != "unknown":
            meta_parts.append(year)
        if meta_parts:
            header += f"  [dim]({', '.join(meta_parts)})[/dim]"
        console.print(header)

        if section:
            console.print(f"    [italic dim]Section: {section}[/italic dim]")

        content = doc.page_content
        if content.startswith("[From:"):
            content = content.split("]\n", 1)[-1]
        preview = content[:300].replace("\n", "\n    ")
        console.print(f"    {preview}")
        if len(content) > 300:
            console.print(f"    [dim]... ({len(content)} caracteres en total)[/dim]")
        console.print()


# ============================================================================
# rag export
# ============================================================================

@app.command()
def export(
    query_text: str = typer.Argument(help="Pregunta a realizar"),
    output: str = typer.Option(None, "--output", "-o", help="Ruta del archivo de salida"),
    format: str = typer.Option("markdown", "--format", "-f", help="Formato: markdown o json"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Consultar y exportar la respuesta con fuentes a un archivo.

    \b
    Formatos disponibles:
      markdown  Documento .md con respuesta y fuentes formateadas
      json      Datos estructurados con metadata completa

    \b
    Ejemplos:
      rag export "Explain GRACE-FO" -o report.md
      rag export "TWS methods" -f json -o results.json
    """
    _apply_collection(collection)
    settings = get_settings(project_root)
    if not _require_ready(settings):
        raise typer.Exit(1)

    from rag.generation import build_rag_chain_with_sources
    from rag.retrieval import build_retriever

    with console.status("[bold]Cargando pipeline...[/bold]"):
        retriever = build_retriever(settings)
        chain = build_rag_chain_with_sources(retriever, settings)

    with console.status("[bold]Generando respuesta...[/bold]"):
        result = chain(query_text)

    ext = ".json" if format == "json" else ".md"
    if output is None:
        slug = "".join(c if c.isalnum() or c in " _-" else "" for c in query_text[:40])
        slug = slug.strip().replace(" ", "_")
        output = f"data/export_{slug}{ext}"

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "timestamp": datetime.now().isoformat(),
            "question": query_text,
            "answer": result["answer"],
            "sources": [
                {
                    "source": d.metadata.get("source", "?"),
                    "page": d.metadata.get("page", "?"),
                    "year": d.metadata.get("year", "?"),
                    "content": d.page_content,
                }
                for d in result.get("source_documents", [])
            ],
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        lines = [
            f"# Query: {query_text}\n",
            f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            "## Respuesta\n",
            result["answer"] + "\n",
            "## Fuentes\n",
        ]
        for i, doc in enumerate(result.get("source_documents", []), 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            lines.append(f"### {i}. {source} (pagina {page})\n")
            content = doc.page_content
            if content.startswith("[From:"):
                content = content.split("]\n", 1)[-1]
            lines.append(f"```\n{content[:500]}\n```\n")

        with open(output, "w") as f:
            f.write("\n".join(lines))

    console.print(f"\n[bold green]Exportado a:[/bold green] {output}")
    _display_answer(result, verbose=False)


# ============================================================================
# rag evaluate
# ============================================================================

@app.command(name="evaluate")
def evaluate_cmd(
    generate: bool = typer.Option(False, "--generate", "-g", help="Generar test set sintetico primero"),
    size: int = typer.Option(None, "--size", "-s", help="Numero de preguntas a generar (default: config.yaml)"),
    testset: str = typer.Option(None, help="Ruta a un test set CSV existente"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Evaluar la calidad del sistema RAG con metricas RAGAS.

    \b
    Metricas:
      Faithfulness         La respuesta es fiel al contexto (no alucina)
      Response Relevancy   La respuesta es relevante a la pregunta
      Context Precision    Los documentos relevantes estan bien rankeados
      Context Recall       Se recuperaron todos los documentos necesarios

    \b
    Como funciona:
      - Las RESPUESTAS las genera el pipeline RAG con tu modelo configurado
        (ej: arcee-ai/trinity-large-thinking via OpenRouter)
      - Las METRICAS RAGAS las calcula gpt-4o-mini (OpenAI) porque RAGAS
        requiere parsing JSON estricto que solo es confiable con modelos OpenAI

    \b
    Flujo tipico:
      rag evaluate --generate              # Genera test set + evalua
      rag evaluate --generate --size 10    # Rapido, 10 preguntas
      rag evaluate                         # Re-evalua con test set existente

    \b
    Comparar tecnicas:
      rag config set retrieval.self_rag false
      rag evaluate --generate --size 10    # Baseline sin Self-RAG
      rag config set retrieval.self_rag true
      rag evaluate                         # Con Self-RAG (mismo test set)
    """
    _apply_collection(collection)
    settings = get_settings(project_root)
    if not _require_ready(settings):
        raise typer.Exit(1)

    from rag.evaluation import generate_testset, run_evaluation
    from rag.generation import build_rag_chain_with_sources
    from rag.retrieval import build_retriever

    testset_path = Path(testset) if testset else None

    if generate:
        console.print("[bold]Paso 1/2: Generando test set sintetico...[/bold]\n")
        generate_testset(settings, output_path=testset_path, testset_size=size)
        console.print()

    console.print("[bold]Construyendo pipeline para evaluacion...[/bold]")
    console.print(f"  RAG pipeline: [cyan]{settings.llm.model}[/cyan]")
    console.print(f"  Metricas RAGAS: [cyan]{settings.evaluation.eval_model}[/cyan] (OpenAI)\n")
    retriever = build_retriever(settings)
    chain = build_rag_chain_with_sources(retriever, settings)

    if generate:
        console.print("[bold]Paso 2/2: Ejecutando evaluacion...[/bold]\n")
    run_evaluation(chain, settings, testset_path=testset_path)


# ============================================================================
# rag info
# ============================================================================

@app.command()
def info(
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Ver estadisticas del indice, inventario de documentos y configuracion."""
    _apply_collection(collection)
    settings = get_settings(project_root)

    chroma_path = settings.chroma_path
    if not chroma_path.exists():
        console.print("[yellow]No hay indice creado.[/yellow]")
        console.print("\n[bold]Para empezar:[/bold]")
        console.print("  1. Coloca PDFs en ./docs/")
        console.print("  2. Ejecuta: [cyan]rag ingest[/cyan]")
        return

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    with console.status("[bold]Cargando informacion del indice...[/bold]"):
        embeddings = OpenAIEmbeddings(
            model=settings.llm.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings,
            collection_name=settings.domain.collection_name,
        )
        count = vectorstore._collection.count()
        results = vectorstore.get(include=["metadatas"])

    # Aggregate per-document stats
    doc_stats: dict[str, dict] = {}
    for meta in results["metadatas"]:
        if not meta:
            continue
        source = meta.get("source", "unknown")
        if source not in doc_stats:
            doc_stats[source] = {
                "chunks": 0, "pages": set(),
                "year": meta.get("year", "?"),
                "lang": meta.get("language", "?"),
            }
        doc_stats[source]["chunks"] += 1
        doc_stats[source]["pages"].add(meta.get("page", 0))

    # Overview
    overview = Table(title="Resumen del Indice", show_edge=False, box=box.SIMPLE)
    overview.add_column("", style="cyan")
    overview.add_column("", style="green")
    overview.add_row("Total de chunks", str(count))
    overview.add_row("Documentos", str(len(doc_stats)))
    overview.add_row("Modelo de embeddings", settings.llm.embedding_model)

    bm25_path = settings.bm25_full_path
    if bm25_path.exists():
        size_mb = os.path.getsize(bm25_path) / (1024 * 1024)
        overview.add_row("Indice BM25", f"{size_mb:.1f} MB")
    console.print(overview)

    # Per-document table
    console.print()
    doc_table = Table(title="Documentos Indexados", show_edge=False, box=box.SIMPLE)
    doc_table.add_column("#", style="dim", justify="right")
    doc_table.add_column("Documento", style="white")
    doc_table.add_column("Ano", style="cyan", justify="center")
    doc_table.add_column("Idioma", justify="center")
    doc_table.add_column("Pags", style="green", justify="right")
    doc_table.add_column("Chunks", style="green", justify="right")

    for i, (source, stats) in enumerate(sorted(doc_stats.items()), 1):
        name = source[:58] + "..." if len(source) > 58 else source
        lang_icon = "ES" if stats["lang"] == "es" else "EN"
        doc_table.add_row(
            str(i), name, stats["year"], lang_icon,
            str(len(stats["pages"])), str(stats["chunks"]),
        )
    console.print(doc_table)

    # Config
    console.print()
    cfg = Table(title="Configuracion Actual", show_edge=False, box=box.SIMPLE)
    cfg.add_column("Parametro", style="cyan")
    cfg.add_column("Valor", style="green")
    cfg.add_row("Modelo LLM", settings.llm.model)
    cfg.add_row("Temperatura", str(settings.llm.temperature))
    cfg.add_row("Chunk size / overlap", f"{settings.chunking.chunk_size} / {settings.chunking.chunk_overlap}")
    cfg.add_row("Semantic Chunking", f"{'Si (umbral: ' + str(settings.chunking.similarity_threshold) + ')' if settings.chunking.semantic else 'No'}")
    cfg.add_row("Contextual Retrieval", "Si" if settings.chunking.contextual_retrieval else "No")
    cfg.add_row("Retrieval (dense + BM25)", f"{settings.retrieval.dense_k} + {settings.retrieval.bm25_k}")
    cfg.add_row("Rerank top-k", str(settings.retrieval.rerank_top_k))
    cfg.add_row("Multi-query", "Si" if settings.retrieval.multi_query else "No")
    cfg.add_row("Self-RAG", f"{'Si (max retries: ' + str(settings.retrieval.self_rag_max_retries) + ')' if settings.retrieval.self_rag else 'No'}")
    console.print(cfg)


# ============================================================================
# rag docs list|add|remove
# ============================================================================

@docs_app.command("list")
def docs_list(
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Listar PDFs y su estado de indexacion."""
    _apply_collection(collection)
    settings = get_settings(project_root)
    docs_path = settings.docs_path
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[yellow]No hay PDFs en {docs_path}/[/yellow]")
        console.print("\n[bold]Para agregar documentos:[/bold]")
        console.print("  [cyan]rag docs add archivo.pdf[/cyan]")
        console.print("  o copia PDFs directamente a ./docs/")
        return

    # Check indexed
    indexed_sources: set[str] = set()
    if settings.chroma_path.exists():
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.llm.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        vectorstore = Chroma(
            persist_directory=str(settings.chroma_path),
            embedding_function=embeddings,
            collection_name=settings.domain.collection_name,
        )
        results = vectorstore.get(include=["metadatas"])
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                indexed_sources.add(meta["source"])

    table = Table(title="Documentos", show_edge=False, box=box.SIMPLE)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Archivo", style="white")
    table.add_column("Tamano", style="cyan", justify="right")
    table.add_column("Indexado", justify="center")

    not_indexed = 0
    for i, pdf in enumerate(pdf_files, 1):
        size = pdf.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        else:
            size_str = f"{size / 1024:.0f} KB"

        is_indexed = pdf.name in indexed_sources
        if not is_indexed:
            not_indexed += 1
        indexed = "[green]Si[/green]" if is_indexed else "[red]No[/red]"
        name = pdf.name[:63] + "..." if len(pdf.name) > 63 else pdf.name
        table.add_row(str(i), name, size_str, indexed)

    console.print(table)
    console.print(f"\n  [dim]{len(pdf_files)} archivos, {len(indexed_sources)} indexados[/dim]")

    if not_indexed > 0:
        console.print(f"\n  [yellow]{not_indexed} archivo(s) sin indexar.[/yellow]")
        console.print("  Ejecuta [cyan]rag ingest[/cyan] para indexarlos.")


@docs_app.command("add")
def docs_add(
    paths: list[str] = typer.Argument(help="Rutas de archivos PDF a agregar"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Agregar archivos PDF al directorio de documentos.

    \b
    Ejemplo:
      rag docs add paper1.pdf paper2.pdf ~/downloads/estudio.pdf
    """
    import shutil
    _apply_collection(collection)
    settings = get_settings(project_root)
    docs_path = settings.docs_path
    docs_path.mkdir(parents=True, exist_ok=True)

    added = 0
    for p in paths:
        src = Path(p).expanduser()
        if not src.exists():
            console.print(f"  [bold red]x[/bold red] No encontrado: {p}")
            continue
        if src.suffix.lower() != ".pdf":
            console.print(f"  [yellow]![/yellow] No es PDF, omitido: {p}")
            continue

        dest = docs_path / src.name
        if dest.exists():
            console.print(f"  [yellow]![/yellow] Ya existe: {src.name}")
            continue

        shutil.copy2(src, dest)
        size = dest.stat().st_size / (1024 * 1024)
        console.print(f"  [bold green]+[/bold green] {src.name} ({size:.1f} MB)")
        added += 1

    if added:
        console.print(f"\n[green]{added} archivo(s) agregado(s).[/green]")
        console.print("Siguiente paso: [cyan]rag ingest[/cyan]  para indexarlos.")
    elif not paths:
        console.print("[yellow]No se especificaron archivos.[/yellow]")
        console.print("Uso: [cyan]rag docs add archivo1.pdf archivo2.pdf[/cyan]")


@docs_app.command("remove")
def docs_remove(
    names: list[str] = typer.Argument(help="Nombres de PDFs a eliminar"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirmar sin preguntar"),
    collection: str = typer.Option(None, "--collection", "-c", help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Eliminar PDFs del directorio y purgar del indice.

    \b
    Ejemplo:
      rag docs remove paper_viejo.pdf
      rag docs remove archivo1.pdf archivo2.pdf --yes
    """
    _apply_collection(collection)
    settings = get_settings(project_root)

    if not yes:
        file_list = ", ".join(names)
        confirm = typer.confirm(
            f"Se eliminaran {len(names)} archivo(s) ({file_list}) y sus chunks del indice. Continuar?"
        )
        if not confirm:
            console.print("[dim]Cancelado.[/dim]")
            return

    for name in names:
        file_path = settings.docs_path / name
        if file_path.exists():
            file_path.unlink()
            console.print(f"  [bold red]-[/bold red] Eliminado: {name}")
        else:
            console.print(f"  [yellow]![/yellow] Archivo no encontrado: {name}")

        # Purge from ChromaDB
        if settings.chroma_path.exists():
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=settings.llm.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
            vectorstore = Chroma(
                persist_directory=str(settings.chroma_path),
                embedding_function=embeddings,
                collection_name=settings.domain.collection_name,
            )
            results = vectorstore.get(where={"source": name}, include=[])
            if results["ids"]:
                vectorstore.delete(ids=results["ids"])
                console.print(f"       [dim]{len(results['ids'])} chunks eliminados del indice[/dim]")

    console.print("\n[dim]Nota: El indice BM25 necesita reconstruirse.[/dim]")
    console.print("Ejecuta: [cyan]rag ingest --force[/cyan]")


# ============================================================================
# rag config show|set
# ============================================================================

@config_app.command("show")
def config_show(
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Ver la configuracion actual del sistema."""
    settings = get_settings(project_root)
    config_path = settings.project_root / "config.yaml"

    table = Table(title=f"Configuracion ({config_path})", show_edge=False, box=box.SIMPLE)
    table.add_column("Seccion", style="cyan")
    table.add_column("Parametro", style="white")
    table.add_column("Valor", style="green")

    table.add_row("chunking", "chunk_size", str(settings.chunking.chunk_size))
    table.add_row("", "chunk_overlap", str(settings.chunking.chunk_overlap))
    table.add_row("", "semantic", str(settings.chunking.semantic))
    table.add_row("", "similarity_threshold", str(settings.chunking.similarity_threshold))
    table.add_row("", "contextual_retrieval", str(settings.chunking.contextual_retrieval))
    table.add_row("", "context_model", str(settings.chunking.context_model or "(usa llm.model)"))
    table.add_row("retrieval", "dense_k", str(settings.retrieval.dense_k))
    table.add_row("", "bm25_k", str(settings.retrieval.bm25_k))
    table.add_row("", "rerank_top_k", str(settings.retrieval.rerank_top_k))
    table.add_row("", "multi_query", str(settings.retrieval.multi_query))
    table.add_row("", "self_rag", str(settings.retrieval.self_rag))
    table.add_row("", "self_rag_max_retries", str(settings.retrieval.self_rag_max_retries))
    table.add_row("llm", "model", settings.llm.model)
    table.add_row("", "temperature", str(settings.llm.temperature))
    table.add_row("", "embedding_model", settings.llm.embedding_model)
    table.add_row("", "base_url", str(settings.llm.base_url or "(default OpenAI)"))
    table.add_row("evaluation", "test_set_size", str(settings.evaluation.test_set_size))
    table.add_row("", "eval_model", settings.evaluation.eval_model)
    console.print(table)

    # API keys
    console.print()
    openai_ok = "[green]configurada[/green]" if settings.openai_api_key else "[red]falta[/red]"
    cohere_ok = "[green]configurada[/green]" if settings.cohere_api_key else "[red]falta[/red]"
    console.print(f"  OPENAI_API_KEY:     {openai_ok}")
    console.print(f"  COHERE_API_KEY:     {cohere_ok}")
    if settings.llm_base_url and "openrouter" in settings.llm_base_url:
        or_ok = "[green]configurada[/green]" if settings.openrouter_api_key else "[red]falta[/red]"
        console.print(f"  OPENROUTER_API_KEY: {or_ok}")

    console.print("\n[dim]Modificar: rag config set seccion.parametro valor[/dim]")
    console.print("[dim]Ejemplo:   rag config set llm.model gpt-4o-mini[/dim]")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Clave (seccion.parametro)"),
    value: str = typer.Argument(help="Nuevo valor"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Modificar un parametro en config.yaml.

    \b
    Ejemplos:
      rag config set llm.model gpt-4o-mini
      rag config set chunking.semantic true
      rag config set retrieval.self_rag true

    \b
    Claves disponibles:
      CHUNKING
      chunking.chunk_size              Caracteres por chunk (default: 1000)
      chunking.chunk_overlap           Solapamiento entre chunks (default: 200)
      chunking.semantic                Chunking semantico por similitud (default: false)
      chunking.similarity_threshold    Umbral de corte semantico (default: 0.82)
      chunking.contextual_retrieval    Contexto LLM por chunk (default: false)
      chunking.context_model           Modelo para contexto (default: usa llm.model)

    \b
      RETRIEVAL
      retrieval.dense_k                Candidatos de busqueda densa (default: 20)
      retrieval.bm25_k                 Candidatos de busqueda BM25 (default: 20)
      retrieval.rerank_top_k           Documentos tras reranking (default: 5)
      retrieval.multi_query            Expansion multi-query (default: true)
      retrieval.self_rag               Self-RAG: grading + hallucination check (default: false)
      retrieval.self_rag_max_retries   Reintentos de retrieval en Self-RAG (default: 2)

    \b
      LLM
      llm.model                        Modelo de lenguaje (default: gpt-4o)
      llm.temperature                  Temperatura del LLM (default: 0.1)
      llm.embedding_model              Modelo de embeddings
      llm.base_url                     URL del proveedor (OpenRouter, etc.)

    \b
      EVALUACION
      evaluation.test_set_size         Preguntas en test sintetico (default: 30)
      evaluation.eval_model            Modelo OpenAI para RAGAS (default: gpt-4o-mini)
    """
    import yaml

    root = Path(project_root).resolve()
    config_path = root / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    parts = key.split(".")
    if len(parts) != 2:
        console.print("[red]Formato incorrecto.[/red] Usa: seccion.parametro")
        console.print("[dim]Ejemplo: rag config set llm.model gpt-4o-mini[/dim]")
        raise typer.Exit(1)

    valid_sections = {"chunking", "retrieval", "llm", "evaluation"}
    section, param = parts
    if section not in valid_sections:
        console.print(f"[red]Seccion desconocida: {section}[/red]")
        console.print(f"[dim]Secciones validas: {', '.join(sorted(valid_sections))}[/dim]")
        raise typer.Exit(1)

    if section not in data:
        data[section] = {}

    # Type coercion
    if value.lower() in ("true", "false"):
        typed_value = value.lower() == "true"
    elif value.isdigit():
        typed_value = int(value)
    else:
        try:
            typed_value = float(value)
        except ValueError:
            typed_value = value

    old_value = data[section].get(param, "[dim]no definido[/dim]")
    data[section][param] = typed_value

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    get_settings.cache_clear()

    console.print(f"\n  [bold green]{key}[/bold green]: {old_value} -> [bold]{typed_value}[/bold]")
    console.print(f"  [dim]Guardado en {config_path}[/dim]")

    # Warn if re-ingest needed
    if section == "chunking":
        console.print("\n  [yellow]Los cambios en chunking requieren re-indexar.[/yellow]")
        console.print("  Ejecuta: [cyan]rag ingest --force[/cyan]")
    elif section == "retrieval" and param in ("self_rag", "self_rag_max_retries"):
        console.print("\n  [dim]Self-RAG se aplica en tiempo de consulta, no requiere re-indexar.[/dim]")


# ============================================================================
# Display helpers
# ============================================================================

def _run_streaming_query(retriever, settings: Settings, question: str) -> dict:
    """Stream tokens into a live-updating Rich panel. Return a result dict.

    Mirrors the shape of build_rag_chain_with_sources output so downstream
    display/export helpers don't need to care about streaming.
    """
    from rich.live import Live
    from rich.markdown import Markdown

    from rag.generation import build_rag_chain_streaming

    chain = build_rag_chain_streaming(retriever, settings)

    answer_buffer = ""
    sources: list = []
    reflection: list[dict] = []
    question_final = question
    request_id = ""

    with Live(
        Panel("[dim]Buscando en el indice...[/dim]", title="Respuesta", border_style="green"),
        console=console,
        refresh_per_second=20,
        transient=False,
    ) as live:
        try:
            for event in chain(question):
                kind = event.get("event")
                if kind == "retrieval_start":
                    request_id = event.get("request_id", "")
                elif kind == "sources":
                    sources = event.get("documents", [])
                    live.update(Panel(
                        f"[dim]Generando respuesta ({len(sources)} fuentes)...[/dim]",
                        title="Respuesta", border_style="green",
                    ))
                elif kind == "reflection":
                    reflection.append(event.get("step", {}))
                elif kind == "token":
                    answer_buffer += event.get("content", "")
                    live.update(Panel(
                        Markdown(answer_buffer) if answer_buffer else "",
                        title="Respuesta",
                        border_style="green",
                        padding=(1, 2),
                    ))
                elif kind == "regenerating":
                    answer_buffer = ""  # reset for regenerated answer
                    live.update(Panel(
                        "[yellow]Regenerando (hallucination detectada)...[/yellow]",
                        title="Respuesta", border_style="yellow",
                    ))
                elif kind == "done":
                    answer_buffer = event.get("answer", answer_buffer)
                    sources = event.get("source_documents", sources)
                    if event.get("reflection"):
                        reflection = event["reflection"]
                    question_final = event.get("question", question)
                elif kind == "error":
                    live.update(Panel(
                        f"[red]{event.get('message', 'error')}[/red]",
                        title="Error", border_style="red",
                    ))
                    return {"answer": "", "source_documents": [], "question": question}
        except KeyboardInterrupt:
            live.update(Panel(
                answer_buffer + "\n\n[dim][cancelado][/dim]",
                title="Respuesta", border_style="yellow",
            ))

    # Compact source line below the live panel
    if sources:
        seen: list[str] = []
        for d in sources:
            s = d.metadata.get("source", "?")
            p = d.metadata.get("page", "?")
            tag = f"{s} p.{p}"
            if tag not in seen:
                seen.append(tag)
        console.print(f"  [dim]Fuentes: {' | '.join(seen)}[/dim]")

    return {
        "answer": answer_buffer,
        "source_documents": sources,
        "question": question_final,
        "request_id": request_id,
        "reflection": reflection,
    }


def _display_answer(result: dict, verbose: bool = False) -> None:
    console.print()
    console.print(Panel(
        Markdown(result["answer"]),
        title="Respuesta",
        border_style="green",
        padding=(1, 2),
    ))

    docs = result.get("source_documents", [])
    if not docs:
        return

    if verbose:
        _display_sources(docs, detailed=True)
    else:
        # Compact source line
        seen = []
        for d in docs:
            s = d.metadata.get("source", "?")
            p = d.metadata.get("page", "?")
            tag = f"{s} p.{p}"
            if tag not in seen:
                seen.append(tag)
        console.print(f"  [dim]Fuentes: {' | '.join(seen)}[/dim]")


def _display_sources(docs: list, detailed: bool = True) -> None:
    table = Table(title="Fuentes", show_edge=False, padding=(0, 1), box=box.SIMPLE)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Documento", style="cyan")
    table.add_column("Pag", justify="center")
    table.add_column("Seccion", style="dim")
    if detailed:
        table.add_column("Contenido", style="white", max_width=55)

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "?")
        page = str(doc.metadata.get("page", "?"))
        section = doc.metadata.get("section", "")[:35]

        if detailed:
            content = doc.page_content
            if content.startswith("[From:"):
                content = content.split("]\n", 1)[-1]
            preview = content[:100].replace("\n", " ")
            table.add_row(str(i), source[:45], page, section, preview)
        else:
            table.add_row(str(i), source[:45], page, section)

    console.print(table)


def _display_reflection(reflection: list[dict]) -> None:
    """Display Self-RAG reflection log in verbose mode."""
    console.print("\n  [bold dim]Self-RAG Reflection:[/bold dim]")
    for step in reflection:
        name = step.get("step", "")
        if name == "grade_documents":
            ratio = step.get("ratio", 0)
            color = "green" if ratio >= 0.5 else "yellow" if ratio >= 0.3 else "red"
            console.print(
                f"    [{color}]Grading[/{color}] attempt {step.get('attempt', '?')}: "
                f"{step.get('relevant', '?')}/{step.get('retrieved', '?')} relevant "
                f"({ratio:.0%})"
            )
        elif name == "reformulate":
            console.print(
                f"    [yellow]Reformulated[/yellow]: {step.get('reformulated', '')[:80]}"
            )
        elif name == "hallucination_check" or name == "hallucination_recheck":
            grounded = step.get("grounded", "?")
            relevant = step.get("relevant", "?")
            g_color = "green" if grounded == "yes" else "red"
            r_color = "green" if relevant == "yes" else "red"
            label = "Recheck" if "recheck" in name else "Check"
            console.print(
                f"    [{g_color}]Grounded: {grounded}[/{g_color}] | "
                f"[{r_color}]Relevant: {relevant}[/{r_color}]  ({label})"
            )
            if step.get("issues"):
                console.print(f"    [dim]Issues: {step['issues']}[/dim]")
        elif name == "regenerate":
            console.print(
                f"    [yellow]Regenerated[/yellow]: {step.get('reason', '')}"
            )


def _display_history(history: list[dict]) -> None:
    if not history:
        console.print("[yellow]Aun no hay historial. Haz una pregunta para empezar.[/yellow]")
        return

    console.print(Rule("Historial de la Sesion"))
    for i, entry in enumerate(history, 1):
        ts = entry["timestamp"][:16].replace("T", " ")
        console.print(f"\n  [dim]{ts}[/dim]")
        console.print(f"  [bold blue]P{i}:[/bold blue] {entry['question']}")
        answer_preview = entry["answer"][:200]
        if len(entry["answer"]) > 200:
            answer_preview += "..."
        console.print(f"  [bold green]R:[/bold green]  {answer_preview}")
        if entry.get("sources"):
            src_list = ", ".join(f"{s['source']} p.{s['page']}" for s in entry["sources"][:3])
            console.print(f"  [dim]Fuentes: {src_list}[/dim]")

    console.print(f"\n[dim]{len(history)} intercambio(s)[/dim]")


def _export_session(history: list[dict], output_path: str | None = None) -> None:
    if not history:
        console.print("[yellow]No hay historial para exportar. Haz algunas preguntas primero.[/yellow]")
        return

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/session_{ts}.md"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Sesion RAG — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
    ]
    for i, entry in enumerate(history, 1):
        ts = entry["timestamp"][:16].replace("T", " ")
        lines.append(f"## Pregunta {i}: {entry['question']}\n")
        lines.append(f"*{ts}*\n")
        lines.append(f"{entry['answer']}\n")
        if entry.get("sources"):
            lines.append("**Fuentes:**\n")
            for s in entry["sources"]:
                lines.append(f"- {s['source']} (pagina {s['page']})")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[bold green]Sesion exportada a:[/bold green] {output_path}")
    console.print(f"[dim]({len(history)} preguntas guardadas)[/dim]")


def _build_chat_prompt():
    """Build a prompt_toolkit session with slash command tab-completion."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.styles import Style

    style = Style.from_dict({
        "prompt": "bold ansiblue",
    })

    class SlashCompleter(Completer):
        commands: ClassVar[list[tuple[str, str]]] = [
            ("/sources", "Ver fuentes de la ultima respuesta"),
            ("/s", "Atajo de /sources"),
            ("/history", "Ver historial de la conversacion"),
            ("/h", "Atajo de /history"),
            ("/export", "Exportar sesion a Markdown"),
            ("/e", "Atajo de /export"),
            ("/verbose", "Alternar modo detallado"),
            ("/v", "Atajo de /verbose"),
            ("/info", "Ver estadisticas del indice"),
            ("/i", "Atajo de /info"),
            ("/model", "Ver modelo y proveedor actual"),
            ("/m", "Atajo de /model"),
            ("/reflection", "Ver log Self-RAG de ultima respuesta"),
            ("/r", "Atajo de /reflection"),
            ("/clear", "Limpiar historial"),
            ("/help", "Ver todos los comandos"),
            ("/quit", "Salir de la sesion"),
            ("/q", "Atajo de /quit"),
        ]

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            for cmd, desc in self.commands:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )

    return PromptSession(
        message=[("class:prompt", "Pregunta> ")],
        style=style,
        completer=SlashCompleter(),
        complete_while_typing=False,
    )


def _chat_quick_commands() -> None:
    """Show compact inline command list when user types just '/'."""
    cmds = [
        "[cyan]/sources[/cyan] [dim]/s[/dim]",
        "[cyan]/history[/cyan] [dim]/h[/dim]",
        "[cyan]/export[/cyan] [dim]/e[/dim]",
        "[cyan]/verbose[/cyan] [dim]/v[/dim]",
        "[cyan]/info[/cyan] [dim]/i[/dim]",
        "[cyan]/model[/cyan] [dim]/m[/dim]",
        "[cyan]/reflection[/cyan] [dim]/r[/dim]",
        "[cyan]/clear[/cyan]",
        "[cyan]/help[/cyan]",
        "[cyan]/quit[/cyan] [dim]/q[/dim]",
    ]
    console.print("  " + "  |  ".join(cmds))


def _chat_help() -> None:
    table = Table(show_header=False, show_edge=False, padding=(0, 2), box=box.SIMPLE)
    table.add_column(style="cyan bold")
    table.add_column(style="white")
    table.add_row("/sources  /s", "Ver fuentes de la ultima respuesta (documento, pagina, fragmento)")
    table.add_row("/reflection /r", "Ver log de verificacion Self-RAG: grading, hallucination check, retries")
    table.add_row("/verbose  /v", "Alternar modo detallado: muestra fuentes y Self-RAG automaticamente")
    table.add_row("/history  /h", "Ver historial de preguntas y respuestas de la sesion")
    table.add_row("/export   /e [path]", "Exportar sesion completa a archivo Markdown")
    table.add_row("/info     /i", "Ver estadisticas del indice (docs, chunks, modelo)")
    table.add_row("/model    /m", "Ver modelo LLM y proveedor activo")
    table.add_row("/clear", "Limpiar historial de la sesion")
    table.add_row("/help", "Mostrar esta ayuda")
    table.add_row("/quit     /q", "Salir de la sesion")
    console.print(Panel(table, title="Comandos disponibles", border_style="blue", padding=(1, 2)))


def _chat_goodbye(history: list[dict]) -> None:
    console.print()
    if history:
        console.print(f"[dim]Sesion finalizada ({len(history)} preguntas).[/dim]")
        console.print("[dim]Tip: Usa /export antes de salir para guardar el historial.[/dim]")
    else:
        console.print("[dim]Sesion finalizada.[/dim]")


def _chat_quick_info(settings: Settings) -> None:
    """Quick index stats for chat session."""
    if not settings.chroma_path.exists():
        console.print("[yellow]No hay indice.[/yellow]")
        return

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    vectorstore = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
        collection_name=settings.domain.collection_name,
    )
    count = vectorstore._collection.count()
    results = vectorstore.get(include=["metadatas"])
    sources = {m.get("source") for m in results["metadatas"] if m}

    console.print(f"  [cyan]Documentos:[/cyan]  {len(sources)}")
    console.print(f"  [cyan]Chunks:[/cyan]      {count}")
    console.print(f"  [cyan]Modelo:[/cyan]      {settings.llm.model}")

    # Active features
    active = []
    if settings.chunking.semantic:
        active.append("Semantic Chunking")
    if settings.chunking.contextual_retrieval:
        active.append("Contextual Retrieval")
    if settings.retrieval.multi_query:
        active.append("Multi-Query")
    if settings.retrieval.self_rag:
        active.append("Self-RAG")
    if active:
        console.print(f"  [cyan]Tecnicas:[/cyan]   {', '.join(active)}")


# ============================================================================
# rag collection list|create|switch|delete|info
# ============================================================================


@collection_app.command("list")
def collection_list(
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Listar todas las colecciones disponibles."""
    settings = get_settings(project_root)
    mgr = CollectionManager(settings)
    infos = mgr.list()

    if not infos:
        console.print("[yellow]No hay colecciones.[/yellow]")
        console.print("Crea una con: [cyan]rag collection create <nombre>[/cyan]")
        return

    table = Table(title="Colecciones", show_edge=False, box=box.SIMPLE)
    table.add_column("Activa", justify="center")
    table.add_column("Nombre", style="cyan")
    table.add_column("Display", style="white")
    table.add_column("PDFs", style="green", justify="right")
    table.add_column("Indice", justify="center")
    table.add_column("Creada", style="dim")

    for c in infos:
        active_mark = "[bold green]*[/bold green]" if c.is_active else ""
        name = c.name + (" [dim](legacy)[/dim]" if c.is_legacy else "")
        idx = "[green]si[/green]" if c.has_index else "[red]no[/red]"
        created = c.created_at[:10] if c.created_at else "-"
        table.add_row(active_mark, name, c.display_name, str(c.pdf_count), idx, created)

    console.print(table)
    console.print(f"\n  [dim]Activa: {mgr.get_active()}[/dim]")
    console.print("  [dim]Cambiar: rag collection switch <nombre>[/dim]")


@collection_app.command("create")
def collection_create(
    name: str = typer.Argument(help="Nombre corto (letras, digitos, _, -)"),
    display: str = typer.Option(None, "--display", help="Nombre amigable"),
    description: str = typer.Option("", "--description", "-d", help="Descripcion"),
    activate: bool = typer.Option(
        True, "--activate/--no-activate", help="Activar la coleccion tras crearla",
    ),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Crear una nueva coleccion."""
    if not valid_collection_name(name):
        console.print("[red]Nombre invalido.[/red] Usa letras, digitos, _ o -.")
        raise typer.Exit(1)

    settings = get_settings(project_root)
    mgr = CollectionManager(settings)

    try:
        info = mgr.create(name, display_name=display, description=description)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"\n  [bold green]+[/bold green] Coleccion creada: [cyan]{info.name}[/cyan]")
    console.print(f"    Carpeta: [dim]{info.path}[/dim]")

    if activate:
        mgr.set_active(name)
        console.print("    [green]Activada.[/green]")

    console.print("\n[bold]Siguiente paso:[/bold]")
    console.print(f"  1. Copia PDFs a [cyan]{info.path / 'docs'}[/cyan]")
    console.print(f"     o usa: [cyan]rag docs add archivo.pdf --collection {name}[/cyan]")
    console.print(f"  2. Indexa:   [cyan]rag ingest --collection {name}[/cyan]")


@collection_app.command("switch")
def collection_switch(
    name: str = typer.Argument(help="Nombre de la coleccion a activar"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Cambiar la coleccion activa."""
    settings = get_settings(project_root)
    mgr = CollectionManager(settings)
    if not mgr.exists(name):
        console.print(f"[red]Coleccion '{name}' no existe.[/red]")
        console.print("Disponibles: " + ", ".join(c.name for c in mgr.list()))
        raise typer.Exit(1)

    mgr.set_active(name)
    console.print(f"[green]Coleccion activa: [cyan]{name}[/cyan][/green]")


@collection_app.command("delete")
def collection_delete(
    name: str = typer.Argument(help="Nombre de la coleccion a eliminar"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirmar sin preguntar"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Eliminar una coleccion (borra PDFs, indices y metadata)."""
    settings = get_settings(project_root)
    mgr = CollectionManager(settings)

    if not mgr.exists(name):
        console.print(f"[red]Coleccion '{name}' no existe.[/red]")
        raise typer.Exit(1)

    if not yes:
        info = mgr.get(name)
        console.print(
            f"[yellow]Se eliminara la coleccion '{name}' "
            f"({info.pdf_count} PDFs, indice: {'si' if info.has_index else 'no'}).[/yellow]"
        )
        if not typer.confirm("Continuar?"):
            console.print("[dim]Cancelado.[/dim]")
            return

    try:
        mgr.delete(name)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Coleccion eliminada: {name}[/green]")


@collection_app.command("info")
def collection_info(
    name: str = typer.Argument(None, help="Coleccion (default: activa)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Ver informacion detallada de una coleccion."""
    settings = get_settings(project_root)
    mgr = CollectionManager(settings)
    target = name or mgr.get_active()

    if not mgr.exists(target):
        console.print(f"[red]Coleccion '{target}' no existe.[/red]")
        raise typer.Exit(1)

    info = mgr.get(target)
    table = Table(show_edge=False, box=box.SIMPLE)
    table.add_column("", style="cyan")
    table.add_column("", style="white")
    table.add_row("Nombre", info.name)
    table.add_row("Display", info.display_name)
    table.add_row("Descripcion", info.description or "(sin descripcion)")
    table.add_row("Activa", "Si" if info.is_active else "No")
    table.add_row("Legacy", "Si" if info.is_legacy else "No")
    table.add_row("Ruta", str(info.path))
    table.add_row("PDFs", str(info.pdf_count))
    table.add_row("Indice completo", "Si" if info.has_index else "No")
    table.add_row("Creada", info.created_at or "-")
    console.print(table)


# ============================================================================
# rag init (onboarding wizard)
# ============================================================================


@app.command()
def init(
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
    force: bool = typer.Option(False, "--force", help="Sobreescribir config existente"),
) -> None:
    """Asistente interactivo de configuracion inicial.

    \b
    Verifica dependencias, solicita las API keys, crea config.yaml y .env,
    y prepara la primera coleccion. Ejecuta esto la primera vez que usas el
    sistema en un proyecto.
    """
    from rich.prompt import Confirm, Prompt

    root = Path(project_root).resolve()
    console.print(LOGO)
    console.print(Panel(
        "[bold]Bienvenido[/bold]\n\n"
        "Este asistente configurara tu RAG en 1 minuto.\n"
        "Puedes cancelar en cualquier momento con [cyan]Ctrl+C[/cyan] "
        "y volver a ejecutar [cyan]rag init[/cyan].",
        border_style="blue",
        padding=(1, 2),
    ))

    config_file = root / "config.yaml"
    env_file = root / ".env"

    # --- Tesseract dep check ---
    import shutil as _shutil
    tess = _shutil.which("tesseract")
    if not tess:
        console.print("\n[yellow]![/yellow] Tesseract OCR no encontrado.")
        console.print("  [dim]Instalalo con: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa[/dim]")
        console.print("  [dim](No bloquea la configuracion — pero OCR de PDFs escaneados no funcionara.)[/dim]")
    else:
        console.print(f"\n[green]✓[/green] Tesseract: [dim]{tess}[/dim]")

    # --- Existing config detection ---
    overwrite_config = force
    if config_file.exists() and not force:
        console.print(f"\n[yellow]!Existe {config_file}[/yellow]")
        overwrite_config = Confirm.ask("  Sobreescribir config.yaml?", default=False)
    overwrite_env = force
    if env_file.exists() and not force:
        console.print(f"[yellow]!Existe {env_file}[/yellow]")
        overwrite_env = Confirm.ask("  Sobreescribir .env?", default=False)

    # --- Domain & collection ---
    console.print("\n[bold]Dominio del corpus[/bold]")
    domain_name = Prompt.ask(
        "  Nombre del dominio",
        default="Documents",
    )
    collection_slug = _slugify(domain_name) or "rag_docs"
    collection_slug = Prompt.ask(
        "  Nombre de la coleccion (ChromaDB)",
        default=collection_slug,
    )
    grader_description = Prompt.ask(
        "  Descripcion para el grader Self-RAG",
        default=f"a {domain_name.lower()} research RAG system",
    )

    # --- Example queries ---
    console.print("\n[bold]Preguntas de ejemplo[/bold] [dim](vacio para terminar)[/dim]")
    examples: list[str] = []
    for i in range(1, 6):
        q = Prompt.ask(f"  Ejemplo {i}", default="")
        if not q:
            break
        examples.append(q)
    if not examples:
        examples = [
            "What are the main findings of these documents?",
            "Summarize the key methodology used.",
        ]

    # --- API keys ---
    console.print("\n[bold]API Keys[/bold]")
    api_keys: dict[str, str] = {}
    openai_key = Prompt.ask(
        "  OPENAI_API_KEY [dim](requerida)[/dim]",
        default="", password=True, show_default=False,
    )
    if openai_key:
        api_keys["OPENAI_API_KEY"] = openai_key
    cohere_key = Prompt.ask(
        "  COHERE_API_KEY [dim](para reranking)[/dim]",
        default="", password=True, show_default=False,
    )
    if cohere_key:
        api_keys["COHERE_API_KEY"] = cohere_key
    openrouter_key = Prompt.ask(
        "  OPENROUTER_API_KEY [dim](opcional, si usas OpenRouter)[/dim]",
        default="", password=True, show_default=False,
    )
    if openrouter_key:
        api_keys["OPENROUTER_API_KEY"] = openrouter_key

    # --- LLM model ---
    console.print("\n[bold]Modelo[/bold]")
    use_openrouter = bool(openrouter_key) and Confirm.ask(
        "  Usar OpenRouter para la generacion?", default=True,
    )
    if use_openrouter:
        llm_model = Prompt.ask("  Modelo", default="anthropic/claude-sonnet-4")
        llm_base_url = "https://openrouter.ai/api/v1"
    else:
        llm_model = Prompt.ask("  Modelo", default="gpt-4o-mini")
        llm_base_url = None

    # --- Write config.yaml ---
    if overwrite_config or not config_file.exists():
        import yaml as _yaml
        config_data = {
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "semantic": False,
                "similarity_threshold": 0.82,
                "contextual_retrieval": False,
                "context_workers": 8,
            },
            "retrieval": {
                "dense_k": 20,
                "bm25_k": 20,
                "rerank_top_k": 5,
                "multi_query": True,
                "self_rag": False,
                "self_rag_max_retries": 2,
            },
            "llm": {
                "model": llm_model,
                "temperature": 0.1,
                "embedding_model": "text-embedding-3-small",
                "base_url": llm_base_url,
            },
            "evaluation": {
                "test_set_size": 30,
                "eval_model": "gpt-4o-mini",
            },
            "domain": {
                "name": domain_name,
                "collection_name": collection_slug,
                "grader_description": grader_description,
                "example_queries": examples,
            },
        }
        with open(config_file, "w") as f:
            _yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        console.print("\n[green]✓[/green] config.yaml creado")

    # --- Write .env ---
    if api_keys and (overwrite_env or not env_file.exists()):
        existing: dict[str, str] = {}
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()
        existing.update(api_keys)
        with open(env_file, "w") as f:
            for k, v in existing.items():
                f.write(f"{k}={v}\n")
        env_file.chmod(0o600)
        console.print("[green]✓[/green] .env creado (permisos 600)")

    # --- Create first collection if none ---
    settings = get_settings(project_root)
    get_settings.cache_clear()
    settings = get_settings(project_root)
    mgr = CollectionManager(settings)
    if not mgr.list():
        default_info = mgr.create("default", display_name=domain_name,
                                  description=f"Default {domain_name} collection")
        mgr.set_active("default")
        console.print(f"[green]✓[/green] Coleccion 'default' creada en {default_info.path}")

    # --- Next steps ---
    console.print("\n[bold]Listo.[/bold] Siguiente paso:")
    active = mgr.get_active()
    docs_path = settings._collection_dir(active) / "docs" if not mgr.get(active).is_legacy else settings.docs_path
    console.print(f"  1. Copia PDFs a [cyan]{docs_path}[/cyan]")
    console.print("     o usa: [cyan]rag docs add archivo.pdf[/cyan]")
    console.print("  2. Indexa:     [cyan]rag ingest[/cyan]")
    console.print("  3. Consulta:   [cyan]rag chat[/cyan]  (CLI interactivo)")
    console.print("                 [cyan]rag serve[/cyan] (UI web en http://localhost:8765)")


def _slugify(text: str) -> str:
    """Turn a display name into a lowercase filesystem-safe slug."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("_")
    slug = "".join(out).strip("_")
    # Collapse repeated underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


# ============================================================================
# rag serve (FastAPI + web UI)
# ============================================================================


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host de escucha"),
    port: int = typer.Option(8765, "--port", "-p", help="Puerto"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (desarrollo)"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Iniciar el servidor FastAPI + UI web.

    \b
    Expone:
      - Web UI en http://host:port/
      - API REST en /api/*
      - SSE streaming en POST /api/query

    \b
    La UI web permite cambiar entre colecciones, ver fuentes con
    citas numericas [1][2], y el dashboard de metricas.

    \b
    Ejemplos:
      rag serve                     # localhost:8765
      rag serve --host 0.0.0.0      # accesible desde LAN
      rag serve --port 8000
    """
    settings = get_settings(project_root)
    if not _require_ready(settings, need_index=False):
        return

    try:
        import uvicorn
    except ImportError as e:
        console.print("[red]Falta uvicorn. Instala: [cyan]pip install 'uvicorn[standard]'[/cyan][/red]")
        raise typer.Exit(1) from e

    os.environ["RAG_PROJECT_ROOT"] = str(Path(project_root).resolve())

    console.print(LOGO)
    console.print("[bold]Servidor RAG[/bold]")
    console.print(f"  Web UI:  [cyan]http://{host}:{port}/[/cyan]")
    console.print(f"  API:     [cyan]http://{host}:{port}/api[/cyan]")
    console.print(f"  Docs:    [cyan]http://{host}:{port}/docs[/cyan]")
    console.print()
    console.print("[dim]Ctrl+C para detener.[/dim]\n")

    uvicorn.run(
        "rag.server_factory:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ============================================================================
# rag stats (metrics dashboard in terminal)
# ============================================================================


@app.command()
def stats(
    collection: str = typer.Option(None, "--collection", "-c", help="Filtrar por coleccion"),
    limit: int = typer.Option(10, "--limit", "-n", help="Ultimas N consultas a mostrar"),
    project_root: str = typer.Option(".", "--root", "-r", help="Directorio raiz"),
) -> None:
    """Ver metricas de uso: latencia, volumen, errores."""
    settings = get_settings(project_root)
    from rag.metrics import MetricsStore

    store = MetricsStore(settings.metrics_db_path)
    data = store.summary(limit=limit, collection=collection)

    # Overview
    ov = Table(title="Metricas", show_edge=False, box=box.SIMPLE)
    ov.add_column("", style="cyan")
    ov.add_column("", style="green", justify="right")
    ov.add_row("Total de consultas", str(data["total"]))
    ov.add_row("Errores", str(data["errors"]))
    ov.add_row("Latencia media", f"{data['avg_latency_ms']} ms")
    ov.add_row("Latencia p50", f"{data['p50_latency_ms']} ms")
    ov.add_row("Latencia p95", f"{data['p95_latency_ms']} ms")
    console.print(ov)

    # Per-collection
    if data["per_collection"] and not collection:
        console.print()
        pc = Table(title="Por coleccion", show_edge=False, box=box.SIMPLE)
        pc.add_column("Coleccion", style="cyan")
        pc.add_column("Consultas", justify="right")
        pc.add_column("Latencia media", justify="right")
        for row in data["per_collection"]:
            pc.add_row(row["collection"], str(row["count"]), f"{row['avg_ms']} ms")
        console.print(pc)

    # Recent
    if data["recent"]:
        console.print()
        rc = Table(title=f"Ultimas {len(data['recent'])}", show_edge=False, box=box.SIMPLE)
        rc.add_column("#", style="dim", justify="right")
        rc.add_column("Coleccion", style="cyan")
        rc.add_column("Pregunta", style="white")
        rc.add_column("Latencia", justify="right")
        rc.add_column("Docs", justify="right")
        for i, r in enumerate(data["recent"], 1):
            q = r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"]
            err = " [red](err)[/red]" if r["error"] else ""
            rc.add_row(
                str(i), r["collection"], q + err,
                f"{r['latency_ms']} ms", str(r["doc_count"]),
            )
        console.print(rc)
    else:
        console.print("\n[dim]Sin consultas registradas todavia.[/dim]")


if __name__ == "__main__":
    app()
