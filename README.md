# RAG Hydrology

Sistema RAG (Retrieval-Augmented Generation) de estado del arte para consultar papers de investigacion sobre recursos hidricos, datos satelitales GRACE e hidrologia.

## Arquitectura

```
Query del usuario
       |
       v
  Multi-Query Expansion (3 reformulaciones via LLM)
       |
       v
  +----+----+
  |         |
  v         v
Dense    Sparse
(ChromaDB) (BM25)
  |         |
  +----+----+
       |
       v
  Reciprocal Rank Fusion (RRF)
       |
       v
  Cohere Reranker (top 5)
       |
       v
  GPT-4o con citas [Source: file, Page: N]
```

### Modulos

| Modulo | Archivo | Descripcion |
|---|---|---|
| Configuracion | `src/rag/config.py` | Pydantic Settings + YAML. Centraliza parametros de chunking, retrieval, LLM y evaluacion |
| Ingestion | `src/rag/ingest.py` | Parsing PDF con pymupdf4llm, deduplicacion MD5, chunking con contextual headers, indexacion dual ChromaDB + BM25 |
| Retrieval | `src/rag/retrieval.py` | HybridRetriever (dense + sparse con RRF), multi-query expansion, reranking Cohere |
| Generacion | `src/rag/generation.py` | Cadena RAG con soporte de citas y variante con fuentes para evaluacion |
| Evaluacion | `src/rag/evaluation.py` | Generacion de test set sintetico y metricas RAGAS (faithfulness, relevancy, context precision/recall) |
| CLI | `src/rag/cli.py` | Interfaz de linea de comandos con Typer |

## Requisitos

### Sistema

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
```

### Python >= 3.11

```bash
pip install -e ".[dev]"
```

### API Keys

Crea un archivo `.env` en la raiz del proyecto (usa `.env.example` como plantilla):

```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

- **OpenAI**: Embeddings (`text-embedding-3-small`) y generacion (`gpt-4o`)
- **Cohere**: Reranker (`rerank-v3.5`). Tier gratuito: 1000 llamadas/mes

## Uso

### 1. Ingestar documentos

Coloca los PDFs en `./docs/` y ejecuta:

```bash
rag ingest
```

Esto parsea los PDFs, genera chunks con contextual headers, crea embeddings en ChromaDB y construye el indice BM25.

Opciones:
- `--force` / `-f`: Re-ingesta todos los documentos (borra indices existentes)
- `--docs-dir PATH`: Directorio alternativo de PDFs

### 2. Consultar

```bash
rag query "What is GRACE and how does it measure terrestrial water storage?"
```

```bash
rag query "Que informacion contiene el atlas del agua?" -v
```

Opciones:
- `--verbose` / `-v`: Muestra documentos fuente con preview del contenido

### 3. Evaluar

Genera un test set sintetico y ejecuta metricas RAGAS:

```bash
rag evaluate --generate
```

Para evaluar con un test set existente:

```bash
rag evaluate --testset data/testset.csv
```

### 4. Informacion del indice

```bash
rag info
```

Muestra estadisticas del indice (total de chunks, documentos indexados, configuracion actual).

## Configuracion

Los parametros se configuran en `config.yaml`:

```yaml
chunking:
  chunk_size: 1000        # Caracteres por chunk
  chunk_overlap: 200      # Solapamiento entre chunks

retrieval:
  dense_k: 20             # Candidatos de busqueda densa
  bm25_k: 20              # Candidatos de busqueda sparse
  rerank_top_k: 5         # Documentos finales tras reranking
  multi_query: true       # Expansion multi-query activada

llm:
  model: "gpt-4o"
  temperature: 0.1
  embedding_model: "text-embedding-3-small"

evaluation:
  test_set_size: 30       # Preguntas en el test set sintetico
```

## Tecnicas implementadas

| Tecnica | Implementacion | Proposito |
|---|---|---|
| Parsing PDF cientifico | pymupdf4llm | Preserva tablas, columnas y estructura de papers academicos |
| Contextual headers | Prepend `[From: titulo \| Page: N \| Section: X]` | Mejora retrieval al anclar cada chunk a su documento/seccion de origen |
| Busqueda hibrida | Dense (ChromaDB) + Sparse (BM25) con RRF | Combina similitud semantica con coincidencia lexica exacta |
| Multi-query expansion | LLM genera 3 reformulaciones del query | Captura diferentes aspectos y terminologia del mismo concepto |
| Reranking | Cohere rerank-v3.5 | Reordena candidatos con un modelo cross-encoder mas preciso |
| Deduplicacion | Hash MD5 por archivo | Detecta y omite PDFs duplicados durante ingestion |
| Evaluacion automatizada | RAGAS 0.4 | Mide faithfulness, relevancy, context precision y recall |

## Estructura del proyecto

```
rag-hydrology/
├── pyproject.toml          # Dependencias y metadata del proyecto
├── config.yaml             # Parametros tunables
├── .env                    # API keys (no versionado)
├── .env.example            # Plantilla de API keys
├── docs/                   # PDFs fuente
├── data/
│   ├── chroma/             # Vector store persistido
│   ├── bm25_index.pkl      # Indice BM25 serializado
│   ├── testset.csv         # Test set sintetico (generado)
│   └── eval_results.csv    # Resultados de evaluacion (generado)
├── src/rag/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   └── cli.py
└── tests/
    ├── test_ingest.py
    ├── test_retrieval.py
    └── test_generation.py
```

## Tests

```bash
pytest tests/ -v
```

## Metricas de evaluacion

| Metrica | Que mide |
|---|---|
| **Faithfulness** | La respuesta es fiel al contexto recuperado (no alucina) |
| **Response Relevancy** | La respuesta es relevante a la pregunta |
| **Context Precision** | Los documentos recuperados relevantes estan en las posiciones mas altas |
| **Context Recall** | Se recuperaron todos los documentos necesarios para responder |
