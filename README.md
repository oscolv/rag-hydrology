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

## Comandos

### `rag ingest` — Indexar documentos

```bash
rag ingest              # Indexar todos los PDFs en ./docs/
rag ingest --force      # Re-indexar desde cero
rag ingest --docs-dir /ruta/a/pdfs
```

### `rag query` — Consulta unica

```bash
rag query "What is GRACE and how does it measure terrestrial water storage?"
rag query "Que informacion contiene el atlas del agua?" -v
```

`-v` muestra las fuentes con preview del contenido.

### `rag chat` — Sesion interactiva

```bash
rag chat            # Inicia sesion de chat
rag chat -v         # Con fuentes siempre visibles
```

Dentro de la sesion:

| Comando | Accion |
|---|---|
| `/sources` | Ver fuentes de la ultima respuesta (vista detallada) |
| `/history` | Ver historial de la conversacion |
| `/export [path]` | Exportar sesion a Markdown |
| `/clear` | Limpiar historial |
| `/verbose` | Alternar modo verbose |
| `/help` | Ayuda de comandos |
| `/quit` | Salir |

### `rag search` — Busqueda sin LLM

Retrieval puro (sin generacion), util para depurar relevancia:

```bash
rag search "GRACE terrestrial water storage" -k 10
```

### `rag export` — Consulta con exportacion

```bash
rag export "Explain GRACE-FO mission" -o report.md
rag export "TWS estimation methods" -f json -o results.json
```

Formatos: `markdown` (default) y `json`.

### `rag docs` — Gestion de documentos

```bash
rag docs list                        # Listar PDFs y estado de indexacion
rag docs add paper1.pdf paper2.pdf   # Copiar PDFs a docs/
rag docs remove paper_viejo.pdf      # Eliminar PDF y purgar del indice
```

### `rag config` — Configuracion

```bash
rag config show                                # Ver configuracion actual
rag config set llm.model gpt-4o-mini           # Cambiar modelo
rag config set chunking.chunk_size 1500        # Cambiar tamano de chunk
rag config set retrieval.multi_query false      # Desactivar multi-query
```

### `rag evaluate` — Evaluacion RAGAS

Mide objetivamente que tan bueno es el sistema RAG. Sin evaluacion solo sabes
"parece que funciona"; con RAGAS tienes numeros concretos.

```bash
rag evaluate --generate     # Generar test set sintetico + evaluar
rag evaluate                # Evaluar con test set existente
rag evaluate --testset custom_tests.csv
```

**`--generate`** ejecuta dos pasos:

1. **Genera preguntas sinteticas**: RAGAS lee los chunks indexados y usa el LLM
   (`gpt-4o-mini`) para crear ~30 pares de pregunta + respuesta de referencia
   basados en el contenido real de tus papers. Se guardan en `data/testset.csv`.

2. **Evalua el pipeline**: Pasa cada pregunta por el pipeline completo
   (hybrid search -> rerank -> GPT-4o), compara contra la referencia y calcula
   4 metricas.

**Sin `--generate`** solo ejecuta el paso 2 con un test set existente.

#### Metricas

| Metrica | Que mide | Si el score es bajo... |
|---|---|---|
| **Faithfulness** | La respuesta solo usa informacion de los documentos (detecta alucinaciones) | Bajar `temperature`, mejorar el prompt |
| **Response Relevancy** | La respuesta contesta lo que se pregunto | Mejorar el prompt del sistema |
| **Context Precision** | Los documentos relevantes quedaron arriba en el ranking | Subir `rerank_top_k`, ajustar chunking |
| **Context Recall** | Se recuperaron todos los documentos necesarios | Subir `dense_k`/`bm25_k`, reducir `chunk_size` |

Ejemplo de interpretacion:
- **Faithfulness 0.95** → solo 5% de respuestas tienen info que no viene de los docs
- **Context Recall 0.60** → el retriever se pierde 40% de los documentos relevantes

Los resultados detallados (por pregunta) se guardan en `data/eval_results.csv`.

#### Configuracion de evaluacion

La evaluacion usa `gpt-4o-mini` por defecto (mas barato, limites de TPM mas altos)
para evitar errores de rate limit. La generacion de respuestas (`query`/`chat`)
sigue usando `gpt-4o`.

```bash
rag config set evaluation.eval_model gpt-4o-mini   # modelo para evaluacion
rag config set evaluation.test_set_size 20          # menos preguntas = mas rapido
```

### `rag status` — Diagnostico del sistema

```bash
rag status
```

Verifica API keys, documentos, indices y Tesseract. Muestra problemas con
soluciones especificas y sugiere el siguiente paso.

### `rag info` — Estadisticas del indice

```bash
rag info
```

Muestra inventario de documentos (chunks, paginas, ano, idioma), tamano de indices y configuracion.

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
  eval_model: "gpt-4o-mini"  # Modelo para evaluacion (mas barato, evita rate limits)
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

## Evaluacion del sistema

La evaluacion con RAGAS es la forma estandar de medir objetivamente la calidad
de un sistema RAG. Permite tomar decisiones informadas sobre que ajustar
en lugar de adivinar.

### Flujo de evaluacion

```
Chunks indexados
      |
      v
TestsetGenerator (gpt-4o-mini)
      |
      v
data/testset.csv (30 preguntas sinteticas + respuestas de referencia)
      |
      v
Pipeline RAG completo (por cada pregunta)
      |
      v
RAGAS Metrics (faithfulness, relevancy, precision, recall)
      |
      v
data/eval_results.csv (scores por pregunta)
```

### Metricas

| Metrica | Que mide | Score bajo indica... | Que ajustar |
|---|---|---|---|
| **Faithfulness** | No alucina: usa solo info de los docs | El LLM inventa informacion | `llm.temperature`, prompt |
| **Response Relevancy** | Contesta lo que se pregunto | Respuestas fuera de tema | Prompt del sistema |
| **Context Precision** | Docs relevantes arriba en el ranking | El reranker no prioriza bien | `rerank_top_k`, chunking |
| **Context Recall** | Recupero todos los docs necesarios | El retriever pierde informacion | `dense_k`, `bm25_k`, `chunk_size` |

### Interpretacion de resultados

- **>= 0.8**: Buen desempeno
- **0.5 - 0.8**: Aceptable, hay espacio para mejorar
- **< 0.5**: Necesita atencion, ajustar parametros

### Ciclo de mejora

```bash
rag evaluate --generate          # Medir estado actual
rag config set chunking.chunk_size 800   # Ajustar parametro
rag ingest --force               # Re-indexar
rag evaluate                     # Medir de nuevo con el mismo test set
```

Comparar los scores antes y despues de cada cambio permite optimizar
el sistema de forma sistematica.
