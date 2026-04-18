"""Module-level app instance so `uvicorn rag.server_factory:app` works.

The `rag serve` CLI command sets RAG_PROJECT_ROOT before launching uvicorn,
and we honor it here. This indirection is needed because uvicorn --reload
requires the app to be importable via an ASGI string.
"""

import os

from rag.server import create_app

app = create_app(os.environ.get("RAG_PROJECT_ROOT", "."))
