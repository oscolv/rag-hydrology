/* global Alpine, marked, DOMPurify */

function ragApp() {
  return {
    tab: "chat",
    collections: [],
    activeCollection: null,
    collectionsOpen: false,
    showCreateCollection: false,
    newCollection: { name: "", display_name: "", description: "" },
    messages: [],
    question: "",
    streaming: false,
    stats: {},
    health: { model: "", provider: "", embedding_model: "" },
    examples: [
      "¿Cuáles son los hallazgos principales de estos documentos?",
      "Resume la metodología clave utilizada.",
      "¿Qué limitaciones reportan los autores?",
      "Lista las fuentes de datos y su resolución espacial/temporal.",
    ],
    toast: "",

    async init() {
      await Promise.all([this.loadCollections(), this.loadHealth()]);
    },

    async loadHealth() {
      try {
        const r = await fetch("/api/health");
        if (r.ok) this.health = await r.json();
      } catch (_e) { /* non-fatal */ }
    },

    async loadCollections() {
      try {
        const r = await fetch("/api/collections");
        const data = await r.json();
        this.collections = data.collections;
        this.activeCollection = this.collections.find((c) => c.is_active) || this.collections[0];
      } catch (e) {
        this.showToast("No se pudo cargar colecciones: " + e.message);
      }
    },

    async switchCollection(name) {
      try {
        await fetch(`/api/collections/${encodeURIComponent(name)}/activate`, { method: "POST" });
        await this.loadCollections();
        this.collectionsOpen = false;
        this.showToast(`Coleccion activa: ${name}`);
      } catch (e) {
        this.showToast("Error: " + e.message);
      }
    },

    async createCollection() {
      try {
        const r = await fetch("/api/collections", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.newCollection),
        });
        if (!r.ok) {
          const err = await r.json();
          throw new Error(err.detail || "Error al crear");
        }
        const name = this.newCollection.name;
        await fetch(`/api/collections/${encodeURIComponent(name)}/activate`, { method: "POST" });
        this.showCreateCollection = false;
        this.newCollection = { name: "", display_name: "", description: "" };
        await this.loadCollections();
        this.showToast(`Coleccion '${name}' creada y activada`);
      } catch (e) {
        this.showToast(e.message);
      }
    },

    async submit() {
      const q = this.question.trim();
      if (!q || this.streaming) return;

      const now = Date.now();
      this.messages.push({
        question: q,
        answer: "",
        documents: [],
        reflection: [],
        citations: [],
        request_id: "",
        feedback: 0,           // 0 = none, 1 = up, -1 = down
        feedbackComment: "",
        feedbackCommentOpen: false,
        feedbackSending: false,
        model: this.health.model || "",
        streaming: true,
        status: "Conectando con el servidor...",
        phase: "connecting",
        startedAt: now,
        lastEventAt: now,
        elapsed: 0,
        sinceLastEvent: 0,
        stallHint: "",
      });
      // Grab the REACTIVE proxy — Alpine wraps on push, the raw object we just
      // passed in is not tracked, so mutations through `msg` below must go
      // through this reference to trigger re-renders.
      const msg = this.messages[this.messages.length - 1];
      this.question = "";
      this.streaming = true;
      this.scrollToBottom();

      // Heartbeat: refresh elapsed time + show stall hints every 500ms so the
      // user knows the browser is alive even when the LLM goes silent for 30s+.
      const timer = setInterval(() => {
        if (!msg.streaming) {
          clearInterval(timer);
          return;
        }
        const t = Date.now();
        msg.elapsed = Math.floor((t - msg.startedAt) / 1000);
        msg.sinceLastEvent = Math.floor((t - msg.lastEventAt) / 1000);
        msg.stallHint = this.stallHintFor(msg);
      }, 500);

      const markEvent = (phase, status) => {
        msg.lastEventAt = Date.now();
        if (phase) msg.phase = phase;
        if (status !== undefined) msg.status = status;
      };

      try {
        await this.streamQuery(q, (event, data) => {
          if (event === "retrieval_start") {
            markEvent("retrieval", "Buscando documentos en el indice...");
          } else if (event === "sources") {
            msg.documents = data.documents;
            markEvent(
              "generating",
              `Esperando respuesta del modelo (${data.documents.length} fuentes enviadas)...`,
            );
          } else if (event === "reflection") {
            msg.reflection.push(data.step);
            markEvent("grading", this.formatReflectionStep(data.step));
          } else if (event === "token") {
            if (data.regenerated && msg.answer) {
              msg.answer = data.content;
            } else {
              msg.answer += data.content;
            }
            markEvent("streaming");
          } else if (event === "regenerating") {
            msg.answer = "";
            markEvent("generating", "Regenerando respuesta...");
          } else if (event === "done") {
            msg.answer = data.answer;
            msg.documents = data.documents;
            msg.reflection = data.reflection || msg.reflection;
            msg.citations = data.citations || [];
            msg.request_id = data.request_id || msg.request_id;
            msg.streaming = false;
          } else if (event === "error") {
            msg.answer += `\n\n**Error:** ${data.message}`;
            msg.streaming = false;
          }
          this.scrollToBottom();
        });
      } catch (e) {
        msg.answer += `\n\n**Error:** ${e.message}`;
        msg.streaming = false;
      } finally {
        clearInterval(timer);
        msg.streaming = false;
        this.streaming = false;
      }
    },

    stallHintFor(msg) {
      // Only show a hint after a real silent gap (>8s without events).
      const gap = msg.sinceLastEvent;
      if (gap < 8) return "";
      if (msg.phase === "connecting") {
        return "El servidor esta tardando en responder. Revisa la terminal donde corre `rag serve`.";
      }
      if (msg.phase === "retrieval" || msg.phase === "grading") {
        return "La recuperacion esta tardando. Si usas `multi_query: true`, se hacen varias busquedas.";
      }
      if (msg.phase === "generating") {
        return "El modelo esta 'pensando'. Los modelos de reasoning (p. ej. trinity-thinking) pueden tardar 30-60s antes de emitir el primer token.";
      }
      if (msg.phase === "streaming" && gap > 15) {
        return "El streaming se pauso. Puede ser una llamada de verificacion (Self-RAG) o un hiccup de red.";
      }
      return "";
    },

    async streamQuery(question, onEvent) {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`${response.status}: ${text}`);
      }
      if (!response.body) {
        throw new Error("Streaming no soportado por este navegador (response.body es null)");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const raw of parts) {
          if (!raw.trim()) continue;
          let ev = "message";
          let data = "";
          for (const line of raw.split("\n")) {
            if (line.startsWith("event: ")) ev = line.slice(7).trim();
            else if (line.startsWith("data: ")) data += line.slice(6);
          }
          if (data) {
            try {
              onEvent(ev, JSON.parse(data));
            } catch (e) {
              console.warn("Bad SSE payload", e, data);
            }
          }
        }
      }
    },

    renderAnswer(text, documents, msgIdx) {
      if (!text) return "";
      const mi = typeof msgIdx === "number" ? msgIdx : this.messages.length - 1;
      // Turn [N] into clickable anchors to the corresponding source card.
      const withLinks = text.replace(/\[(\d{1,3})\]/g, (m, n) => {
        const idx = parseInt(n, 10);
        if (!documents || idx < 1 || idx > documents.length) return m;
        return `<a href="#source-${mi}-${idx}" class="citation" data-n="${idx}">[${String(idx).padStart(2, "0")}]</a>`;
      });
      const html = marked.parse(withLinks, { breaks: true });
      return DOMPurify.sanitize(html, { ADD_ATTR: ["data-n"] });
    },

    autogrow(ev) {
      const el = ev.target;
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 180) + "px";
    },

    stripHeader(content) {
      // Strip the "[From: ...]" header line we add in build_chunks_*
      if (content.startsWith("[From:")) {
        const idx = content.indexOf("]\n");
        if (idx > -1) return content.slice(idx + 2);
      }
      return content;
    },

    formatReflectionStep(step) {
      switch (step.step) {
        case "grade_documents":
          return `Grading ${step.attempt}: ${step.relevant}/${step.retrieved} relevantes (${Math.round(step.ratio * 100)}%)`;
        case "reformulate":
          return `Reformulado: "${step.reformulated}"`;
        case "generate":
          return `Generando con ${step.context_docs} documentos`;
        case "hallucination_check":
          return `Check: grounded=${step.grounded}, relevant=${step.relevant}`;
        case "hallucination_recheck":
          return `Recheck: grounded=${step.grounded}, relevant=${step.relevant}`;
        case "regenerate":
          return `Regenerando: ${step.reason}`;
        default:
          return step.step || "step";
      }
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const el = this.$refs.messages;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    async sendFeedback(msg, rating, comment) {
      if (msg.feedbackSending) return;
      msg.feedbackSending = true;
      try {
        const r = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rating,
            request_id: msg.request_id || null,
            comment: comment || null,
            question: msg.question,
            answer: msg.answer,
            collection: this.activeCollection?.name || null,
          }),
        });
        if (!r.ok) {
          const txt = await r.text();
          throw new Error(`${r.status}: ${txt}`);
        }
        msg.feedback = rating;
        msg.feedbackCommentOpen = false;
        this.showToast(rating > 0 ? "Gracias por el feedback" : "Anotado: lo veremos");
      } catch (e) {
        this.showToast("Error: " + e.message);
      } finally {
        msg.feedbackSending = false;
      }
    },

    async loadStats() {
      try {
        const r = await fetch("/api/stats?limit=20");
        this.stats = await r.json();
      } catch (e) {
        this.showToast("Error al cargar stats: " + e.message);
      }
    },

    formatRelativeTime(ts) {
      const seconds = Math.max(0, Math.floor(Date.now() / 1000 - ts));
      if (seconds < 60) return `hace ${seconds}s`;
      if (seconds < 3600) return `hace ${Math.floor(seconds / 60)}m`;
      if (seconds < 86400) return `hace ${Math.floor(seconds / 3600)}h`;
      return `hace ${Math.floor(seconds / 86400)}d`;
    },

    showToast(text) {
      this.toast = text;
      setTimeout(() => { this.toast = ""; }, 2800);
    },
  };
}

window.ragApp = ragApp;
