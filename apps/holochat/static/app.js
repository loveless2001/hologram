const state = {
  history: [],
  citations: [],
};

const elements = {
  project: document.getElementById("project-input"),
  session: document.getElementById("session-input"),
  activeDoc: document.getElementById("active-doc-input"),
  selectedDocs: document.getElementById("selected-docs-input"),
  topK: document.getElementById("top-k-input"),
  message: document.getElementById("message-input"),
  routePanel: document.getElementById("route-panel"),
  status: document.getElementById("status-line"),
  history: document.getElementById("history-list"),
  citations: document.getElementById("citations-list"),
  routeButton: document.getElementById("route-button"),
  sendButton: document.getElementById("send-button"),
};

function ensureSessionId() {
  if (!elements.session.value.trim()) {
    elements.session.value = `session-${Math.random().toString(36).slice(2, 10)}`;
  }
  return elements.session.value.trim();
}

function parseSelectedDocs() {
  return elements.selectedDocs.value
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

function buildContext() {
  return {
    user_message: elements.message.value.trim(),
    active_document_id: elements.activeDoc.value.trim() || null,
    selected_document_ids: parseSelectedDocs(),
    recent_messages: state.history.slice(-6).map((item) => item.content),
    session_summary: null,
    just_uploaded_files: false,
    available_actions: [],
    corpus_search_enabled: true,
  };
}

function buildChatPayload() {
  return {
    project: elements.project.value.trim() || "demo",
    session_id: ensureSessionId(),
    context: buildContext(),
    top_k: Number(elements.topK.value || 5),
  };
}

function buildRoutePayload() {
  return {
    project: elements.project.value.trim() || "demo",
    session_id: ensureSessionId(),
    context: buildContext(),
  };
}

function renderRoute(route) {
  if (!route) {
    elements.routePanel.className = "route-panel empty";
    elements.routePanel.textContent = "No decision yet.";
    return;
  }

  elements.routePanel.className = "route-panel";
  elements.routePanel.innerHTML = `
    <div class="route-pill-row">
      <span class="route-pill">${route.intent}</span>
      <span class="route-pill">${route.response_mode}</span>
      <span class="route-pill">${route.retrieval_scope}</span>
    </div>
    <p class="route-intent">
      confidence ${Number(route.confidence).toFixed(2)}
    </p>
    <p class="route-meta">
      retrieval=${route.needs_retrieval} rerank=${route.needs_rerank}
      history=${route.needs_chat_history}
      ${route.action_name ? ` action=${route.action_name}` : ""}
    </p>
  `;
}

function renderHistory() {
  if (!state.history.length) {
    elements.history.innerHTML = '<p class="empty-state">Messages will appear here after you preview or send.</p>';
    return;
  }

  elements.history.innerHTML = state.history
    .map(
      (entry) => `
        <article class="message-card ${entry.role}">
          <div class="message-meta">${entry.role}</div>
          <p class="message-body">${escapeHtml(entry.content)}</p>
        </article>
      `
    )
    .join("");
}

function renderCitations() {
  if (!state.citations.length) {
    elements.citations.innerHTML = '<p class="empty-state">Grounded traces will appear here.</p>';
    return;
  }

  elements.citations.innerHTML = state.citations
    .map(
      (citation) => `
        <article class="citation-card">
          <div class="citation-meta">
            ${escapeHtml(citation.source_doc || "session")}
            ${citation.page_number ? ` · page ${citation.page_number}` : ""}
            · score ${Number(citation.score).toFixed(3)}
          </div>
          <p class="citation-body">${escapeHtml(citation.content)}</p>
        </article>
      `
    )
    .join("");
}

function setBusy(isBusy, message) {
  elements.routeButton.disabled = isBusy;
  elements.sendButton.disabled = isBusy;
  elements.status.textContent = message;
}

async function previewRoute() {
  const userMessage = elements.message.value.trim();
  if (!userMessage) {
    elements.status.textContent = "Enter a message before previewing the route.";
    return;
  }

  setBusy(true, "Routing request...");
  try {
    const response = await fetch("/chat/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRoutePayload()),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Route request failed.");
    }
    renderRoute(data.route);
    elements.status.textContent = `Route ready: ${data.route.intent}`;
  } catch (error) {
    elements.status.textContent = error.message;
  } finally {
    setBusy(false, elements.status.textContent);
  }
}

async function sendMessage() {
  const userMessage = elements.message.value.trim();
  if (!userMessage) {
    elements.status.textContent = "Enter a message before sending.";
    return;
  }

  const outbound = buildChatPayload();
  setBusy(true, "Sending request...");
  try {
    const response = await fetch("/chat/respond", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(outbound),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Chat request failed.");
    }

    state.history.push({ role: "user", content: outbound.context.user_message });
    state.history.push({ role: "assistant", content: data.reply });
    state.citations = data.citations || [];

    renderRoute(data.route);
    renderHistory();
    renderCitations();

    elements.message.value = "";
    elements.status.textContent = `Responded via ${data.route.intent}.`;
  } catch (error) {
    elements.status.textContent = error.message;
  } finally {
    setBusy(false, elements.status.textContent);
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

elements.routeButton.addEventListener("click", previewRoute);
elements.sendButton.addEventListener("click", sendMessage);
elements.message.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    sendMessage();
  }
});

renderRoute(null);
renderHistory();
renderCitations();
