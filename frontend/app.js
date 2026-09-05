document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const queryInput = document.getElementById('queryInput');
  const sendBtn = document.getElementById('sendBtn');
  const ragFusionToggle = document.getElementById('ragFusionToggle');
  const chatThread = document.getElementById('chatThread');
  const suggestionsCard = document.getElementById('suggestionsCard');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const poolBadge = document.getElementById('poolBadge');
  const modelBadge = document.getElementById('modelBadge');
  
  // Modals
  const resumeModal = document.getElementById('resumeModal');
  const closeResumeModal = document.getElementById('closeResumeModal');
  const doneResumeBtn = document.getElementById('doneResumeBtn');
  const copyResumeBtn = document.getElementById('copyResumeBtn');
  const resumeContent = document.getElementById('resumeContent');
  const resumeModalTitle = document.getElementById('resumeModalTitle');
  const resumeModalBadge = document.getElementById('resumeModalBadge');

  const uploadModal = document.getElementById('uploadModal');
  const uploadBtn = document.getElementById('uploadBtn');
  const closeUploadModal = document.getElementById('closeUploadModal');
  const closeUploadBtn = document.getElementById('closeUploadBtn');
  const dropZone = document.getElementById('dropZone');
  const selectFileBtn = document.getElementById('selectFileBtn');
  const csvFileInput = document.getElementById('csvFileInput');
  const uploadStatus = document.getElementById('uploadStatus');
  const resetDefaultBtn = document.getElementById('resetDefaultBtn');

  const helpModal = document.getElementById('helpModal');
  const helpBtn = document.getElementById('helpBtn');
  const closeHelpModal = document.getElementById('closeHelpModal');
  const closeHelpBtn = document.getElementById('closeHelpBtn');

  const toast = document.getElementById('toast');

  let chatHistory = [];
  let isGenerating = false;

  // Auto-resize textarea
  queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
  });

  // Enter to send (Shift+Enter for newline)
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  });

  sendBtn.addEventListener('click', handleSubmit);

  // Suggestions chip click
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const query = btn.getAttribute('data-query');
      if (query) {
        queryInput.value = query;
        queryInput.style.height = 'auto';
        handleSubmit();
      }
    });
  });

  // Fetch initial status
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();
      if (data.resumes_count) {
        poolBadge.textContent = `${data.resumes_count.toLocaleString()} Resumes`;
        if (data.is_custom_dataset) {
          poolBadge.textContent = `${data.resumes_count.toLocaleString()} Resumes (Custom)`;
        }
      }
      if (data.badge) {
        modelBadge.textContent = data.badge;
      }
    } catch (e) {
      console.error('Failed to fetch status:', e);
    }
  }
  fetchStatus();

  // Handle Form Submit
  async function handleSubmit() {
    const text = queryInput.value.trim();
    if (!text || isGenerating) return;

    isGenerating = true;
    sendBtn.disabled = true;
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Hide suggestions card and show clear button
    if (suggestionsCard) {
      suggestionsCard.style.display = 'none';
    }
    clearChatBtn.style.display = 'inline-block';

    // Append user message card
    const userMsg = appendUserMessage(text);
    userMsg.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Create assistant message placeholder
    const { assistantWrap, aiTextContainer, typingCursor, metaContainer, candidateGrid } = createAssistantMessageHolder();
    chatThread.appendChild(assistantWrap);

    const isRagFusion = ragFusionToggle.checked;
    const ragMode = isRagFusion ? 'RAG Fusion' : 'Generic RAG';

    let fullAiResponse = '';

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          rag_mode: ragMode,
          chat_history: chatHistory.slice(-4)
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;

          const jsonStr = trimmed.replace(/^data:\s*/, '');
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'meta') {
              renderMetaBanner(metaContainer, data);
              if (data.candidates && data.candidates.length > 0) {
                renderCandidateCards(candidateGrid, data.candidates);
              }
            } else if (data.type === 'token') {
              fullAiResponse += data.content;
              renderMarkdown(aiTextContainer, fullAiResponse);
              aiTextContainer.appendChild(typingCursor);
              const rect = assistantWrap.getBoundingClientRect();
              if (rect.bottom > window.innerHeight) {
                window.scrollBy({ top: rect.bottom - window.innerHeight + 30, behavior: 'smooth' });
              }
            } else if (data.type === 'error') {
              aiTextContainer.innerHTML = `<span style="color:#ef4444;">⚠️ ${data.message}</span>`;
            } else if (data.type === 'done') {
              typingCursor.remove();
            }
          } catch (err) {
            console.error('Error parsing SSE frame:', err, jsonStr);
          }
        }
      }

      // Record to chat history
      chatHistory.push(text);
      if (fullAiResponse) {
        chatHistory.push(fullAiResponse);
      }

    } catch (error) {
      console.error('Chat error:', error);
      aiTextContainer.innerHTML = `<span style="color:#ef4444;">⚠️ Failed to connect to server: ${error.message}</span>`;
    } finally {
      typingCursor.remove();
      isGenerating = false;
      sendBtn.disabled = false;
      queryInput.focus();
    }
  }

  function appendUserMessage(text) {
    const userMsg = document.createElement('div');
    userMsg.className = 'message-item';
    userMsg.innerHTML = `<div class="user-message-card">${escapeHtml(text)}</div>`;
    chatThread.appendChild(userMsg);
  }

  function createAssistantMessageHolder() {
    const assistantWrap = document.createElement('div');
    assistantWrap.className = 'assistant-message-wrap';

    const metaContainer = document.createElement('div');
    metaContainer.className = 'rag-meta-banner';
    metaContainer.style.display = 'none';

    const candidateGrid = document.createElement('div');
    candidateGrid.className = 'candidate-grid';
    candidateGrid.style.display = 'none';

    const aiTextContainer = document.createElement('div');
    aiTextContainer.className = 'ai-text-content';

    const typingCursor = document.createElement('span');
    typingCursor.className = 'typing-cursor';
    aiTextContainer.appendChild(typingCursor);

    assistantWrap.appendChild(metaContainer);
    assistantWrap.appendChild(candidateGrid);
    assistantWrap.appendChild(aiTextContainer);

    return { assistantWrap, aiTextContainer, typingCursor, metaContainer, candidateGrid };
  }

  function renderMetaBanner(container, data) {
    container.style.display = 'flex';
    
    let typeLabel = 'Semantic RAG';
    if (data.query_type === 'retrieve_applicant_id') {
      typeLabel = 'Direct Applicant ID Lookup';
    } else if (data.query_type === 'no_retrieve') {
      typeLabel = 'Chat History Follow-up';
    } else if (data.rag_mode === 'RAG Fusion') {
      typeLabel = 'RAG Fusion Search';
    }

    let html = `
      <div class="meta-top-row">
        <span class="meta-badge">⚡ ${typeLabel}</span>
        <span class="meta-latency">Latency: ${data.time_elapsed}s · ${data.total_retrieved || 0} retrieved</span>
      </div>
    `;

    if (data.subquestions && data.subquestions.length > 1) {
      html += `
        <div class="subquestions-box">
          <div class="subquestions-title">RAG Fusion Sub-Queries Generated:</div>
          ${data.subquestions.map((q, i) => `<div class="subquestion-item">• ${escapeHtml(q.replace(/^\d+\.\s*/, ''))}</div>`).join('')}
        </div>
      `;
    }

    container.innerHTML = html;
  }

  function renderCandidateCards(grid, candidates) {
    grid.style.display = 'grid';
    grid.innerHTML = '';

    candidates.forEach(c => {
      const card = document.createElement('div');
      card.className = 'candidate-card';
      
      const scoreTag = c.score !== null ? `<span class="score-badge">Match: ${(c.score * 100).toFixed(1)}%</span>` : '';
      
      card.innerHTML = `
        <div>
          <div class="candidate-header">
            <span class="rank-badge">#${c.rank}</span>
            ${scoreTag}
          </div>
          <div class="candidate-id" style="margin: 6px 0 4px 0;">Applicant ${escapeHtml(c.id)}</div>
          <div class="candidate-snippet">${escapeHtml(c.snippet)}</div>
        </div>
        <button class="inspect-resume-btn" data-id="${escapeHtml(c.id)}">Inspect Full Resume</button>
      `;

      card.querySelector('.inspect-resume-btn').addEventListener('click', () => {
        openResumeModal(c.id);
      });

      grid.appendChild(card);
    });
  }

  // Markdown rendering helper
  function renderMarkdown(container, text) {
    if (window.marked && typeof window.marked.parse === 'function') {
      container.innerHTML = window.marked.parse(text);
      return;
    }
    // Simple fast safe markdown formatter
    let formatted = escapeHtml(text)
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code style="background:#f1f5f9;padding:2px 5px;border-radius:4px;font-family:var(--font-mono);font-size:12.5px;">$1</code>')
      .replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>')
      .replace(/^\s*\d+\.\s+(.*$)/gim, '<li>$1</li>')
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');

    container.innerHTML = formatted;
  }

  function escapeHtml(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Inspect Resume Modal
  async function openResumeModal(candidateId) {
    resumeModalTitle.textContent = `Applicant ID ${candidateId}`;
    resumeModalBadge.textContent = `Candidate Profile`;
    resumeContent.textContent = 'Loading candidate resume details...';
    resumeModal.classList.add('active');

    try {
      const res = await fetch(`/api/candidate/${encodeURIComponent(candidateId)}`);
      if (!res.ok) throw new Error('Resume not found');
      const data = await res.json();
      resumeContent.textContent = data.resume || 'No resume text available.';
    } catch (e) {
      resumeContent.textContent = `Error loading resume: ${e.message}`;
    }
  }

  closeResumeModal.addEventListener('click', () => resumeModal.classList.remove('active'));
  doneResumeBtn.addEventListener('click', () => resumeModal.classList.remove('active'));
  resumeModal.addEventListener('click', (e) => {
    if (e.target === resumeModal) resumeModal.classList.remove('active');
  });

  copyResumeBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(resumeContent.textContent);
    showToast('Resume copied to clipboard!');
  });

  // Help Modal
  helpBtn.addEventListener('click', () => helpModal.classList.add('active'));
  closeHelpModal.addEventListener('click', () => helpModal.classList.remove('active'));
  closeHelpBtn.addEventListener('click', () => helpModal.classList.remove('active'));
  helpModal.addEventListener('click', (e) => {
    if (e.target === helpModal) helpModal.classList.remove('active');
  });

  // Upload Modal
  uploadBtn.addEventListener('click', () => {
    uploadModal.classList.add('active');
    uploadStatus.style.display = 'none';
  });
  closeUploadModal.addEventListener('click', () => uploadModal.classList.remove('active'));
  closeUploadBtn.addEventListener('click', () => uploadModal.classList.remove('active'));
  uploadModal.addEventListener('click', (e) => {
    if (e.target === uploadModal) uploadModal.classList.remove('active');
  });

  selectFileBtn.addEventListener('click', () => csvFileInput.click());
  csvFileInput.addEventListener('change', () => {
    if (csvFileInput.files.length > 0) {
      handleFileUpload(csvFileInput.files[0]);
    }
  });

  // Drag & drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showUploadStatus('Please upload a CSV file only.', 'error');
      return;
    }

    showUploadStatus(`Indexing ${file.name}... This may take a few moments.`, 'loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      showUploadStatus(`✅ ${data.message}`, 'success');
      poolBadge.textContent = `${data.count.toLocaleString()} Resumes (Custom)`;
      showToast(`Indexed ${data.count} resumes from ${file.name}`);
    } catch (e) {
      showUploadStatus(`❌ ${e.message}`, 'error');
    }
  }

  function showUploadStatus(msg, type) {
    uploadStatus.style.display = 'block';
    uploadStatus.className = `upload-status ${type}`;
    uploadStatus.textContent = msg;
  }

  resetDefaultBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      const data = await res.json();
      poolBadge.textContent = `${data.count.toLocaleString()} Resumes`;
      showUploadStatus('Reset to default synthetic resume pool.', 'success');
      showToast('Reset to default resumes');
    } catch (e) {
      showUploadStatus(`Error resetting: ${e.message}`, 'error');
    }
  });

  // Clear Chat
  clearChatBtn.addEventListener('click', () => {
    chatThread.innerHTML = '';
    chatHistory = [];
    if (suggestionsCard) {
      suggestionsCard.style.display = 'block';
    }
    clearChatBtn.style.display = 'none';
    showToast('Conversation cleared');
  });

  // Toast helper
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2500);
  }
});
