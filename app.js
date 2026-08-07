const DEFAULT_SETTINGS = { baseUrl: '/v1', apiKey: '', model: 'llama3', temperature: 0.7 };

const $ = id => document.getElementById(id);

const ls = {
  get: (k, def) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : def; } catch { return def; } },
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v))
};

const el = {
  sidebar: $('sidebar'),
  chatList: $('chatList'),
  newChatBtn: $('newChatBtn'),
  settingsBtn: $('settingsBtn'),
  modelsBtn: $('modelsBtn'),
  tasksBtn: $('tasksBtn'),
  settingsOverlay: $('settingsOverlay'),
  modelsOverlay: $('modelsOverlay'),
  tasksOverlay: $('tasksOverlay'),
  baseUrl: $('baseUrl'),
  apiKey: $('apiKey'),
  model: $('model'),
  temperature: $('temperature'),
  modelList: $('modelList'),
  modelsStatus: $('modelsStatus'),
  systemPrompt: $('systemPrompt'),
  toolsList: $('toolsList'),
  saveSettings: $('saveSettings'),
  fetchModels: $('fetchModels'),
  exportChat: $('exportChat'),
  importChat: $('importChat'),
  modelsListView: $('modelsListView'),
  modelsOverlayStatus: $('modelsOverlayStatus'),
  refreshModels: $('refreshModels'),
  closeModels: $('closeModels'),
  tasksListView: $('tasksListView'),
  tasksOverlayStatus: $('tasksOverlayStatus'),
  refreshTasks: $('refreshTasks'),
  closeTasks: $('closeTasks'),
  stopTask: $('stopTask'),
  taskDetail: $('taskDetail'),
  taskDetailTitle: $('taskDetailTitle'),
  taskDetailLog: $('taskDetailLog'),
  localPreset: $('localPreset'),
  modelSearch: $('modelSearch'),
  topbarTitle: $('topbarTitle'),
  topbarModel: $('topbarModel'),
  menuToggle: $('menuToggle'),
  messages: $('messages'),
  prompt: $('prompt'),
  sendBtn: $('sendBtn'),
  stopBtn: $('stopBtn')
};

const state = {
  chats: ls.get('forgeai_chats', []),
  currentId: null,
  settings: ls.get('forgeai_settings', DEFAULT_SETTINGS),
  streaming: false,
  controller: null,
  assistantContent: null,
  localStatus: 'unknown',
  modelFilter: '',
  tasksPoll: null,
  selectedTask: null
};

const robotSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/><circle cx="9" cy="15" r="1"/><circle cx="15" cy="15" r="1"/></svg>`;
const toolIcon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>`;

function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function inlineFormat(text) {
  let t = escapeHtml(text);
  const codes = [];
  t = t.replace(/`([^`]+)`/g, (m, c) => { const i = codes.push(c) - 1; return `@@CODE_${i}@@`; });
  t = t.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
       .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
       .replace(/\*(.+?)\*/g, '<em>$1</em>')
       .replace(/__(.+?)__/g, '<strong>$1</strong>')
       .replace(/_(.+?)_/g, '<em>$1</em>')
       .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  t = t.replace(/@@CODE_(\d+)@@/g, (m, i) => `<code>${codes[Number(i)]}</code>`);
  return t;
}

function renderCodeBlock(code, lang) {
  const l = escapeHtml(lang || 'text');
  return `<div class="code-block"><div class="code-header"><span class="code-lang">${l}</span><button class="copy-btn" type="button">Copy</button></div><pre><code>${escapeHtml(code)}</code></pre></div>`;
}

function renderMarkdown(src) {
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  const isBlock = l => /^(#{1,6}\s|>|```|[*+-]\s|\d+\.\s|---+\s*$)/.test(l);
  let out = '', i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith('```')) {
      const m = line.match(/^```(\w*)/);
      const lang = m ? m[1] : '';
      i++;
      const code = [];
      while (i < lines.length && !lines[i].startsWith('```')) { code.push(lines[i]); i++; }
      if (i < lines.length) i++;
      out += renderCodeBlock(code.join('\n'), lang);
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.+)$/);
    if (h) { out += `<h${h[1].length}>${inlineFormat(h[2])}</h${h[1].length}>`; i++; continue; }
    if (line.startsWith('> ')) {
      const q = [];
      while (i < lines.length && lines[i].startsWith('> ')) { q.push(lines[i].slice(2)); i++; }
      out += `<blockquote>${inlineFormat(q.join(' '))}</blockquote>`;
      continue;
    }
    if (/^[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i])) { items.push(lines[i].replace(/^[-*+]\s+/, '')); i++; }
      out += `<ul>${items.map(it => `<li>${inlineFormat(it)}</li>`).join('')}</ul>`;
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) { items.push(lines[i].replace(/^\d+\.\s+/, '')); i++; }
      out += `<ol>${items.map(it => `<li>${inlineFormat(it)}</li>`).join('')}</ol>`;
      continue;
    }
    if (/^-{3,}\s*$/.test(line)) { out += '<hr>'; i++; continue; }
    if (line.trim() === '') { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() !== '' && !isBlock(lines[i])) { para.push(lines[i]); i++; }
    out += `<p>${inlineFormat(para.join(' '))}</p>`;
  }
  return out;
}

function newChatObject(title = 'New chat') {
  const now = Date.now();
  return { id: uuid(), title, createdAt: now, updatedAt: now, messages: [], systemPrompt: '', enabledTools: [] };
}

function save() {
  ls.set('forgeai_chats', state.chats);
  ls.set('forgeai_settings', state.settings);
}

function getCurrentChat() { return state.chats.find(c => c.id === state.currentId) || null; }

function buildApiMessages(chat) {
  const messages = [];
  if (chat.systemPrompt) messages.push({ role: 'system', content: chat.systemPrompt });
  chat.messages.forEach(m => {
    if (m.role === 'assistant' && m.tool_calls) messages.push({ role: 'assistant', content: m.content || '', tool_calls: m.tool_calls });
    else if (m.role === 'tool') messages.push({ role: 'tool', tool_call_id: m.tool_call_id, content: m.content });
    else messages.push({ role: m.role, content: m.content });
  });
  return messages;
}

function appendMessage(m, idx) {
  const role = m.role;
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  wrapper.dataset.index = idx;
  const inner = document.createElement('div');
  inner.className = 'message-inner';
  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}-avatar`;
  avatar.innerHTML = role === 'assistant' ? robotSvg : 'U';
  const body = document.createElement('div');
  body.className = 'content';
  body.innerHTML = role === 'assistant' ? renderMarkdown(m.content || '') : escapeHtml(m.content).replace(/\n/g, '<br>');
  const actions = document.createElement('div');
  actions.className = 'message-actions';
  if (role === 'user') {
    const edit = document.createElement('button');
    edit.className = 'msg-action';
    edit.textContent = 'Edit';
    edit.addEventListener('click', () => editMessage(idx));
    actions.appendChild(edit);
  } else if (role === 'assistant' && !m.tool_calls) {
    const regen = document.createElement('button');
    regen.className = 'msg-action';
    regen.textContent = 'Regenerate';
    regen.addEventListener('click', () => regenerateMessage(idx));
    actions.appendChild(regen);
  }
  inner.appendChild(avatar);
  inner.appendChild(body);
  inner.appendChild(actions);
  wrapper.appendChild(inner);
  el.messages.appendChild(wrapper);
}

function appendToolSummary(m, idx) {
  const names = m.tool_calls.map(t => t.function?.name).filter(Boolean).join(', ');
  const wrapper = document.createElement('div');
  wrapper.className = 'tool-message';
  wrapper.dataset.index = idx;
  const inner = document.createElement('div');
  inner.className = 'message-inner';
  const bubble = document.createElement('div');
  bubble.className = 'tool-bubble';
  bubble.innerHTML = `${toolIcon} Used tools: ${escapeHtml(names)}`;
  inner.appendChild(bubble);
  wrapper.appendChild(inner);
  el.messages.appendChild(wrapper);
}

function renderWelcome() {
  el.messages.innerHTML = `
    <div class="welcome">
      <h1>What can I help with?</h1>
      <div class="suggestions">
        <button class="suggestion" data-text="Explain a complex topic in simple terms">Explain a topic</button>
        <button class="suggestion" data-text="Write a Python function that calculates fibonacci">Write code</button>
        <button class="suggestion" data-text="Brainstorm ideas for a side project">Brainstorm ideas</button>
      </div>
    </div>`;
  el.messages.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', () => { el.prompt.value = btn.dataset.text; autoResize(); el.prompt.focus(); updateInputState(); });
  });
}

function renderMessages() {
  const chat = getCurrentChat();
  if (!chat || !chat.messages.length) { renderWelcome(); return; }
  el.messages.innerHTML = '';
  chat.messages.forEach((m, idx) => {
    if (m.role === 'tool') return;
    if (m.role === 'assistant' && m.tool_calls && !m.content) { appendToolSummary(m, idx); return; }
    appendMessage(m, idx);
  });
  scrollToBottom();
}

function renderSidebar() {
  el.chatList.innerHTML = '';
  const sorted = state.chats.slice().sort((a, b) => b.updatedAt - a.updatedAt);
  sorted.forEach(chat => {
    const item = document.createElement('div');
    item.className = `chat-item ${chat.id === state.currentId ? 'active' : ''}`;
    item.dataset.id = chat.id;
    const title = document.createElement('span');
    title.textContent = chat.title || 'New chat';
    const del = document.createElement('button');
    del.className = 'delete-chat';
    del.textContent = '×';
    del.dataset.id = chat.id;
    item.appendChild(title);
    item.appendChild(del);
    el.chatList.appendChild(item);
  });
}

function renderTopbar() {
  const chat = getCurrentChat();
  el.topbarTitle.textContent = chat ? (chat.title || 'New chat') : 'New chat';
  const status = state.localStatus || 'unknown';
  const dotColor = status === 'connected' ? '#22c55e' : status === 'disconnected' ? '#ef4444' : '#9ca3af';
  el.topbarModel.innerHTML = `<span class="model-name">${escapeHtml(state.settings.model)}</span><span class="status-dot" style="background:${dotColor}" title="Local backend: ${status}"></span>`;
}

async function checkLocalStatus() {
  const base = (state.settings.baseUrl || DEFAULT_SETTINGS.baseUrl).replace(/\/*$/, '');
  const headers = {};
  if (state.settings.apiKey) headers['Authorization'] = `Bearer ${state.settings.apiKey}`;
  try {
    const res = await fetch(`${base}/models`, { headers, signal: AbortSignal.timeout(5000) });
    state.localStatus = res.ok ? 'connected' : 'disconnected';
  } catch {
    state.localStatus = 'disconnected';
  }
  renderTopbar();
}

function render() {
  renderSidebar();
  renderTopbar();
  renderMessages();
  updateInputState();
}

function scrollToBottom() {
  const m = el.messages;
  if (m.scrollHeight - m.scrollTop - m.clientHeight < 80) m.scrollTop = m.scrollHeight;
}

function autoResize() {
  el.prompt.style.height = 'auto';
  el.prompt.style.height = Math.min(el.prompt.scrollHeight, 200) + 'px';
}

function updateInputState() {
  el.stopBtn.classList.toggle('hidden', !state.streaming);
  el.sendBtn.classList.toggle('hidden', state.streaming);
  el.sendBtn.disabled = !el.prompt.value.trim() || state.streaming;
  el.prompt.disabled = state.streaming;
}

function createAssistantPlaceholder() {
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant streaming';
  const inner = document.createElement('div');
  inner.className = 'message-inner';
  const avatar = document.createElement('div');
  avatar.className = 'avatar assistant-avatar';
  avatar.innerHTML = robotSvg;
  const body = document.createElement('div');
  body.className = 'content';
  body.innerHTML = '<div class="spinner"><span></span><span></span><span></span></div>';
  inner.appendChild(avatar);
  inner.appendChild(body);
  wrapper.appendChild(inner);
  el.messages.appendChild(wrapper);
  scrollToBottom();
  return body;
}

function updateAssistantContent(raw) {
  if (!state.assistantContent) return;
  state.assistantContent.innerHTML = renderMarkdown(raw);
  scrollToBottom();
}

function abortStream() {
  if (state.controller) {
    state.controller.abort();
    state.controller = null;
    state.streaming = false;
    updateInputState();
  }
}

async function apiFetch(payload) {
  const { baseUrl, model, apiKey, temperature } = state.settings;
  const url = baseUrl.replace(/\/*$/, '') + '/chat/completions';
  const headers = { 'Content-Type': 'application/json' };
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
  return fetch(url, { method: 'POST', headers, body: JSON.stringify({ model, ...payload, temperature }), signal: state.controller.signal });
}

async function generate() {
  const chat = getCurrentChat();
  if (!chat) return;
  state.streaming = true;
  updateInputState();
  state.controller = new AbortController();
  state.assistantContent = createAssistantPlaceholder();
  let raw = '';
  try {
    const res = await apiFetch({ messages: buildApiMessages(chat), stream: true });
    if (!res.ok) { const t = await res.text(); throw new Error(`HTTP ${res.status}: ${t}`); }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n');
      buffer = parts.pop();
      for (const line of parts) {
        if (!line.trim() || !line.startsWith('data:')) continue;
        const data = line.replace(/^data:\s*/, '');
        if (data === '[DONE]') continue;
        try {
          const json = JSON.parse(data);
          const chunk = json.choices?.[0]?.delta?.content ?? json.choices?.[0]?.message?.content ?? '';
          if (chunk) { raw += chunk; updateAssistantContent(raw); }
        } catch {}
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      raw = `⚠️ ${err.message}`;
      updateAssistantContent(raw);
    } else {
      raw = raw || '(stopped)';
      updateAssistantContent(raw);
    }
  } finally {
    state.streaming = false;
    state.controller = null;
    state.assistantContent = null;
    chat.messages.push({ role: 'assistant', content: raw, createdAt: Date.now() });
    chat.updatedAt = Date.now();
    save();
    render();
  }
}

async function generateWithTools() {
  const chat = getCurrentChat();
  if (!chat) return;
  state.streaming = true;
  updateInputState();
  state.controller = new AbortController();
  state.assistantContent = createAssistantPlaceholder();
  let finalContent = '';
  try {
    const tools = getToolsForApi(chat.enabledTools || []);
    for (let round = 0; round < 5; round++) {
      const messages = buildApiMessages(chat);
      const res = await apiFetch({ messages, tools, tool_choice: 'auto', stream: false });
      if (!res.ok) { const t = await res.text(); throw new Error(`HTTP ${res.status}: ${t}`); }
      const data = await res.json();
      const choice = data.choices?.[0];
      const msg = choice?.message;
      if (!msg) throw new Error('No response');
      if (msg.tool_calls?.length) {
        const names = msg.tool_calls.map(t => t.function?.name).filter(Boolean).join(', ');
        updateAssistantContent(`*Using tools: ${names}*`);
        chat.messages.push({ role: 'assistant', content: '', tool_calls: msg.tool_calls, createdAt: Date.now() });
        for (const tc of msg.tool_calls) {
          let args = {};
          try { args = JSON.parse(tc.function?.arguments || '{}'); } catch {}
          const output = await executeTool(tc.function?.name, args);
          chat.messages.push({ role: 'tool', tool_call_id: tc.id, content: output, createdAt: Date.now() });
        }
        chat.updatedAt = Date.now();
        save();
      } else {
        finalContent = msg.content || '';
        break;
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') finalContent = `⚠️ ${err.message}`; else finalContent = '(stopped)';
  } finally {
    state.streaming = false;
    state.controller = null;
    state.assistantContent = null;
    const content = finalContent || '(no response)';
    chat.messages.push({ role: 'assistant', content, createdAt: Date.now() });
    chat.updatedAt = Date.now();
    save();
    render();
  }
}

function startGeneration() {
  const chat = getCurrentChat();
  if (!chat) return;
  if ((chat.enabledTools || []).length > 0) generateWithTools();
  else generate();
}

function sendMessage() {
  const text = el.prompt.value.trim();
  if (!text || state.streaming) return;
  const chat = getCurrentChat();
  if (!chat) return;
  chat.messages.push({ role: 'user', content: text, createdAt: Date.now() });
  chat.updatedAt = Date.now();
  if (chat.title === 'New chat' && chat.messages.filter(m => m.role === 'user').length === 1) {
    chat.title = text.slice(0, 40).replace(/\n/g, ' ') || 'New chat';
  }
  save();
  el.prompt.value = '';
  autoResize();
  render();
  startGeneration();
}

function editMessage(idx) {
  const chat = getCurrentChat();
  if (!chat || state.streaming) return;
  const m = chat.messages[idx];
  if (!m || m.role !== 'user') return;
  el.prompt.value = m.content;
  autoResize();
  chat.messages.splice(idx);
  chat.updatedAt = Date.now();
  save();
  render();
  el.prompt.focus();
}

function regenerateMessage(idx) {
  const chat = getCurrentChat();
  if (!chat || state.streaming) return;
  const m = chat.messages[idx];
  if (!m || m.role !== 'assistant') return;
  chat.messages.splice(idx);
  chat.updatedAt = Date.now();
  save();
  render();
  startGeneration();
}

function newChat() {
  const c = newChatObject();
  state.chats.unshift(c);
  state.currentId = c.id;
  save();
  render();
  el.prompt.focus();
}

function switchChat(id) {
  if (state.streaming) abortStream();
  state.currentId = id;
  save();
  render();
  el.prompt.focus();
  el.sidebar.classList.remove('open');
}

function deleteChat(id) {
  state.chats = state.chats.filter(c => c.id !== id);
  if (state.currentId === id) {
    if (state.chats.length) state.currentId = state.chats[0].id;
    else { const c = newChatObject(); state.chats = [c]; state.currentId = c.id; }
  }
  save();
  render();
}

function openSettings() {
  el.baseUrl.value = state.settings.baseUrl;
  el.apiKey.value = state.settings.apiKey;
  el.model.value = state.settings.model;
  el.temperature.value = state.settings.temperature;
  el.localPreset.value = [...el.localPreset.options].some(o => o.value === state.settings.baseUrl) ? state.settings.baseUrl : '';
  const chat = getCurrentChat();
  el.systemPrompt.value = chat ? (chat.systemPrompt || '') : '';
  renderToolsList(el.toolsList, chat ? (chat.enabledTools || []) : [], () => {});
  el.modelsStatus.textContent = '';
  el.settingsOverlay.classList.remove('hidden');
}

function closeSettings() { el.settingsOverlay.classList.add('hidden'); }

function saveSettings() {
  state.settings.baseUrl = el.baseUrl.value.trim() || DEFAULT_SETTINGS.baseUrl;
  state.settings.apiKey = el.apiKey.value.trim();
  state.settings.model = el.model.value.trim() || DEFAULT_SETTINGS.model;
  const t = parseFloat(el.temperature.value);
  state.settings.temperature = Number.isNaN(t) ? DEFAULT_SETTINGS.temperature : t;
  const chat = getCurrentChat();
  if (chat) {
    chat.systemPrompt = el.systemPrompt.value.trim();
    const enabled = [];
    el.toolsList.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => enabled.push(cb.value));
    chat.enabledTools = enabled;
  }
  save();
  checkLocalStatus();
  closeSettings();
  render();
}

async function fetchModels() {
  const base = el.baseUrl.value.trim().replace(/\/*$/, '');
  const key = el.apiKey.value.trim();
  const headers = {};
  if (key) headers['Authorization'] = `Bearer ${key}`;
  try {
    const res = await fetch(`${base}/models`, { headers });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    const models = data.data?.map(m => m.id) || data.models?.map(m => m.id) || [];
    el.modelList.innerHTML = '';
    models.forEach(id => { const o = document.createElement('option'); o.value = id; el.modelList.appendChild(o); });
    el.modelsStatus.textContent = `${models.length} models found`;
  } catch (err) {
    el.modelsStatus.textContent = `Failed: ${err.message}`;
  }
}

function openModels() {
  el.modelsOverlay.classList.remove('hidden');
  loadModels();
}

function closeModelsOverlay() { el.modelsOverlay.classList.add('hidden'); }

async function loadModels() {
  const { baseUrl, apiKey } = state.settings;
  const url = baseUrl.replace(/\/*$/, '') + '/models';
  const headers = {};
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
  try {
    const res = await fetch(url, { headers });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    const models = data.data?.map(m => m.id) || data.models?.map(m => m.id) || [];
    renderModelList(models);
    el.modelsOverlayStatus.textContent = `${models.length} models found`;
  } catch (err) {
    el.modelsOverlayStatus.textContent = `Failed: ${err.message}`;
  }
}

function renderModelList(models) {
  state.lastModels = models;
  el.modelsListView.innerHTML = '';
  const term = state.modelFilter || '';
  const filtered = term ? models.filter(id => id.toLowerCase().includes(term)) : models;
  if (!filtered.length) {
    el.modelsListView.innerHTML = '<div class="status">No models found.</div>';
    el.modelsOverlayStatus.textContent = '0 models shown';
    return;
  }
  filtered.forEach(id => {
    const item = document.createElement('div');
    item.className = 'model-item';
    const name = document.createElement('div');
    name.className = 'model-id';
    name.textContent = id;
    const btn = document.createElement('button');
    btn.className = 'model-select';
    btn.textContent = 'Select';
    btn.addEventListener('click', () => { state.settings.model = id; save(); render(); closeModelsOverlay(); });
    item.appendChild(name);
    item.appendChild(btn);
    el.modelsListView.appendChild(item);
  });
  el.modelsOverlayStatus.textContent = `${filtered.length} model${filtered.length === 1 ? '' : 's'} shown`;
}

function openTasks() {
  el.tasksOverlay.classList.remove('hidden');
  loadTasks();
  state.tasksPoll = setInterval(loadTasks, 5000);
}

function closeTasksOverlay() {
  el.tasksOverlay.classList.add('hidden');
  el.taskDetail.classList.add('hidden');
  if (state.tasksPoll) { clearInterval(state.tasksPoll); state.tasksPoll = null; }
}

async function loadTasks() {
  try {
    const res = await fetch('/tasks');
    const data = await res.json();
    renderTasks(data.tasks || []);
    el.tasksOverlayStatus.textContent = `${(data.tasks || []).length} task(s)`;
  } catch (err) {
    el.tasksOverlayStatus.textContent = `Failed: ${err.message}`;
  }
}

function statusClass(status) {
  if (status === 'running') return 'running';
  if (status === 'completed') return 'completed';
  return 'failed';
}

function renderTasks(tasks) {
  el.tasksListView.innerHTML = '';
  if (!tasks.length) {
    el.tasksListView.innerHTML = '<div class="status">No tasks yet. Start training from the terminal.</div>';
    return;
  }
  tasks.forEach(task => {
    const item = document.createElement('div');
    item.className = 'task-item';
    const name = document.createElement('span');
    name.className = 'task-name';
    name.textContent = `${task.name} (${task.id})`;
    const badge = document.createElement('span');
    badge.className = `task-status ${statusClass(task.status)}`;
    badge.textContent = task.status;
    const meta = document.createElement('span');
    meta.className = 'task-meta';
    const prog = task.progress || {};
    const metrics = task.metrics || {};
    let metaText = '';
    if (prog.step && prog.max_steps) metaText += `step ${prog.step}/${prog.max_steps} `;
    if (metrics.loss != null) metaText += `loss ${metrics.loss} `;
    if (metrics.tok_per_sec != null) metaText += `${metrics.tok_per_sec} tok/s`;
    meta.textContent = metaText.trim() || task.message || '';
    item.appendChild(name);
    item.appendChild(badge);
    item.appendChild(meta);
    item.addEventListener('click', () => showTaskDetail(task.id));
    el.tasksListView.appendChild(item);
  });
}

async function showTaskDetail(taskId) {
  state.selectedTask = taskId;
  try {
    const res = await fetch(`/tasks/${taskId}`);
    const task = await res.json();
    el.taskDetailTitle.textContent = `${task.name} — ${task.status}`;
    const metrics = JSON.stringify(task.metrics || {}, null, 2);
    const progress = JSON.stringify(task.progress || {}, null, 2);
    const log = (task.log_tail || []).join('\n');
    el.taskDetailLog.textContent = `METRICS:\n${metrics}\n\nPROGRESS:\n${progress}\n\nLOG:\n${log}`;
    el.taskDetail.classList.remove('hidden');
  } catch (err) {
    el.taskDetailLog.textContent = `Error loading task: ${err.message}`;
    el.taskDetail.classList.remove('hidden');
  }
}

async function stopSelectedTask() {
  if (!state.selectedTask) return;
  try {
    await fetch(`/tasks/${state.selectedTask}/stop`, { method: 'POST' });
    await showTaskDetail(state.selectedTask);
  } catch (err) {
    alert(err.message);
  }
}

function exportChat() {
  const chat = getCurrentChat();
  if (!chat) return;
  const blob = new Blob([JSON.stringify(chat, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${(chat.title || 'chat').replace(/[^a-z0-9\-_]/gi, '_')}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function importChat(file) {
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    if (!data.messages || !Array.isArray(data.messages)) throw new Error('Invalid chat file');
    const c = newChatObject(data.title || 'Imported chat');
    c.messages = data.messages.map(m => ({
      role: m.role,
      content: m.content,
      tool_call_id: m.tool_call_id,
      tool_calls: m.tool_calls,
      createdAt: m.createdAt || Date.now()
    }));
    c.systemPrompt = data.systemPrompt || '';
    c.enabledTools = data.enabledTools || [];
    state.chats.unshift(c);
    state.currentId = c.id;
    save();
    render();
  } catch (err) {
    alert('Import failed: ' + err.message);
  }
}

function bindEvents() {
  el.newChatBtn.addEventListener('click', newChat);
  el.settingsBtn.addEventListener('click', openSettings);
  el.modelsBtn.addEventListener('click', openModels);
  el.tasksBtn.addEventListener('click', openTasks);
  el.saveSettings.addEventListener('click', saveSettings);
  el.fetchModels.addEventListener('click', fetchModels);
  el.refreshModels.addEventListener('click', loadModels);
  el.closeModels.addEventListener('click', closeModelsOverlay);
  el.refreshTasks.addEventListener('click', loadTasks);
  el.closeTasks.addEventListener('click', closeTasksOverlay);
  el.stopTask.addEventListener('click', stopSelectedTask);
  el.exportChat.addEventListener('click', exportChat);
  el.importChat.addEventListener('change', e => { if (e.target.files[0]) importChat(e.target.files[0]); e.target.value = ''; });
  el.localPreset.addEventListener('change', () => {
    const preset = el.localPreset.value;
    if (preset) { el.baseUrl.value = preset; el.apiKey.value = ''; }
    if (preset === 'http://localhost:8080/v1') { el.model.value = 'custom-research-llm'; }
  });
  el.topbarModel.addEventListener('click', openModels);
  el.modelSearch.addEventListener('input', () => { state.modelFilter = el.modelSearch.value.trim().toLowerCase(); renderModelList(state.lastModels || []); });
  el.settingsOverlay.addEventListener('click', e => { if (e.target === el.settingsOverlay) closeSettings(); });
  el.modelsOverlay.addEventListener('click', e => { if (e.target === el.modelsOverlay) closeModelsOverlay(); });
  el.tasksOverlay.addEventListener('click', e => { if (e.target === el.tasksOverlay) closeTasksOverlay(); });
  el.menuToggle.addEventListener('click', () => el.sidebar.classList.toggle('open'));
  el.sendBtn.addEventListener('click', sendMessage);
  el.stopBtn.addEventListener('click', abortStream);
  el.prompt.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
  el.prompt.addEventListener('input', () => { autoResize(); updateInputState(); });
  el.chatList.addEventListener('click', e => {
    const del = e.target.closest('.delete-chat');
    if (del) { e.stopPropagation(); deleteChat(del.dataset.id); return; }
    const item = e.target.closest('.chat-item');
    if (item) switchChat(item.dataset.id);
  });
  el.messages.addEventListener('click', e => {
    const btn = e.target.closest('.copy-btn');
    if (!btn) return;
    const code = btn.closest('.code-block')?.querySelector('pre code');
    if (!code) return;
    navigator.clipboard.writeText(code.textContent).then(() => {
      const original = btn.textContent;
      btn.textContent = 'Copied';
      setTimeout(() => btn.textContent = original, 1500);
    }).catch(() => { btn.textContent = 'Failed'; setTimeout(() => btn.textContent = 'Copy', 1500); });
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeSettings(); closeModelsOverlay(); closeTasksOverlay(); } });
}

function init() {
  if (!state.chats.length) state.chats.push(newChatObject());
  state.currentId = state.chats[0].id;
  bindEvents();
  render();
  checkLocalStatus();
  autoResize();
  el.prompt.focus();
}

init();
