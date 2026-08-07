const TOOLS = [
  {
    name: 'get_current_time',
    description: 'Get the current date and time in ISO 8601 format.',
    parameters: { type: 'object', properties: {} },
    handler: async () => ({ result: new Date().toISOString() })
  },
  {
    name: 'calculate',
    description: 'Evaluate a simple mathematical expression safely, such as "2 + 2 * 5" or "(10 - 3) / 7".',
    parameters: {
      type: 'object',
      properties: { expression: { type: 'string', description: 'The math expression to evaluate' } },
      required: ['expression']
    },
    handler: async (args) => {
      const res = await fetch('/tool/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: args.expression })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Calculate failed');
      return data;
    }
  },
  {
    name: 'read_file',
    description: 'Read a text file inside the project folder. Provide a relative path like "app.js" or "folder/file.txt".',
    parameters: {
      type: 'object',
      properties: { path: { type: 'string', description: 'Relative file path' } },
      required: ['path']
    },
    handler: async (args) => {
      const res = await fetch(`/tool/read_file?path=${encodeURIComponent(args.path)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Read failed');
      return data;
    }
  },
  {
    name: 'list_directory',
    description: 'List files and folders inside a project directory. Use an empty path for the project root.',
    parameters: {
      type: 'object',
      properties: { path: { type: 'string', description: 'Relative directory path' } }
    },
    handler: async (args) => {
      const path = args.path || '';
      const res = await fetch(`/tool/list_directory?path=${encodeURIComponent(path)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'List failed');
      return data;
    }
  },
  {
    name: 'write_file',
    description: 'Write text content to a file inside the project folder. Creates parent directories if needed. Provide a relative path like "notes.txt" or "folder/file.txt".',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Relative file path' },
        content: { type: 'string', description: 'Text content to write' }
      },
      required: ['path', 'content']
    },
    handler: async (args) => {
      const res = await fetch('/tool/write_file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: args.path, content: args.content })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Write failed');
      return data;
    }
  },
  {
    name: 'search_files',
    description: 'Search project files by name or content. Returns matching file paths and short content snippets.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search term' },
        path: { type: 'string', description: 'Relative directory path (empty for root)' }
      },
      required: ['query']
    },
    handler: async (args) => {
      const body = { query: args.query, path: args.path || '' };
      const res = await fetch('/tool/search_files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Search failed');
      return data;
    }
  }
];

function getToolsForApi(enabled = []) {
  const enabledSet = new Set(enabled);
  return TOOLS
    .filter(t => enabledSet.size === 0 || enabledSet.has(t.name))
    .map(t => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
}

async function executeTool(name, args) {
  const tool = TOOLS.find(t => t.name === name);
  if (!tool) return JSON.stringify({ error: `Tool ${name} not found` });
  try {
    const result = await tool.handler(args || {});
    if (result && 'result' in result) {
      if (typeof result.result === 'string' || typeof result.result === 'number') return String(result.result);
      return JSON.stringify(result.result);
    }
    return JSON.stringify(result);
  } catch (err) {
    return `Error: ${err.message}`;
  }
}

function renderToolsList(container, enabledTools, onChange) {
  container.innerHTML = '';
  TOOLS.forEach(tool => {
    const label = document.createElement('label');
    label.className = 'tool-item';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = tool.name;
    cb.checked = enabledTools.includes(tool.name);
    cb.addEventListener('change', () => onChange(tool.name, cb.checked));
    const text = document.createElement('span');
    text.textContent = tool.name;
    const desc = document.createElement('small');
    desc.textContent = tool.description;
    label.appendChild(cb);
    const info = document.createElement('div');
    info.className = 'tool-info';
    info.appendChild(text);
    info.appendChild(desc);
    label.appendChild(info);
    container.appendChild(label);
  });
}
