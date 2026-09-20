// Agent tool metadata — categorization + grouping for the tools picker.
// Names come from GET /api/agent/tools (the harness is source of truth);
// this module only decides which group a tool renders under.

export interface ToolDef {
  name: string
  description: string
}

export interface ToolGroup {
  label: string
  tools: ToolDef[]
}

const CATEGORY_RULES: [RegExp, string][] = [
  [/^(read_file|read_file_lines|list_dir|find_files|file_info|search_repo|grep_project)$/, 'Files · read'],
  [/^(write_file|edit_file|append_file|create_file|rename_file|delete_file|search_replace|project_search_replace|undo_edit)$/, 'Files · write'],
  [/^(run_python|run_cmd|run_tests)$/, 'Shell & tests'],
  [/^git_/, 'Git'],
  [/^(web_search|web_fetch|wikipedia_search|arxiv_search)/, 'Web'],
  [/lora|checkpoint_compare/, 'LoRA & models'],
  [/^(remember|recall_memory|forget|library_|check_library|list_allowed_libraries)/, 'Memory & library'],
  [/^(get_time|set_timer|check_timer|cancel_timer|list_timers|backup_)/, 'System'],
  [/sub_agent|subagent/, 'Sub-agents'],
]

export function toolCategory(name: string): string {
  for (const [re, label] of CATEGORY_RULES) {
    if (re.test(name)) return label
  }
  return 'Other'
}

const GROUP_ORDER = [
  'Files · read', 'Files · write', 'Shell & tests', 'Git', 'Web',
  'LoRA & models', 'Memory & library', 'System', 'Sub-agents', 'Other',
]

export function groupTools(defs: ToolDef[]): ToolGroup[] {
  const map = new Map<string, ToolDef[]>()
  for (const d of defs) {
    const cat = toolCategory(d.name)
    const arr = map.get(cat) ?? []
    arr.push(d)
    map.set(cat, arr)
  }
  return GROUP_ORDER
    .filter((g) => map.has(g))
    .map((g) => ({ label: g, tools: map.get(g)! }))
}
