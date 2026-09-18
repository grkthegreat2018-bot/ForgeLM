// Agent page — chat-based agentic mode.
// Left: run list + config. Right: the selected run as a chat thread —
// the task and mid-run user messages are user bubbles; agent text,
// thinking and tool activity are assistant-side cards. The composer
// steers a running agent or starts a new run when idle.
// Run events persist server-side; switching pages re-hydrates the
// thread via /api/agent/runs/{id} and live events merge by seq.

import { clsx } from 'clsx'
import {
  Bot, Brain, Check, ChevronDown, ChevronRight, FolderOpen,
  Send, Square, Wrench, X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { parseStream, toolCallName, toolCallSummary } from '../lib/streamParse'
import { useForge } from '../lib/store'
import type { AgentEvent, AgentRunInfo, ToolResultRec } from '../lib/types'
import {
  Btn, Card, Check as CheckBox, EmptyState, Input,
  Select, Spinner, Tag, Textarea,
} from '../components/ui'

const APPROVAL_MODES = [
  { v: 'destructive', label: 'destructive ops' },
  { v: 'all', label: 'all tools' },
  { v: 'none', label: 'none (YOLO)' },
]
const ROUND_MODES = [
  { v: 'auto', label: 'auto — model decides' },
  { v: '8', label: '8 rounds' },
  { v: '16', label: '16 rounds' },
  { v: '32', label: '32 rounds' },
  { v: '64', label: '64 rounds' },
]
const AGENT_TOOLS = [
  'read_file', 'write_file', 'edit_file', 'search_replace', 'project_search_replace',
  'run_python', 'run_cmd', 'run_tests', 'list_dir', 'find_files', 'git_status',
  'git_diff', 'web_search', 'checkpoint_compare', 'undo_edit', 'delete_file',
  'create_file', 'rename_file', 'git_revert', 'git_stash', 'git_branch',
  'search_repo', 'read_file_lines', 'spawn_subagent', 'lora_status',
  'lora_apply', 'lora_remove', 'lora_merge', 'lora_keep_for_task',
  'lora_edit_training_data', 'start_lora_training', 'check_lora_training_progress',
  'library_list', 'library_install', 'backup_list', 'backup_restore', 'backup_delete',
]

export default function Agent() {
  const engine = useForge((s) => s.engine)
  const agentRuns = useForge((s) => s.agentRuns)
  const runEvents = useForge((s) => s.agentRunEvents)
  const selectedId = useForge((s) => s.selectedAgentRun)
  const selectRun = useForge((s) => s.selectAgentRun)
  const respond = useForge((s) => s.agentRespond)
  const sendMessage = useForge((s) => s.agentMessage)
  const clearRunEvents = useForge((s) => s.clearRunEvents)

  const [input, setInput] = useState('')
  const [rounds, setRounds] = useState('auto')
  const [approval, setApproval] = useState('destructive')
  const [project, setProject] = useState('')
  const [toolsOpen, setToolsOpen] = useState(false)
  const [tools, setTools] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const feedRef = useRef<HTMLDivElement>(null)

  // pick a run to display: explicit selection → active run → newest
  const run = useMemo(() => {
    const sel = agentRuns.find((r) => r.run_id === selectedId)
    return sel ?? agentRuns.find((r) => r.status === 'running') ?? agentRuns[0]
  }, [agentRuns, selectedId])

  const events = run ? runEvents[run.run_id] ?? [] : []

  const pendingApproval = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i]
      if (e.kind === 'approval_requested') {
        const later = events.slice(i + 1).some(
          (x) => x.kind === 'tool_result' || x.kind === 'finished' || x.kind === 'failed')
        if (!later) return e
      }
    }
    return null
  }, [events])

  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
  }, [events.length])

  const ready = engine.state === 'ready'
  const running = run?.status === 'running'

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return
    setError('')
    setBusy(true)
    try {
      if (running && run) {
        // steer the running agent — injected as a user turn next round
        await sendMessage(run.run_id, text)
      } else {
        const r = await api.post<{ ok: boolean; run_id?: string; error?: string }>(
          '/api/agent/runs', {
            task: text,
            project: project || undefined,
            max_rounds: rounds === 'auto' ? null : Number(rounds),
            approval_mode: approval,
            enabled_tools: tools.size ? [...tools] : undefined,
          })
        if (!r.ok) throw new Error(r.error || 'failed to start')
        if (r.run_id) selectRun(r.run_id)
      }
      setInput('')
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  const cancel = async () => {
    if (running && run) {
      await api.post(`/api/agent/runs/${run.run_id}/cancel`).catch(() => undefined)
    }
  }

  const toggleTool = (t: string) => {
    setTools((s) => {
      const n = new Set(s)
      if (n.has(t)) n.delete(t); else n.add(t)
      return n
    })
  }

  return (
    <div className="flex h-full min-h-0">
      {/* left: runs + config */}
      <div className="w-[280px] shrink-0 border-r border-border overflow-y-auto p-3 space-y-3">
        <Card title="Runs">
          <div className="space-y-1">
            {agentRuns.slice(0, 20).map((r) => (
              <RunRow key={r.run_id} run={r}
                active={r.run_id === run?.run_id}
                onClick={() => selectRun(r.run_id)} />
            ))}
            {!agentRuns.length && (
              <div className="text-[11.5px] text-text-faint">no runs yet</div>
            )}
          </div>
        </Card>

        <Card title="Configuration">
          <div className="space-y-2.5">
            <div className="text-[12px] text-text-dim space-y-1">
              <span>Project</span>
              <Input value={project} onChange={(e) => setProject(e.target.value)}
                placeholder="name → ForgeAI_Projects/<name>" className="w-full" />
            </div>
            <div className="flex items-center justify-between text-[12px] text-text-dim">
              <span>Stop</span>
              <Select value={rounds} onChange={(e) => setRounds(e.target.value)}>
                {ROUND_MODES.map((m) =>
                  <option key={m.v} value={m.v}>{m.label}</option>)}
              </Select>
            </div>
            <div className="flex items-center justify-between text-[12px] text-text-dim">
              <span>Approve</span>
              <Select value={approval} onChange={(e) => setApproval(e.target.value)}>
                {APPROVAL_MODES.map((m) =>
                  <option key={m.v} value={m.v}>{m.label}</option>)}
              </Select>
            </div>
          </div>
        </Card>

        <Card>
          <button
            onClick={() => setToolsOpen(!toolsOpen)}
            className="flex items-center justify-between w-full text-[12px] text-text-dim hover:text-text cursor-pointer">
            <span className="flex items-center gap-1.5">
              <Wrench size={12} /> Tools {tools.size > 0 && `(${tools.size} selected)`}
            </span>
            {toolsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          {toolsOpen && (
            <div className="mt-2 space-y-1 max-h-56 overflow-y-auto">
              <div className="flex gap-2 mb-1.5">
                <button onClick={() => setTools(new Set(AGENT_TOOLS))}
                  className="text-[10.5px] text-accent-hi hover:underline cursor-pointer">all</button>
                <button onClick={() => setTools(new Set())}
                  className="text-[10.5px] text-text-faint hover:underline cursor-pointer">none</button>
              </div>
              {AGENT_TOOLS.map((t) => (
                <CheckBox key={t} label={<span className="font-mono text-[11px]">{t}</span>}
                  checked={tools.size === 0 || tools.has(t)}
                  onChange={() => toggleTool(t)} />
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* right: chat thread */}
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex items-center gap-2 px-5 py-2 border-b border-border">
          {run ? (
            <>
              <span className="text-[12px] font-medium text-text-dim truncate max-w-[40%]"
                title={run.task}>{run.task}</span>
              {run.project && (
                <span title={run.workspace}>
                  <Tag kind="accent"><FolderOpen size={11} /> {run.project}</Tag>
                </span>
              )}
              {run.status === 'running' && <Tag kind="accent"><Spinner /> running</Tag>}
              {run.status !== 'running' && (
                <Tag kind={run.status === 'done' ? 'ok' : 'err'}>{run.status}</Tag>
              )}
              <span className="text-[10.5px] text-text-faint">
                {run.rounds > 0 && `${run.rounds} rounds`}
              </span>
            </>
          ) : (
            <span className="text-[12px] font-medium text-text-dim">Agent</span>
          )}
          <div className="ml-auto flex gap-1.5">
            {running && (
              <Btn variant="danger" onClick={cancel}>
                <Square size={12} /> Stop
              </Btn>
            )}
            <Btn variant="ghost" disabled={!run || !events.length}
              onClick={() => run && clearRunEvents(run.run_id)}>
              Clear
            </Btn>
          </div>
        </div>

        <div ref={feedRef} className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
          {!run && (
            <EmptyState
              icon={<Bot size={44} strokeWidth={1.2} />}
              title="Agent idle"
              desc="Send a message below. The agent plans, calls tools (files, shell, web, tests), and reports back — the whole thread streams here live. Projects land in ForgeAI_Projects/<name>." />
          )}
          {events.map((e) => <EventRow key={e.seq} evt={e} />)}
          {pendingApproval && run && (
            <div className="bg-warn/10 border border-warn/40 rounded-card p-4">
              <div className="flex items-center gap-2 mb-1">
                <Tag kind="warn">approval required</Tag>
                <span className="text-[12.5px] font-medium">
                  {(pendingApproval.data as { tool?: string }).tool ??
                    (pendingApproval.data as { name?: string }).name}
                </span>
              </div>
              <pre className="text-[11.5px] text-text-dim font-mono whitespace-pre-wrap max-h-40 overflow-y-auto mb-3">
                {JSON.stringify((pendingApproval.data as { args?: unknown }).args, null, 2)}
              </pre>
              <div className="flex gap-2">
                <Btn variant="primary" onClick={() => respond(run.run_id, true)}>
                  <Check size={13} /> Approve
                </Btn>
                <Btn variant="danger" onClick={() => respond(run.run_id, false)}>
                  <X size={13} /> Deny
                </Btn>
              </div>
            </div>
          )}
        </div>

        {/* composer — steers a live run or starts a new one */}
        <div className="border-t border-border px-5 py-3">
          {error && <div className="text-[11.5px] text-err mb-2">{error}</div>}
          <div className="flex gap-2 items-end">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
              }}
              rows={2}
              placeholder={running
                ? 'Steer the agent — your message joins the next round…'
                : ready
                  ? 'Describe the task — a new project folder is created in ForgeAI_Projects…'
                  : 'Load a model first…'}
              disabled={!ready && !running}
              className="flex-1 !text-[13px] resize-none" />
            <Btn variant="primary" onClick={send}
              disabled={!input.trim() || busy || (!ready && !running)}>
              {busy ? <Spinner /> : <Send size={13} />}
            </Btn>
          </div>
        </div>
      </div>
    </div>
  )
}

function RunRow({ run, active, onClick }: {
  run: AgentRunInfo; active: boolean; onClick: () => void
}) {
  const kind = run.status === 'running' ? 'accent'
    : run.status === 'done' ? 'ok' : 'err'
  return (
    <button onClick={onClick}
      className={clsx(
        'w-full flex items-center gap-2 py-1.5 px-2 rounded-lg text-left transition-colors cursor-pointer',
        active ? 'bg-panel-alt' : 'hover:bg-panel-alt/60')}>
      <Tag kind={kind} className="shrink-0">{run.status}</Tag>
      <span className="text-[11.5px] text-text-dim truncate flex-1" title={run.task}>
        {run.project || run.task}
      </span>
    </button>
  )
}

function EventRow({ evt }: { evt: AgentEvent }) {
  const d = evt.data as Record<string, unknown>
  switch (evt.kind) {
    case 'started':
      return <UserBubble text={String(d.task ?? '')} />
    case 'user_message':
      return <UserBubble text={String(evt.data ?? '')} />
    case 'round_started':
      return evt.round != null && evt.round > 0 ? (
        <div className="flex items-center gap-2 pt-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-text-faint">
            round {evt.round + 1}
          </span>
          <div className="flex-1 h-px bg-border" />
        </div>
      ) : null
    case 'prompt_rendered':
    case 'raw_output':
    case 'created':
      return null
    case 'text': {
      const segs = parseStream(String(d.text ?? ''))
      return (
        <div className="space-y-2">
          {segs.map((s, i) => s.kind === 'think' ? (
            <ThinkCard key={i} text={s.text} streaming={!s.closed} />
          ) : s.kind === 'body' ? (
            <div key={i}
              className="text-[13px] whitespace-pre-wrap leading-relaxed max-w-[85%]">
              {s.text}
            </div>
          ) : s.kind === 'tool_call' ? (
            <ToolCallCard key={i} name={toolCallName(s.text)}
              args={s.text} streaming={!s.closed} />
          ) : null)}
        </div>
      )
    }
    case 'tool_call':
      return <ToolCallCard name={String(d.name ?? 'tool')}
        args={JSON.stringify(d)} streaming={false} />
    case 'tool_result':
      return <ToolResultCard rec={d as unknown as ToolResultRec} />
    case 'finished':
      return (
        <div className="bg-ok/10 border border-ok/30 rounded-lg px-3 py-2 text-[12.5px] text-ok">
          {String(d.summary || 'run finished')}
          {typeof d.rounds === 'number' && (
            <span className="text-text-faint"> · {d.rounds} rounds</span>)}
        </div>
      )
    case 'failed':
      return (
        <div className="bg-err/10 border border-err/30 rounded-lg px-3 py-2 text-[12.5px] text-err">
          {String(d.error || 'run failed')}
        </div>
      )
    default:
      return null
  }
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="bg-accent/15 border border-accent/30 rounded-2xl rounded-tr-sm px-4 py-2 text-[13px] whitespace-pre-wrap max-w-[75%]">
        {text}
      </div>
    </div>
  )
}

function ThinkCard({ text, streaming }: { text: string; streaming: boolean }) {
  const [open, setOpen] = useState(streaming)
  return (
    <div className="bg-think/8 border-l-2 border-think rounded-r-lg overflow-hidden max-w-[85%]">
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] text-think w-full hover:bg-think/10 transition-colors cursor-pointer">
        <Brain size={11} />
        {streaming ? 'thinking…' : 'thought'}
        {open ? <ChevronDown size={12} className="ml-auto" />
          : <ChevronRight size={12} className="ml-auto" />}
      </button>
      {open && (
        <div className="px-3 pb-2 text-[12px] text-text-dim whitespace-pre-wrap font-mono max-h-72 overflow-y-auto">
          {text.trim()}
        </div>
      )}
    </div>
  )
}

function ToolCallCard({ name, args, streaming }: {
  name: string; args: string; streaming: boolean
}) {
  const [open, setOpen] = useState(false)
  let summary = ''
  try {
    summary = toolCallSummary(args || '{}')
  } catch { /* fall through to raw */ }
  if (!summary) summary = args.length > 90 ? args.slice(0, 90) + '…' : args
  return (
    <div className="border border-border rounded-lg overflow-hidden max-w-[85%]">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-left hover:bg-panel-alt transition-colors cursor-pointer">
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Wrench size={11} className="text-accent" />
        <span className="text-[12px] font-mono">{name}</span>
        {streaming && <Spinner />}
        <span className="text-[10.5px] text-text-faint truncate ml-1">{summary}</span>
      </button>
      {open && (
        <pre className="px-3 py-2 text-[11.5px] font-mono text-text-dim whitespace-pre-wrap max-h-64 overflow-y-auto border-t border-border bg-bg-alt">
          {args}
        </pre>
      )}
    </div>
  )
}

function ToolResultCard({ rec }: { rec: ToolResultRec }) {
  const [open, setOpen] = useState(false)
  const text = String(rec.result?.text ?? JSON.stringify(rec.result))
  const preview = text.length > 90 ? text.slice(0, 90) + '…' : text
  return (
    <div className={clsx(
      'border rounded-lg overflow-hidden max-w-[85%]',
      rec.ok ? 'border-border' : 'border-err/40')}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-left hover:bg-panel-alt transition-colors cursor-pointer">
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className={clsx('w-1.5 h-1.5 rounded-full shrink-0',
          rec.ok ? 'bg-ok' : 'bg-err')} />
        <span className="text-[12px] font-mono">{rec.name}</span>
        <span className="text-[10.5px] text-text-faint truncate">{preview}</span>
        <span className="text-[10.5px] text-text-faint ml-auto shrink-0">
          {rec.elapsed_s.toFixed(2)}s
        </span>
      </button>
      {open && (
        <pre className="px-3 py-2 text-[11.5px] font-mono text-text-dim whitespace-pre-wrap max-h-64 overflow-y-auto border-t border-border bg-bg-alt">
          {text}
        </pre>
      )}
    </div>
  )
}
