// Chat Studio — 3-column: conversation list | messages | settings.
// Streaming SSE, think/tool_call segmentation, ratings, export.

import { clsx } from 'clsx'
import {
  Brain, ChevronRight, Download, MessageSquare, Pencil, Plus, Send,
  Square, Star, Trash2, Wrench,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { api, ssePost } from '../lib/api'
import {
  parseStream, toolCallName, toolCallSummary, type Segment,
} from '../lib/streamParse'
import { useForge } from '../lib/store'
import type { ChatMessage, ChatSummary, Conversation } from '../lib/types'
import {
  Btn, Card, Check, EmptyState, IconBtn, Input,
  NumInput, Spinner, Tag, Textarea,
} from '../components/ui'

interface StreamState {
  raw: string
  toolResults: { name: string; result: string }[]
  tokPerSec: number
  pendingCalls: number
}

export default function Chat() {
  const engine = useForge((s) => s.engine)
  const location = useLocation()
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [search, setSearch] = useState('')
  const [conv, setConv] = useState<Conversation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [stream, setStream] = useState<StreamState | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // settings
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [topK, setTopK] = useState(50)
  const [repPenalty, setRepPenalty] = useState(1.05)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [useMaster, setUseMaster] = useState(true)
  const [toolsEnabled, setToolsEnabled] = useState(true)
  const [showThinking, setShowThinking] = useState(true)
  const [draft, setDraft] = useState('')

  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const loadList = () =>
    api.get<{ conversations: ChatSummary[] }>('/api/chats')
      .then((r) => setChats(r.conversations))
      .catch(() => undefined)

  useEffect(() => { loadList() }, [])

  // deep link from Agent / other pages: /chat?open=<id>
  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const open = params.get('open')
    if (open) void loadConv(open)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.search])

  const loadConv = async (id: string) => {
    try {
      const c = await api.get<Conversation & { error?: string }>(`/api/chats/${id}`)
      if (c.error || !c.messages) return
      setConv(c)
      setError('')
    } catch (e) { setError(String(e)) }
  }

  const newChat = async () => {
    const c = await api.post<Conversation>('/api/chats', { title: 'New chat' })
    setConv(c)
    loadList()
  }

  const deleteChat = async (id: string) => {
    await api.del(`/api/chats/${id}`).catch(() => undefined)
    if (conv?.id === id) setConv(null)
    loadList()
  }

  const exportRated = async () => {
    try {
      const r = await api.post<{ path: string; examples: number }>('/api/chats/export')
      setError(`exported ${r.examples} examples → ${r.path}`)
    } catch (e) { setError(String(e)) }
  }

  const renameChat = async () => {
    if (!conv) return
    const title = window.prompt('Rename conversation', conv.title)
    if (!title?.trim()) return
    await api.post(`/api/chats/${conv.id}/rename`, { title: title.trim() })
    setConv({ ...conv, title: title.trim() })
    loadList()
  }

  const rate = async (idx: number, rating: 'good' | 'bad') => {
    if (!conv) return
    await api.post(`/api/chats/${conv.id}/rate`, { msg_idx: idx, rating })
    setConv({
      ...conv,
      messages: conv.messages.map((m, i) => i === idx ? { ...m, rating } : m),
    })
  }

  const send = async () => {
    const text = draft.trim()
    if (!text || busy) return
    if (engine.state !== 'ready') {
      setError('Engine not ready — load a model on the Engine or Models page.')
      return
    }
    setError('')
    setDraft('')
    setBusy(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl

    // optimistic user bubble; server persists it too via chat_store
    const userMsg: ChatMessage = { role: 'user', content: text, ts: Date.now() / 1000 }
    setConv((c) => c
      ? { ...c, messages: [...c.messages, userMsg] }
      : { id: '', title: text.slice(0, 48), model: '', created_at: 0,
          updated_at: 0, messages: [userMsg] })

    setStream({ raw: '', toolResults: [], tokPerSec: 0, pendingCalls: 0 })
    const toolResults: { name: string; result: string }[] = []
    let raw = ''
    let t0 = 0
    let nTok = 0
    let convId = conv?.id ?? ''
    try {
      await ssePost('/api/chat/send', {
        conv_id: convId,
        message: text,
        system_prompt: systemPrompt || undefined,
        use_master: useMaster,
        model: '',
        max_new_tokens: maxTokens,
        temperature, top_p: topP, top_k: topK,
        repetition_penalty: repPenalty,
        tools_enabled: toolsEnabled,
        thinking: true,
      }, (evt) => {
        const tps = nTok && t0 ? nTok / ((Date.now() - t0) / 1000) : 0
        if (evt.type === 'conv') {
          convId = (evt.data as { id: string }).id
        } else if (evt.type === 'token') {
          if (!t0) t0 = Date.now()
          nTok++
          raw += evt.data as string
          setStream({ raw, toolResults, tokPerSec: tps, pendingCalls: 0 })
        } else if (evt.type === 'tool_call') {
          setStream({ raw, toolResults, tokPerSec: tps, pendingCalls: 1 })
        } else if (evt.type === 'tool_result') {
          const d = evt.data as { name: string; result: { text?: string } }
          toolResults.push({
            name: d.name,
            result: d.result?.text ?? JSON.stringify(d.result),
          })
          setStream({ raw, toolResults, tokPerSec: tps, pendingCalls: 0 })
        } else if (evt.type === 'saved') {
          convId = (evt.data as { id: string }).id
        } else if (evt.type === 'error') {
          setError(String(evt.data))
        }
      }, ctrl.signal)
    } catch (e) {
      if (!ctrl.signal.aborted) setError(String(e))
    } finally {
      setBusy(false)
      abortRef.current = null
      if (raw) {
        setConv((c) => c ? { ...c, messages: [...c.messages,
          { role: 'assistant', content: raw, ts: Date.now() / 1000 }] } : c)
      }
      setStream(null)
      if (convId) loadConv(convId).catch(() => undefined)
      loadList()
      inputRef.current?.focus()
    }
  }

  const cancel = () => abortRef.current?.abort()

  const filteredChats = useMemo(() => {
    const q = search.toLowerCase()
    return chats.filter((c) => c.title.toLowerCase().includes(q))
  }, [chats, search])

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [conv?.messages.length, stream?.raw])

  const ready = engine.state === 'ready'

  return (
    <div className="flex h-full min-h-0">
      {/* ---- left: conversation list ---- */}
      <div className="w-[210px] shrink-0 border-r border-border flex flex-col">
        <div className="p-2.5 space-y-2">
          <div className="flex gap-1.5">
            <Btn variant="primary" className="flex-1 justify-center" onClick={newChat}>
              <Plus size={14} /> New chat
            </Btn>
            <IconBtn title="Rename" onClick={renameChat} className="border border-border">
              <Pencil size={12} />
            </IconBtn>
          </div>
          <Input value={search} onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter…" className="w-full !py-1" />
        </div>
        <div className="flex-1 overflow-y-auto px-1.5 pb-2">
          {filteredChats.map((c) => (
            <button
              key={c.id}
              onClick={() => loadConv(c.id)}
              className={clsx(
                'w-full text-left px-2.5 py-2 rounded-lg mb-0.5 group cursor-pointer transition-colors',
                conv?.id === c.id
                  ? 'bg-accent/15 text-text border border-accent/25'
                  : 'text-text-dim hover:bg-panel-alt border border-transparent')}>
              <div className="flex items-center gap-1.5">
                <span className="text-[12.5px] truncate flex-1">{c.title}</span>
                <IconBtn className="!w-5 !h-5 opacity-0 group-hover:opacity-100 hover:!text-err"
                  title="Delete"
                  onClick={(e) => { e.stopPropagation(); void deleteChat(c.id) }}>
                  <Trash2 size={11} />
                </IconBtn>
              </div>
              <div className="text-[10.5px] text-text-faint mt-0.5 flex gap-2">
                <span>{c.n_messages} msgs</span>
                {c.good > 0 && <span className="text-ok">+{c.good}</span>}
                {c.bad > 0 && <span className="text-err">-{c.bad}</span>}
              </div>
            </button>
          ))}
          {!filteredChats.length && (
            <div className="text-[11.5px] text-text-faint text-center py-6">
              no conversations
            </div>
          )}
        </div>
      </div>

      {/* ---- center: transcript + composer ---- */}
      <div className="flex-1 min-w-0 flex flex-col">
        <div ref={listRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {!conv?.messages.length && !stream && (
            <EmptyState
              icon={<MessageSquare size={40} strokeWidth={1.2} />}
              title={ready ? 'Start a conversation' : 'Engine not loaded'}
              desc={ready
                ? 'Plain assistant chat — or let the model call tools like read_file, list_dir, run_python.'
                : 'Load a checkpoint on the Models page to start chatting.'}
            />
          )}
          {conv?.messages.map((m, i) => (
            <MessageRow key={i} msg={m} showThinking={showThinking}
              onRate={m.role === 'assistant' ? (r) => rate(i, r) : undefined} />
          ))}
          {stream && (
            <StreamingRow stream={stream} showThinking={showThinking} />
          )}
          {error && (
            <div className="text-[12.5px] text-err bg-err/10 border border-err/30 rounded-lg px-3 py-2">
              {error}
            </div>
          )}
        </div>

        {/* composer */}
        <div className="px-6 pb-4 pt-1">
          <div className="bg-panel border border-border rounded-xl shadow-card">
            <Textarea
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void send()
                }
              }}
              placeholder={ready ? 'Message — Enter to send, Shift+Enter for newline' : 'Load a model first…'}
              disabled={!ready && !busy}
              rows={3}
              className="w-full !border-0 !bg-transparent !rounded-xl resize-none text-[13.5px] focus:!outline-none" />
            <div className="flex items-center gap-2 px-3 pb-2.5">
              {busy && <Tag kind="accent"><Spinner /> streaming</Tag>}
              {stream && stream.tokPerSec > 0 && (
                <span className="text-[11px] text-text-faint">
                  {stream.tokPerSec.toFixed(1)} tok/s
                </span>
              )}
              <div className="ml-auto flex gap-1.5">
                {busy ? (
                  <Btn variant="danger" onClick={cancel}>
                    <Square size={12} /> Stop
                  </Btn>
                ) : (
                  <Btn variant="primary" onClick={send}
                    disabled={!draft.trim() || !ready}>
                    <Send size={13} /> Send
                  </Btn>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ---- right: settings ---- */}
      <div className="w-[230px] shrink-0 border-l border-border overflow-y-auto p-3 space-y-3">
        <Card title="Generation">
          <div className="space-y-2">
            <NumInput label="Temp" value={temperature} onChange={setTemperature} min={0} max={2} step={0.05} />
            <NumInput label="Top-p" value={topP} onChange={setTopP} min={0} max={1} step={0.05} />
            <NumInput label="Top-k" value={topK} onChange={setTopK} min={0} max={200} step={1} />
            <NumInput label="Rep. pen." value={repPenalty} onChange={setRepPenalty} min={1} max={1.5} step={0.01} />
            <NumInput label="Max tok" value={maxTokens} onChange={setMaxTokens} min={1} max={8192} step={64} />
          </div>
        </Card>
        <Card title="System prompt">
          <Textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="You are a helpful assistant…"
            rows={3} className="w-full !text-[12px]" />
        </Card>
        <Card title="Session">
          <div className="space-y-2">
            <Check label="Show thinking" checked={showThinking} onChange={setShowThinking} />
            <Check label="Tools enabled" checked={toolsEnabled} onChange={setToolsEnabled} />
            <Check label="Master prompt" checked={useMaster} onChange={setUseMaster} />
          </div>
          <Btn variant="subtle" className="w-full justify-center mt-3" onClick={exportRated}>
            <Download size={12} /> Export rated data
          </Btn>
        </Card>
        <Card title="Active model">
          <div className="text-[12px] text-text-dim font-mono break-all">
            {ready ? (engine.info.checkpoint?.split(/[\\/]/).pop() ?? '—') : 'none loaded'}
          </div>
        </Card>
      </div>
    </div>
  )
}

/* ---------- message rows ---------- */

function MessageRow({ msg, showThinking, onRate }: {
  msg: ChatMessage
  showThinking: boolean
  onRate?: (r: 'good' | 'bad') => void
}) {
  if (msg.role === 'system') return null
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-accent/15 border border-accent/25 rounded-2xl rounded-br-md px-4 py-2.5">
          {msg.image && (
            <img src={`data:image/png;base64,${msg.image}`} alt="attachment"
              className="max-h-48 rounded-lg mb-2" />
          )}
          <div className="text-[13.5px] whitespace-pre-wrap">{msg.content}</div>
        </div>
      </div>
    )
  }
  if (msg.role === 'tool') {
    return (
      <div className="flex">
        <div className="max-w-[80%]">
          <ToolResultCard name={msg.name ?? 'tool'} text={msg.content} />
        </div>
      </div>
    )
  }
  // assistant — segment like the stream parser
  const segs = parseStream(msg.content)
  return (
    <div className="flex flex-col gap-1">
      <div className="max-w-[85%] space-y-2">
        {segs.map((s, i) => (
          <SegmentView key={i} seg={s} showThinking={showThinking} />
        ))}
        {!segs.length && (
          <div className="text-[13.5px] whitespace-pre-wrap">{msg.content}</div>
        )}
        {msg.tool_calls?.map((tc, i) => (
          <ToolCallCard key={`tc${i}`} name={tc.name}
            detail={JSON.stringify(tc.arguments ?? {}, null, 2)} />
        ))}
      </div>
      {onRate && (
        <div className="flex gap-1">
          <IconBtn title="Good"
            className={clsx('!w-6 !h-6', msg.rating === 'good' && '!text-ok')}
            onClick={() => onRate('good')}>
            <Star size={12} fill={msg.rating === 'good' ? 'currentColor' : 'none'} />
          </IconBtn>
          <IconBtn title="Bad"
            className={clsx('!w-6 !h-6', msg.rating === 'bad' && '!text-err')}
            onClick={() => onRate('bad')}>
            <Trash2 size={12} />
          </IconBtn>
        </div>
      )}
    </div>
  )
}

function SegmentView({ seg, showThinking }: { seg: Segment; showThinking: boolean }) {
  if (seg.kind === 'think') {
    if (!showThinking) return null
    return <ThinkCard seg={seg} />
  }
  if (seg.kind === 'tool_call') {
    return (
      <ToolCallCard
        name={seg.closed ? toolCallName(seg.text) : 'calling tool…'}
        summary={seg.closed ? toolCallSummary(seg.text) : ''}
        detail={seg.text}
        running={!seg.closed} />
    )
  }
  return <div className="text-[13.5px] whitespace-pre-wrap">{seg.text}</div>
}

/* ---------- collapsible blocks ---------- */

function ThinkCard({ seg }: { seg: Segment }) {
  // expanded while streaming, auto-collapses when the block completes;
  // a manual click overrides the default
  const [manual, setManual] = useState<boolean | null>(null)
  const open = manual ?? !seg.closed
  const setOpen = (fn: (o: boolean) => boolean) => setManual(fn(open))
  return (
    <div className="bg-think/8 border-l-2 border-think rounded-r-lg">
      <button onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-[11px] text-think cursor-pointer hover:bg-think/10 rounded-r-lg transition-colors">
        <Brain size={11} />
        {seg.closed ? 'Thought' : 'Thinking'}
        {!seg.closed && <Spinner />}
        <ChevronRight size={11}
          className={clsx('ml-auto transition-transform', open && 'rotate-90')} />
      </button>
      {open && (
        <div className="px-3 pb-2.5 text-[12px] text-text-dim whitespace-pre-wrap font-mono max-h-64 overflow-y-auto">
          {seg.text.trim()}
        </div>
      )}
    </div>
  )
}

function ToolCallCard({ name, summary, detail, running }: {
  name: string
  summary?: string
  detail: string
  running?: boolean
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="bg-panel border border-border rounded-lg max-w-full">
      <button onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:bg-panel-alt rounded-lg transition-colors text-left">
        <Wrench size={11} className="text-accent shrink-0" />
        <span className="text-[12px] font-mono shrink-0">{name}</span>
        {summary && (
          <span className="text-[10.5px] text-text-faint truncate flex-1 min-w-0">
            {summary}
          </span>
        )}
        {running && <Spinner />}
        <ChevronRight size={11}
          className={clsx('ml-auto shrink-0 text-text-faint transition-transform', open && 'rotate-90')} />
      </button>
      {open && (
        <pre className="px-3 pb-2.5 pt-1.5 text-[11.5px] font-mono text-text-dim whitespace-pre-wrap max-h-48 overflow-y-auto border-t border-border">
          {detail}
        </pre>
      )}
    </div>
  )
}

function ToolResultCard({ name, text }: { name: string; text: string }) {
  const [open, setOpen] = useState(false)
  const oneLine = text.replace(/\s+/g, ' ').trim()
  return (
    <div className="bg-panel border border-border rounded-lg max-w-full">
      <button onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:bg-panel-alt rounded-lg transition-colors text-left">
        <Wrench size={11} className="text-text-faint shrink-0" />
        <span className="text-[12px] font-mono shrink-0">{name}</span>
        <span className="text-[10.5px] text-text-faint truncate flex-1 min-w-0">
          {oneLine.slice(0, 90)}{oneLine.length > 90 ? '…' : ''}
        </span>
        <ChevronRight size={11}
          className={clsx('ml-auto shrink-0 text-text-faint transition-transform', open && 'rotate-90')} />
      </button>
      {open && (
        <pre className="px-3 pb-2.5 pt-1.5 text-[11.5px] font-mono text-text-dim whitespace-pre-wrap max-h-64 overflow-y-auto border-t border-border">
          {text}
        </pre>
      )}
    </div>
  )
}

function StreamingRow({ stream, showThinking }: {
  stream: StreamState
  showThinking: boolean
}) {
  // thinking mode leaves <think> open in the prompt — the stream starts
  // inside the block, so the leading text is thinking until </think>
  const segs = parseStream(stream.raw, { implicitThink: true })
  return (
    <div className="max-w-[85%] space-y-2">
      {segs.map((s, i) => (
        <SegmentView key={i} seg={s} showThinking={showThinking} />
      ))}
      {stream.pendingCalls > 0 && (
        <Tag kind="accent"><Spinner /> running tool…</Tag>
      )}
      {stream.toolResults.map((tr, i) => (
        <ToolResultCard key={i} name={tr.name} text={tr.result} />
      ))}
      <span className="inline-block w-1.5 h-4 bg-accent animate-pulse rounded-sm align-text-bottom" />
    </div>
  )
}
