// Shared message blocks for Chat + Agent feeds:
// ThinkCard, ToolActivityCard (call + result merged), SegmentView,
// UserBubble, CopyBtn.

import { clsx } from 'clsx'
import {
  Brain, Check, ChevronRight, Copy, Wrench,
} from 'lucide-react'
import { useState } from 'react'
import {
  parseStream, toolCallName, toolCallSummary, toolResultSummary,
  type Segment,
} from '../../lib/streamParse'
import { Spinner } from '../ui'
import { Markdown } from './Markdown'

export function CopyBtn({ text, className }: {
  text: string
  className?: string
}) {
  const [ok, setOk] = useState(false)
  return (
    <button
      title="Copy"
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setOk(true)
          setTimeout(() => setOk(false), 1200)
        })
      }}
      className={clsx(
        'inline-flex items-center justify-center w-6 h-6 rounded-md text-text-faint hover:text-text hover:bg-panel-alt transition-colors cursor-pointer',
        className)}>
      {ok ? <Check size={12} className="text-ok" /> : <Copy size={12} />}
    </button>
  )
}

export function ThinkCard({ text, streaming, className }: {
  text: string
  streaming: boolean
  className?: string
}) {
  // expanded while streaming, auto-collapses on close; click overrides
  const [manual, setManual] = useState<boolean | null>(null)
  const open = manual ?? streaming
  return (
    <div className={clsx(
      'bg-think/8 border-l-2 border-think rounded-r-lg overflow-hidden', className)}>
      <button
        onClick={() => setManual(!open)}
        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-[11px] text-think cursor-pointer hover:bg-think/10 transition-colors">
        <Brain size={11} />
        {streaming ? 'Thinking' : 'Thought'}
        {streaming && <Spinner />}
        <ChevronRight size={11}
          className={clsx('ml-auto transition-transform', open && 'rotate-90')} />
      </button>
      {open && (
        <div className="px-3 pb-2.5 text-[12px] text-text-dim whitespace-pre-wrap font-mono max-h-64 overflow-y-auto">
          {text.trim()}
        </div>
      )}
    </div>
  )
}

export interface ToolResultView {
  name?: string
  ok?: boolean
  text: string
  elapsed_s?: number
}

/** One tool invocation — call args + (optional) execution result merged. */
export function ToolActivityCard({ name, args, summary, result, running }: {
  name: string
  args: string
  summary?: string
  result?: ToolResultView
  running?: boolean
}) {
  const [open, setOpen] = useState(false)
  if (!summary) {
    try { summary = toolCallSummary(args) } catch { /* raw */ }
  }
  const preview = result
    ? (toolResultSummary(result.text)
      || result.text.replace(/\s+/g, ' ').trim().slice(0, 80))
    : (summary ?? '')
  const failed = result !== undefined && result.ok === false
  return (
    <div className={clsx(
      'bg-tool-bg border rounded-lg max-w-full overflow-hidden',
      failed ? 'border-err/40' : 'border-tool-border')}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:bg-panel-alt rounded-lg transition-colors text-left">
        <Wrench size={11} className={clsx('shrink-0',
          failed ? 'text-err' : 'text-accent')} />
        <span className="text-[12px] font-mono shrink-0">{name}</span>
        {running && <Spinner />}
        {!running && result !== undefined && (
          <span className={clsx('w-1.5 h-1.5 rounded-full shrink-0',
            failed ? 'bg-err' : 'bg-ok')} />
        )}
        {preview && (
          <span className="text-[10.5px] text-text-faint truncate flex-1 min-w-0">
            {preview}
          </span>
        )}
        {result?.elapsed_s != null && (
          <span className="text-[10.5px] text-text-faint shrink-0 tabular-nums">
            {result.elapsed_s.toFixed(2)}s
          </span>
        )}
        <ChevronRight size={11}
          className={clsx('shrink-0 text-text-faint transition-transform',
            open && 'rotate-90', !preview && 'ml-auto')} />
      </button>
      {open && (
        <div className="border-t border-border">
          <pre className="px-3 py-2 text-[11.5px] font-mono text-text-dim whitespace-pre-wrap max-h-56 overflow-y-auto">
            {pretty(args)}
          </pre>
          {result && (
            <pre className={clsx(
              'px-3 py-2 text-[11.5px] font-mono whitespace-pre-wrap max-h-64 overflow-y-auto border-t border-border',
              failed ? 'text-err/90' : 'text-text-dim')}>
              {result.text}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

function pretty(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}

export function UserBubble({ text }: { text: string }) {
  // w-full is required: inside an `items-end` flex column the wrapper would
  // otherwise shrink-to-fit, and max-w-[75%] on the bubble resolves against
  // that shrunk width — collapsing short messages to ~1ch columns.
  return (
    <div className="flex w-full justify-end">
      <div className="max-w-[75%] bg-bubble-user border border-bubble-user-border rounded-2xl rounded-br-md px-4 py-2.5">
        <div className="text-[13.5px] whitespace-pre-wrap break-words">{text}</div>
      </div>
    </div>
  )
}

/** One parse segment → the right card. `result` pairs a tool_call segment
 *  with the tool_result that answered it (by call order). */
export function SegmentView({ seg, showThinking, result }: {
  seg: Segment
  showThinking: boolean
  result?: ToolResultView
}) {
  if (seg.kind === 'think') {
    if (!showThinking) return null
    return <ThinkCard text={seg.text} streaming={!seg.closed} />
  }
  if (seg.kind === 'tool_call') {
    return (
      <ToolActivityCard
        name={seg.closed ? toolCallName(seg.text) : 'calling tool…'}
        args={seg.text}
        result={result}
        running={!seg.closed} />
    )
  }
  return <Markdown text={seg.text} />
}

/** Render a whole raw assistant text as segmented blocks. */
export function SegmentedBody({ raw, showThinking, results }: {
  raw: string
  showThinking: boolean
  results?: ToolResultView[]
}) {
  const segs = parseStream(raw)
  let toolIdx = 0
  return (
    <div className="space-y-2 min-w-0">
      {segs.map((s, i) => {
        const res = s.kind === 'tool_call' ? results?.[toolIdx++] : undefined
        return <SegmentView key={i} seg={s} showThinking={showThinking} result={res} />
      })}
      {!segs.length && raw.trim() && <Markdown text={raw} />}
    </div>
  )
}
