// Incremental stream segmentation — React port of the Qt _StreamParser.
// Splits accumulated raw model output into display segments: thinking
// blocks, assistant body text, and tool_call blocks (the JSON is hidden
// behind a card, never shown raw). ChatML / special tokens are stripped.
//
// Jamba quirks handled here:
//  - thinking mode: the chat template ends the prompt with an *open*
//    `<think>` block, so model output starts mid-block — the first
//    `</think>` is an orphan closer, not a matched pair.
//  - tool calls may arrive as bare {"name": ...} JSON without
//    <tool_call> markers (the server-side parser accepts both forms).

export type SegKind = 'think' | 'body' | 'tool_call'

export interface Segment {
  kind: SegKind
  text: string
  closed: boolean   // marker block finished?
}

export interface ParseOpts {
  /** The prompt left a <think> block open — treat the leading region as
   *  thinking until `</think>` appears (streaming display hint). */
  implicitThink?: boolean
}

const SPECIAL_TOKENS = [
  '<|im_start|>', '<|im_end|>', '<|startoftext|>', '<|endoftext|>',
  '<tool_response>', '</tool_response>',
]

const OPENERS = ['<think>', '<tool_call>'] as const
const CLOSERS: Record<string, string> = {
  '<think>': '</think>',
  '<tool_call>': '</tool_call>',
}
// Closers that may appear without their opener (prompt-opened blocks).
const ORPHAN_KIND: Record<string, SegKind> = {
  '</think>': 'think',
  '</tool_call>': 'tool_call',
}
// Bare-JSON tool-call hint (mirrors qwen_parse_tool_calls phase 2).
const JSON_HINT = /\{\s*"name"\s*:/g

export function stripSpecial(text: string): string {
  let out = text
  for (const t of SPECIAL_TOKENS) out = out.split(t).join('')
  return out
}

/** End index (exclusive) of the balanced {...} starting at `start`,
 *  or -1 if the text ends before the braces balance. */
function balancedEnd(text: string, start: number): number {
  let depth = 0, inStr = false, esc = false
  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (inStr) {
      if (esc) esc = false
      else if (ch === '\\') esc = true
      else if (ch === '"') inStr = false
      continue
    }
    if (ch === '"') inStr = true
    else if (ch === '{') depth++
    else if (ch === '}') {
      depth--
      if (!depth) return i + 1
    }
  }
  return -1
}

/** Parse accumulated raw text into segments (re-runs each chunk). */
export function parseStream(raw: string, opts: ParseOpts = {}): Segment[] {
  const segs: Segment[] = []
  let rest = stripSpecial(raw)

  // leading text is a pending think block when the prompt opened one
  // and no closer has been consumed yet
  const leadKind = (): SegKind =>
    opts.implicitThink && segs.length === 0 ? 'think' : 'body'

  while (rest.length) {
    // find the earliest structural event: opener | orphan closer | bare JSON
    let firstIdx = -1
    let mode: 'open' | 'close' | 'json' = 'open'
    let mark = ''
    for (const o of OPENERS) {
      const i = rest.indexOf(o)
      if (i >= 0 && (firstIdx < 0 || i < firstIdx)) {
        firstIdx = i; mode = 'open'; mark = o
      }
    }
    for (const c of Object.keys(ORPHAN_KIND)) {
      const i = rest.indexOf(c)
      if (i >= 0 && (firstIdx < 0 || i < firstIdx)) {
        firstIdx = i; mode = 'close'; mark = c
      }
    }

    let jsonEnd = -1
    JSON_HINT.lastIndex = 0
    const jm = JSON_HINT.exec(rest)
    if (jm && (firstIdx < 0 || jm.index < firstIdx)) {
      const e = balancedEnd(rest, jm.index)
      jsonEnd = e // -1 → unbalanced tail = in-progress call
      firstIdx = jm.index; mode = 'json'
    }

    if (firstIdx < 0) {
      if (rest.trim()) {
        const think = leadKind() === 'think'
        segs.push({ kind: think ? 'think' : 'body', text: rest, closed: !think })
      }
      break
    }

    if (mode === 'close') {
      // orphan closer — the text before it is an implicit segment
      const inner = rest.slice(0, firstIdx)
      if (inner.trim()) {
        segs.push({ kind: ORPHAN_KIND[mark], text: inner, closed: true })
      }
      rest = rest.slice(firstIdx + mark.length)
      continue
    }

    const before = rest.slice(0, firstIdx)
    if (before.trim()) {
      const think = leadKind() === 'think'
      segs.push({ kind: think ? 'think' : 'body', text: before, closed: !think })
    }

    if (mode === 'json') {
      const closed = jsonEnd > 0
      segs.push({
        kind: 'tool_call',
        text: rest.slice(firstIdx, closed ? jsonEnd : rest.length),
        closed,
      })
      rest = closed ? rest.slice(jsonEnd) : ''
      continue
    }

    // mode === 'open'
    const closer = CLOSERS[mark]
    const after = rest.slice(firstIdx + mark.length)
    const closeIdx = after.indexOf(closer)
    if (closeIdx < 0) {
      segs.push({
        kind: mark === '<think>' ? 'think' : 'tool_call',
        text: after, closed: false,
      })
      break
    }
    segs.push({
      kind: mark === '<think>' ? 'think' : 'tool_call',
      text: after.slice(0, closeIdx), closed: true,
    })
    rest = after.slice(closeIdx + closer.length)
  }
  return segs
}

/** Extract the tool name from a tool_call segment's JSON payload. */
export function toolCallName(text: string): string {
  try {
    const j = JSON.parse(text)
    return j.name || 'tool'
  } catch {
    const m = text.match(/"name"\s*:\s*"([^"]+)"/)
    return m?.[1] ?? 'tool'
  }
}

/** One-line `k=v` summary of a tool call's arguments (why it was called). */
export function toolCallSummary(text: string): string {
  try {
    const j = JSON.parse(text)
    const args = j.arguments ?? j.args ?? {}
    const parts = Object.entries(args).slice(0, 3).map(([k, v]) => {
      const s = typeof v === 'string' ? v : JSON.stringify(v)
      return `${k}=${s.length > 40 ? s.slice(0, 40) + '…' : s}`
    })
    return parts.join('   ')
  } catch {
    return ''
  }
}
