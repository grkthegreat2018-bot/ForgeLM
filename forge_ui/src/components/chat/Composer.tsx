// Shared composer — autosizing textarea + footer row (chips left,
// meta + send/stop right). Used by Chat and Agent pages.

import { Send, Square } from 'lucide-react'
import { useLayoutEffect, useRef, type ReactNode } from 'react'
import { Btn } from '../ui'

export function Composer({ value, onChange, onSubmit, onStop, busy,
  disabled, placeholder, chips, meta, sendLabel = 'Send', autoFocus,
  submitTitle }: {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onStop?: () => void
  busy?: boolean
  disabled?: boolean
  placeholder?: string
  chips?: ReactNode
  meta?: ReactNode
  sendLabel?: string
  autoFocus?: boolean
  submitTitle?: string
}) {
  const ref = useRef<HTMLTextAreaElement>(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [value])

  return (
    <div className="bg-panel border border-border rounded-xl shadow-card focus-within:border-accent/50 transition-colors">
      <textarea
        ref={ref}
        value={value}
        autoFocus={autoFocus}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            onSubmit()
          }
        }}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        className="w-full bg-transparent px-3.5 pt-3 pb-1.5 text-[13.5px] text-text placeholder:text-text-faint resize-none focus:outline-none disabled:opacity-50" />
      <div className="flex items-center gap-1 px-2 pb-2">
        {chips}
        <div className="ml-auto flex items-center gap-2">
          {meta}
          {busy && onStop ? (
            <Btn variant="danger" onClick={onStop} title="Stop generation">
              <Square size={12} /> Stop
            </Btn>
          ) : (
            <Btn variant="primary" onClick={onSubmit} title={submitTitle}
              disabled={!value.trim() || disabled || busy}>
              <Send size={13} /> {sendLabel}
            </Btn>
          )}
        </div>
      </div>
    </div>
  )
}
