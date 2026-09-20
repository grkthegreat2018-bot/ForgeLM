// Sticky-scroll feed container — follows new content only while the
// user is at the bottom; shows a "latest" jump pill otherwise.

import { clsx } from 'clsx'
import { ArrowDown } from 'lucide-react'
import {
  useLayoutEffect, useRef, useState, type ReactNode, type UIEvent,
} from 'react'

export function ScrollFeed({ watch, className, children }: {
  /** Value that changes when new content arrives (e.g. message count). */
  watch: unknown
  className?: string
  children: ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = useState(true)

  const onScroll = (e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80)
  }

  useLayoutEffect(() => {
    const el = ref.current
    if (el && atBottom) el.scrollTop = el.scrollHeight
  }, [watch, atBottom])

  const jump = () => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: 'smooth' })
    setAtBottom(true)
  }

  return (
    <div className="relative flex-1 min-h-0">
      <div
        ref={ref}
        onScroll={onScroll}
        className={clsx('h-full overflow-y-auto', className)}>
        {children}
      </div>
      {!atBottom && (
        <button
          onClick={jump}
          className="absolute bottom-3 left-1/2 -translate-x-1/2 inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-panel border border-border-hi text-[11px] text-text-dim hover:text-text hover:border-accent/60 shadow-card cursor-pointer transition-colors fade-in">
          <ArrowDown size={11} /> latest
        </button>
      )}
    </div>
  )
}
