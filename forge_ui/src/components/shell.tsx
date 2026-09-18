// Shell chrome: PageHeader, ApprovalPopups, CommandPalette.

import { Check, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useForge } from '../lib/store'
import { useUi } from '../lib/ui'
import { Btn, StatusDot, Tag } from './ui'
import { PAGES } from './Sidebar'

export function PageHeader({ title, subtitle, actions }: {
  title: string
  subtitle?: string
  actions?: React.ReactNode
}) {
  const engine = useForge((s) => s.engine)
  const progress = useForge((s) => s.engineProgress)
  const [clock, setClock] = useState('')

  useEffect(() => {
    const t = setInterval(
      () => setClock(new Date().toLocaleTimeString()), 1000)
    return () => clearInterval(t)
  }, [])

  const kind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  return (
    <header className="flex items-center gap-4 h-14 px-6 border-b border-border shrink-0">
      <div className="min-w-0">
        <h1 className="text-[15px] font-semibold leading-tight truncate">{title}</h1>
        {subtitle && (
          <div className="text-[11px] text-text-faint leading-tight truncate">
            {progress ?? subtitle}
          </div>
        )}
      </div>
      <div className="ml-auto flex items-center gap-3">
        {actions}
        <Tag kind={kind}>
          <StatusDot kind={kind} pulse={engine.state === 'loading'} />
          {engine.state === 'ready'
            ? (engine.info.config_name ?? 'ready')
            : engine.state}
        </Tag>
        <span className="text-[11.5px] text-text-faint font-mono tabular-nums">
          {clock}
        </span>
      </div>
    </header>
  )
}

/** Floating approval requests (backup restore / library install). */
export function ApprovalPopups() {
  const approvals = useForge((s) => s.approvals)
  const respond = useForge((s) => s.respondApproval)
  if (!approvals.length) return null
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {approvals.map((a) => (
        <div key={a.id}
          className="bg-panel border border-warn/50 rounded-card shadow-card p-4 slide-in-left">
          <div className="flex items-center gap-2 mb-1.5">
            <Tag kind="warn">approval needed</Tag>
            <span className="text-[13px] font-medium">{a.kind}</span>
          </div>
          <div className="text-[12px] text-text-dim mb-3 break-all">
            {(a.detail as { warning?: string }).warning ??
              JSON.stringify(a.detail)}
          </div>
          <div className="flex gap-2">
            <Btn variant="primary" onClick={() => respond(a.id, true)}>
              <Check size={13} /> Approve
            </Btn>
            <Btn variant="danger" onClick={() => respond(a.id, false)}>
              <X size={13} /> Deny
            </Btn>
          </div>
        </div>
      ))}
    </div>
  )
}

/** Ctrl+K palette — jump to any page. */
export function CommandPalette() {
  const open = useUi((s) => s.paletteOpen)
  const setOpen = useUi((s) => s.setPalette)
  const [q, setQ] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    if (open) setQ('')
  }, [open])

  if (!open) return null
  const ql = q.toLowerCase()
  const matches = PAGES.filter(
    (p) => p.label.toLowerCase().includes(ql)
        || p.subtitle.toLowerCase().includes(ql))

  return (
    <div
      className="fixed inset-0 z-50 bg-black/50 flex items-start justify-center pt-[18vh] fade-in"
      onClick={() => setOpen(false)}>
      <div
        className="w-[480px] bg-panel border border-border-hi rounded-card shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}>
        <input
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') setOpen(false)
            if (e.key === 'Enter' && matches.length) {
              navigate(matches[0].path)
              setOpen(false)
            }
          }}
          placeholder="Jump to page…"
          className="w-full bg-transparent px-4 py-3 text-[14px] text-text placeholder:text-text-faint focus:outline-none border-b border-border" />
        <div className="max-h-72 overflow-y-auto py-1">
          {matches.map((p) => (
            <button
              key={p.path}
              onClick={() => { navigate(p.path); setOpen(false) }}
              className="w-full flex items-center gap-3 px-4 py-2 text-left text-[13px] text-text-dim hover:bg-panel-alt hover:text-text transition-colors cursor-pointer">
              <span className="text-text-faint">{p.icon}</span>
              <span>{p.label}</span>
              <span className="ml-auto text-[11px] text-text-faint">{p.subtitle}</span>
            </button>
          ))}
          {!matches.length && (
            <div className="px-4 py-6 text-center text-[12px] text-text-faint">
              no matches
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
