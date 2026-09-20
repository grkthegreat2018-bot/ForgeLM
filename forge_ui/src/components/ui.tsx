// Shared UI primitives — the ForgeAI design system.
// Every page builds from these so restyling means editing one file.

import { clsx } from 'clsx'
import { Loader2 } from 'lucide-react'
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'

export function Card({ children, className, title, actions }: {
  children: ReactNode
  className?: string
  title?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className={clsx('bg-panel border border-border rounded-card shadow-card', className)}>
      {(title || actions) && (
        <div className="flex items-center justify-between px-4 pt-3 pb-1">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-text-faint">{title}</div>
          <div className="flex items-center gap-2">{actions}</div>
        </div>
      )}
      <div className="px-4 py-3">{children}</div>
    </div>
  )
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-widest text-text-faint px-1 pt-4 pb-1.5">
      {children}
    </div>
  )
}

type BtnVariant = 'primary' | 'ghost' | 'danger' | 'subtle' | 'outline'
export function Btn({ variant = 'ghost', className, children, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant }) {
  const styles: Record<BtnVariant, string> = {
    primary:
      'bg-gradient-to-r from-grad-a to-grad-b text-on-accent font-medium hover:brightness-110 disabled:opacity-40',
    ghost:
      'text-text-dim hover:text-text hover:bg-panel-alt disabled:opacity-40',
    danger:
      'text-err hover:bg-err/10 border border-transparent hover:border-err/40 disabled:opacity-40',
    subtle:
      'bg-panel-alt text-text-dim hover:text-text border border-border hover:border-border-hi disabled:opacity-40',
    outline:
      'border border-border-hi text-text hover:border-accent hover:text-accent-hi disabled:opacity-40',
  }
  return (
    <button
      className={clsx(
        'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[13px] transition-all duration-100 cursor-pointer select-none',
        styles[variant], className)}
      {...rest}>
      {children}
    </button>
  )
}

export function IconBtn({ className, children, title, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      title={title}
      className={clsx(
        'inline-flex items-center justify-center w-7 h-7 rounded-md text-text-dim hover:text-text hover:bg-panel-alt transition-colors cursor-pointer',
        className)}
      {...rest}>
      {children}
    </button>
  )
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={clsx(
        'bg-input border border-border rounded-lg px-3 py-1.5 text-[13px] text-text placeholder:text-text-faint',
        'focus:outline-none focus:border-accent/60 transition-colors',
        className)}
      {...rest} />
  )
}

export function Textarea({ className, ref, ...rest }:
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
  & { ref?: React.Ref<HTMLTextAreaElement> }) {
  return (
    <textarea
      ref={ref}
      className={clsx(
        'bg-input border border-border rounded-lg px-3 py-2 text-[13px] text-text placeholder:text-text-faint',
        'focus:outline-none focus:border-accent/60 transition-colors resize-y',
        className)}
      {...rest} />
  )
}

export function Select({ className, children, ...rest }:
  SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={clsx(
        'bg-input border border-border rounded-lg px-2.5 py-1.5 text-[13px] text-text',
        'focus:outline-none focus:border-accent/60 cursor-pointer',
        className)}
      {...rest}>
      {children}
    </select>
  )
}

/** Labeled slider + numeric readout — for sampling params. */
export function Slider({ label, value, onChange, min, max, step, fmt, hint }: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  fmt?: (v: number) => string
  hint?: string
}) {
  return (
    <div className="select-none" title={hint}>
      <div className="flex items-baseline justify-between mb-0.5">
        <span className="text-[11.5px] text-text-dim">{label}</span>
        <span className="text-[11.5px] font-mono text-text tabular-nums">
          {fmt ? fmt(value) : value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 rounded-full bg-panel-alt appearance-none cursor-pointer accent-[#8aa3ff]" />
    </div>
  )
}

/** Vertical label + control wrapper for settings panels. */
export function Field({ label, hint, children }: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div>
      <div className="text-[11.5px] text-text-dim mb-1" title={hint}>{label}</div>
      {children}
    </div>
  )
}

/** Toggle chip — composer quick toggles (Tools / Thinking / …). */
export function Chip({ on, onClick, title, children }: {
  on: boolean
  onClick: () => void
  title?: string
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={clsx(
        'inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer select-none',
        on
          ? 'bg-accent/15 text-accent-hi border-accent/30'
          : 'text-text-faint border-transparent hover:text-text-dim hover:bg-panel-alt')}>
      {children}
    </button>
  )
}

export function NumInput({ label, value, onChange, min, max, step, className }: {
  label?: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  className?: string
}) {
  return (
    <label className={clsx('flex items-center gap-2 text-[12px] text-text-dim', className)}>
      {label && <span className="whitespace-nowrap">{label}</span>}
      <input
        type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="bg-input border border-border rounded-lg px-2 py-1 w-20 text-[12.5px] text-text focus:outline-none focus:border-accent/60" />
    </label>
  )
}

export function Check({ label, checked, onChange, className, title }: {
  label: ReactNode
  checked: boolean
  onChange: (v: boolean) => void
  className?: string
  title?: string
}) {
  return (
    <label title={title}
      className={clsx('flex items-center gap-2 text-[12.5px] text-text-dim cursor-pointer select-none', className)}>
      <input
        type="checkbox" checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-3.5 h-3.5 rounded accent-[#8aa3ff] cursor-pointer" />
      {label}
    </label>
  )
}

type TagKind = 'ok' | 'warn' | 'err' | 'idle' | 'accent' | 'think'
export function Tag({ kind = 'idle', children, className }: {
  kind?: TagKind
  children: ReactNode
  className?: string
}) {
  const styles: Record<TagKind, string> = {
    ok: 'bg-ok/15 text-ok border-ok/30',
    warn: 'bg-warn/15 text-warn border-warn/30',
    err: 'bg-err/15 text-err border-err/30',
    idle: 'bg-panel-alt text-text-dim border-border',
    accent: 'bg-accent/15 text-accent-hi border-accent/30',
    think: 'bg-think/15 text-think border-think/30',
  }
  return (
    <span className={clsx(
      'inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium',
      styles[kind], className)}>
      {children}
    </span>
  )
}

export function StatusDot({ kind = 'idle', pulse }: {
  kind?: TagKind
  pulse?: boolean
}) {
  const colors: Record<TagKind, string> = {
    ok: 'bg-ok', warn: 'bg-warn', err: 'bg-err',
    idle: 'bg-text-faint', accent: 'bg-accent', think: 'bg-think',
  }
  return (
    <span className={clsx('inline-block w-2 h-2 rounded-full',
      colors[kind], pulse && 'pulse-dot')} />
  )
}

export function Progress({ value, className, danger }: {
  value: number
  className?: string
  danger?: boolean
}) {
  const pct = Math.max(0, Math.min(100, value))
  return (
    <div className={clsx('h-1.5 rounded-full bg-panel-alt overflow-hidden', className)}>
      <div
        className={clsx('h-full rounded-full transition-all duration-300',
          danger || pct > 85 ? 'bg-err' : 'bg-gradient-to-r from-grad-a to-grad-b')}
        style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 size={14} className={clsx('animate-spin text-text-dim', className)} />
}

export function EmptyState({ icon, title, desc, action }: {
  icon?: ReactNode
  title: string
  desc?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center py-14 text-center fade-in">
      {icon && <div className="text-text-faint mb-3">{icon}</div>}
      <div className="text-[15px] font-medium text-text-dim">{title}</div>
      {desc && <div className="text-[12.5px] text-text-faint mt-1 max-w-sm">{desc}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function KvRow({ k, v, mono }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-[12px] text-text-faint shrink-0">{k}</span>
      <span className={clsx('text-[12.5px] text-text text-right truncate',
        mono && 'font-mono text-[11.5px]')}>{v}</span>
    </div>
  )
}

export function MetricCard({ title, value, unit, spark, color }: {
  title: string
  value: ReactNode
  unit?: string
  spark?: number[]
  color?: string
}) {
  return (
    <div className="bg-panel border border-border rounded-card shadow-card px-4 py-3 min-w-0">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-text-faint">{title}</div>
      <div className="flex items-end justify-between gap-2 mt-1">
        <div className="text-[22px] font-semibold leading-none truncate">
          {value}
          {unit && <span className="text-[12px] font-normal text-text-dim ml-1">{unit}</span>}
        </div>
        {spark && spark.length > 1 && (
          <Sparkline data={spark} color={color} className="shrink-0 mb-0.5" />
        )}
      </div>
    </div>
  )
}

export function Sparkline({ data, width = 72, height = 20, color = '#8aa3ff', className }: {
  data: number[]
  width?: number
  height?: number
  color?: string
  className?: string
}) {
  if (data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const pts = data.map((v, i) =>
    `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * (height - 2) - 1}`)
    .join(' ')
  return (
    <svg width={width} height={height} className={className}>
      <polyline points={pts} fill="none" stroke={color}
        strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}
