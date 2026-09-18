// Collapsible navigation sidebar — brand, grouped sections, live status.

import { clsx } from 'clsx'
import {
  Activity, Bot, Box, ChevronsLeft, ChevronsRight, Cpu,
  FileText, Gauge, Hammer, Layers, LayoutDashboard, ListTodo,
  MessageSquare, Play, Search, Sparkles, TrendingUp, Zap,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { useForge } from '../lib/store'
import { useUi } from '../lib/ui'

export interface PageDef {
  path: string
  label: string
  icon: React.ReactNode
  section: string
  subtitle: string
}

export const PAGES: PageDef[] = [
  { path: '/agent', label: 'Agent', icon: <Bot size={17} />, section: 'Agentic', subtitle: 'Autonomous tool-using agent' },
  { path: '/chat', label: 'Chat', icon: <MessageSquare size={17} />, section: 'Agentic', subtitle: 'Chat studio' },
  { path: '/generations', label: 'Generations', icon: <Sparkles size={17} />, section: 'Agentic', subtitle: 'Generation playground' },
  { path: '/', label: 'Dashboard', icon: <LayoutDashboard size={17} />, section: 'Model', subtitle: 'System overview' },
  { path: '/engine', label: 'Engine', icon: <Cpu size={17} />, section: 'Model', subtitle: 'Engine control + tuning' },
  { path: '/models', label: 'Models', icon: <Box size={17} />, section: 'Model', subtitle: 'Checkpoints & configs' },
  { path: '/lora', label: 'LoRA', icon: <Layers size={17} />, section: 'Model', subtitle: 'Adapters & merging' },
  { path: '/finetune', label: 'Fine-Tune', icon: <Hammer size={17} />, section: 'Train', subtitle: 'SFT launcher' },
  { path: '/selfplay', label: 'Self-Play', icon: <Zap size={17} />, section: 'Train', subtitle: 'Autonomous training loop' },
  { path: '/training', label: 'Training Live', icon: <TrendingUp size={17} />, section: 'Train', subtitle: 'Run telemetry' },
  { path: '/launch', label: 'Launch', icon: <Play size={17} />, section: 'System', subtitle: 'Process presets' },
  { path: '/tasks', label: 'Tasks', icon: <ListTodo size={17} />, section: 'System', subtitle: 'Process manager' },
  { path: '/compute', label: 'Compute', icon: <Gauge size={17} />, section: 'System', subtitle: 'GPU monitor' },
  { path: '/logs', label: 'Logs', icon: <FileText size={17} />, section: 'System', subtitle: 'Log viewer' },
]

const SECTIONS = ['Agentic', 'Model', 'Train', 'System']

export function Sidebar() {
  const collapsed = useUi((s) => s.collapsed)
  const toggle = useUi((s) => s.toggleCollapsed)
  const [query, setQuery] = useState('')
  const engine = useForge((s) => s.engine)
  const gpu = useForge((s) => s.gpu)
  const connected = useForge((s) => s.connected)

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return PAGES
    return PAGES.filter((p) => p.label.toLowerCase().includes(q))
  }, [query])

  const engKind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  return (
    <aside
      className={clsx(
        'flex flex-col h-full bg-bg-alt border-r border-border transition-all duration-200 shrink-0',
        collapsed ? 'w-[60px]' : 'w-[248px]')}>
      {/* brand */}
      <div className={clsx('flex items-center gap-2.5 pt-4 pb-3',
        collapsed ? 'px-3 justify-center' : 'px-4')}>
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-grad-a to-grad-b flex items-center justify-center shrink-0">
          <Activity size={17} className="text-on-accent" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <div className="text-[14px] font-semibold leading-tight">ForgeAI</div>
            <div className="text-[9.5px] uppercase tracking-[0.14em] text-text-faint">
              Control Center
            </div>
          </div>
        )}
        {!collapsed && (
          <button
            onClick={toggle}
            className="ml-auto text-text-faint hover:text-text transition-colors cursor-pointer"
            title="Collapse (Ctrl+B)">
            <ChevronsLeft size={16} />
          </button>
        )}
      </div>
      {collapsed && (
        <button
          onClick={toggle}
          className="mx-auto mb-1 text-text-faint hover:text-text transition-colors cursor-pointer"
          title="Expand (Ctrl+B)">
          <ChevronsRight size={16} />
        </button>
      )}

      {/* search */}
      {!collapsed && (
        <div className="px-3 pb-2">
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-faint" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search pages…"
              className="w-full bg-input border border-border rounded-lg pl-7 pr-2 py-1.5 text-[12px] text-text placeholder:text-text-faint focus:outline-none focus:border-accent/60" />
          </div>
        </div>
      )}

      {/* nav */}
      <nav className="flex-1 overflow-y-auto px-2 pb-2">
        {SECTIONS.map((sec) => {
          const items = filtered.filter((p) => p.section === sec)
          if (!items.length) return null
          return (
            <div key={sec} className="mt-2">
              {!collapsed && (
                <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-text-faint">
                  {sec}
                </div>
              )}
              {items.map((p) => (
                <NavLink
                  key={p.path}
                  to={p.path}
                  end={p.path === '/'}
                  title={collapsed ? p.label : undefined}
                  className={({ isActive }) => clsx(
                    'flex items-center gap-2.5 rounded-lg mb-0.5 text-[13px] transition-all duration-100 slide-in-left',
                    collapsed ? 'justify-center px-0 py-2' : 'px-2.5 py-1.5',
                    isActive
                      ? 'bg-accent/15 text-accent-hi border border-accent/25'
                      : 'text-text-dim border border-transparent hover:bg-panel-alt hover:text-text',
                  )}>
                  <span className="shrink-0">{p.icon}</span>
                  {!collapsed && <span className="truncate">{p.label}</span>}
                </NavLink>
              ))}
            </div>
          )
        })}
      </nav>

      {/* status footer */}
      <div className={clsx('border-t border-border px-3 py-2.5',
        collapsed && 'flex flex-col items-center gap-1.5')}>
        {collapsed ? (
          <>
            <span
              title={`engine: ${engine.state}`}
              className={clsx('w-2 h-2 rounded-full',
                engKind === 'ok' ? 'bg-ok' : engKind === 'warn' ? 'bg-warn pulse-dot'
                : engKind === 'err' ? 'bg-err' : 'bg-text-faint')} />
            <span
              title={`GPU ${gpu.vram_used_mb}/${gpu.vram_total_mb} MB`}
              className="w-2 h-2 rounded-full bg-accent" />
          </>
        ) : (
          <>
            <div className="flex items-center gap-2 text-[11.5px] text-text-dim">
              <span className={clsx('w-2 h-2 rounded-full',
                engKind === 'ok' ? 'bg-ok' : engKind === 'warn' ? 'bg-warn pulse-dot'
                : engKind === 'err' ? 'bg-err' : 'bg-text-faint')} />
              <span className="truncate">
                {engine.state === 'ready'
                  ? (engine.info.config_name || 'engine ready')
                  : engine.state}
              </span>
              <span className="ml-auto flex items-center gap-1">
                <span className={clsx('w-1.5 h-1.5 rounded-full',
                  connected ? 'bg-ok' : 'bg-err')} />
                {connected ? 'live' : 'off'}
              </span>
            </div>
            {gpu.available && (
              <div className="mt-1.5">
                <div className="flex justify-between text-[10.5px] text-text-faint mb-0.5">
                  <span>VRAM</span>
                  <span>{gpu.vram_used_mb}/{Math.round(gpu.vram_total_mb / 1024)}GB</span>
                </div>
                <div className="h-1 rounded-full bg-panel-alt overflow-hidden">
                  <div
                    className={clsx('h-full rounded-full',
                      gpu.vram_pct > 85 ? 'bg-err' : 'bg-accent')}
                    style={{ width: `${gpu.vram_pct}%` }} />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  )
}
