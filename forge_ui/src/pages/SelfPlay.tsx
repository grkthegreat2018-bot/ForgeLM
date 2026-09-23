// Self-Play — autonomous training loop monitor + launcher.
// Polls /api/selfplay/status every 3s (status.json + events.jsonl tail).

import { Play, RefreshCw, Square, Zap } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { SelfPlayEvent } from '../lib/types'
import {
  Btn, Card, EmptyState, IconBtn, KvRow, MetricCard,
  NumInput, Progress, Select, Spinner, Tag,
} from '../components/ui'

interface SelfPlayStatus {
  status: Record<string, unknown>
  events: SelfPlayEvent[]
  heartbeat_age_s: number | null
  heartbeat_stalled: boolean | null
  topics: string[]
  flux_models?: string[]
}

type TagKind = 'ok' | 'warn' | 'err' | 'idle' | 'accent' | 'think'

const eventKind = (e: SelfPlayEvent): TagKind => {
  const k = String(e.kind ?? '').toLowerCase()
  const lvl = String(e.level ?? '').toLowerCase()
  if (e.success === false || /error|fail/.test(k) || lvl === 'error') return 'err'
  if (lvl === 'warn' || lvl === 'warning') return 'warn'
  if (/success|pass|done/.test(k) || e.success === true) return 'ok'
  if (/train|round|epoch|phase|task|curriculum/.test(k)) return 'accent'
  return 'idle'
}

const fmtScalar = (v: unknown): string => {
  if (v == null) return '—'
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'number') {
    return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/\.?0+$/, '')
  }
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

const fmtTime = (ts?: number) => {
  if (!ts) return '--:--:--'
  const d = new Date(ts * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

const summarize = (e: SelfPlayEvent) =>
  Object.entries(e)
    .filter(([k]) => k !== 'ts' && k !== 'kind')
    .map(([k, v]) =>
      `${k}=${typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v)}`)
    .join('  ')

const METRICS: { key: string; label: string; unit?: string }[] = [
  { key: 'tasks_per_min', label: 'tasks/min' },
  { key: 'gen_tok_s', label: 'gen tok/s' },
  { key: 'epoch_successes', label: 'epoch wins' },
  { key: 'eta_s', label: 'eta', unit: 's' },
  { key: 'step', label: 'step' },
  { key: 'loss', label: 'loss' },
]

export default function SelfPlay() {
  const [status, setStatus] = useState<Record<string, unknown>>({})
  const [events, setEvents] = useState<SelfPlayEvent[]>([])
  const [hbAge, setHbAge] = useState<number | null>(null)
  const [stalled, setStalled] = useState(false)
  const [topics, setTopics] = useState<string[]>([])
  const [fluxModels, setFluxModels] = useState<string[]>([])
  const [mode, setMode] = useState<'sft' | 'grpo' | 'flux'>('sft')
  const [fluxCkpt, setFluxCkpt] = useState('')
  const [fluxGroup, setFluxGroup] = useState(4)
  const [fluxWorkers, setFluxWorkers] = useState(4)
  const [topic, setTopic] = useState('python_algorithms')
  const [epochs, setEpochs] = useState(3)
  const [tasksPerEpoch, setTasksPerEpoch] = useState(50)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const feedRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    try {
      const r = await api.get<SelfPlayStatus>('/api/selfplay/status')
      setStatus(r.status ?? {})
      setEvents(r.events ?? [])
      setHbAge(r.heartbeat_age_s)
      setStalled(r.heartbeat_stalled === true)
      setTopics(r.topics ?? [])
      setFluxModels(r.flux_models ?? [])
      setError('')
    } catch (e) { setError(String(e)) }
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [load])

  // newest events land at the bottom — keep the feed pinned there
  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight })
  }, [events.length])

  const running = String(status.status ?? '') === 'running'

  const start = async () => {
    if (busy) return
    setBusy(true); setError(''); setNote('')
    try {
      const r = await api.post<{ ok: boolean; task_id?: string }>(
        '/api/selfplay/start',
        { topic, epochs, tasks_per_epoch: tasksPerEpoch, mode,
          flux_checkpoint: fluxCkpt,
          flux_group_size: fluxGroup, flux_workers: fluxWorkers })
      if (r.ok) {
        setNote(`Self-play launched${r.task_id ? ` (task ${r.task_id})` : ''} — telemetry appears as the loop writes it.`)
      } else {
        setError('start failed')
      }
      await load()
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  const stop = async () => {
    if (busy) return
    setBusy(true); setError(''); setNote('')
    try {
      await api.post('/api/selfplay/stop')
      setNote('Stop requested — the loop exits at the next task boundary.')
      await load()
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  const scalars = Object.entries(status).filter(([, v]) =>
    v == null || ['string', 'number', 'boolean'].includes(typeof v))

  const num = (k: string): number | undefined => {
    const v = status[k]
    return typeof v === 'number' ? v : undefined
  }
  const tasksTotal = num('tasks_total')
  const tasksDone = num('tasks_done')
  const shownMetrics = METRICS.filter((m) => num(m.key) != null)
  const topicOpts = topics.includes(topic) ? topics : [topic, ...topics]

  return (
    <div className="h-full min-h-0 flex flex-col p-4 gap-4">
      {/* header: launcher + heartbeat */}
      <Card>
        <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
          <label className="text-[12px] text-text-dim space-y-1">
            <span className="block">Mode</span>
            <Select value={mode}
              onChange={(e) => setMode(e.target.value as typeof mode)}>
              <option value="sft">ForgeLM (SFT)</option>
              <option value="grpo">ForgeLM (GRPO)</option>
              <option value="flux">FluxLM RSI</option>
            </Select>
          </label>
          {mode === 'flux' ? (
            <>
              <label className="text-[12px] text-text-dim space-y-1">
                <span className="block">FLUX checkpoint</span>
                <Select value={fluxCkpt}
                  onChange={(e) => setFluxCkpt(e.target.value)}>
                  <option value="">fresh model</option>
                  {fluxModels.map((f) =>
                    <option key={f} value={f}>{f}</option>)}
                </Select>
              </label>
              <NumInput label="Samples / task" value={fluxGroup}
                onChange={setFluxGroup} min={1} max={16} step={1} />
              <NumInput label="Gen workers" value={fluxWorkers}
                onChange={setFluxWorkers} min={1} max={8} step={1} />
            </>
          ) : (
            <label className="text-[12px] text-text-dim space-y-1">
              <span className="block">Topic</span>
              <Select value={topic}
                onChange={(e) => setTopic(e.target.value)}>
                {topicOpts.map((t) => <option key={t} value={t}>{t}</option>)}
              </Select>
            </label>
          )}
          <NumInput label="Epochs" value={epochs} onChange={setEpochs}
            min={1} max={100} step={1} />
          <NumInput label="Tasks / epoch" value={tasksPerEpoch}
            onChange={setTasksPerEpoch} min={1} max={1000} step={5} />
          {running ? (
            <Btn variant="danger" onClick={stop} disabled={busy}>
              {busy ? <Spinner /> : <Square size={12} />} Stop
            </Btn>
          ) : (
            <Btn variant="primary" onClick={start} disabled={busy}>
              {busy ? <Spinner /> : <Play size={13} />} Start self-play
            </Btn>
          )}
          <div className="ml-auto flex items-center gap-2">
            <Tag kind={stalled ? 'err' : running ? 'ok' : 'idle'}>
              {stalled
                ? 'stalled'
                : running
                  ? `live · ${hbAge != null ? `${Math.round(hbAge)}s ago` : 'no heartbeat'}`
                  : 'idle'}
            </Tag>
            <IconBtn title="Refresh" onClick={load}>
              <RefreshCw size={12} />
            </IconBtn>
          </div>
        </div>
        {note && <div className="text-[11.5px] text-ok mt-2">{note}</div>}
        {error && <div className="text-[11.5px] text-err mt-2">{error}</div>}
      </Card>

      {/* body: status+metrics | events feed */}
      <div className="flex-1 min-h-0 grid grid-cols-1 grid-rows-2 lg:grid-cols-2 lg:grid-rows-1 gap-4">
        <div className="min-h-0 overflow-y-auto space-y-4 pr-0.5">
          <Card title="Status">
            {scalars.length ? scalars.map(([k, v]) => (
              <KvRow key={k} k={k} v={fmtScalar(v)}
                mono={typeof v === 'string' && v.length > 40} />
            )) : (
              <div className="text-[12px] text-text-faint py-2">
                no status.json yet — start a run
              </div>
            )}
          </Card>

          {(shownMetrics.length > 0 || (tasksTotal != null && tasksTotal > 0)) && (
            <Card title="Key metrics">
              {tasksTotal != null && tasksTotal > 0 && (
                <div className="mb-3">
                  <div className="flex justify-between text-[10.5px] text-text-faint mb-1">
                    <span>tasks</span>
                    <span>{tasksDone ?? 0}/{tasksTotal}</span>
                  </div>
                  <Progress value={100 * (tasksDone ?? 0) / tasksTotal} />
                </div>
              )}
              {shownMetrics.length > 0 && (
                <div className="grid grid-cols-2 gap-3">
                  {shownMetrics.map((m) => (
                    <MetricCard key={m.key} title={m.label}
                      value={fmtScalar(num(m.key))} unit={m.unit} />
                  ))}
                </div>
              )}
            </Card>
          )}
        </div>

        <div className="min-h-0 flex flex-col bg-panel border border-border rounded-card shadow-card">
          <div className="px-4 pt-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-text-faint">
            Events
          </div>
          <div ref={feedRef} className="flex-1 overflow-y-auto px-4 pb-3">
            {!events.length && (
              <EmptyState
                icon={<Zap size={34} strokeWidth={1.2} />}
                title="No events"
                desc="Start a self-play run — phase, task and round events stream here live." />
            )}
            {events.map((e, i) => (
              <div key={i}
                className="flex items-baseline gap-2 py-1 border-b border-border/40 last:border-0">
                <span className="text-[10.5px] font-mono text-text-faint shrink-0">
                  {fmtTime(e.ts)}
                </span>
                <Tag kind={eventKind(e)} className="shrink-0">
                  {String(e.kind ?? 'event')}
                </Tag>
                <span className="text-[11.5px] text-text-dim truncate"
                  title={summarize(e)}>
                  {summarize(e)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
