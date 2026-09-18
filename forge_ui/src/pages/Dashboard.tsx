// Dashboard — engine state, GPU health, live runs, quick actions.

import { Bot, Cpu, Layers, MessageSquare, Sparkles, Square } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import { Gauge } from '../components/charts'
import { Btn, Card, EmptyState, KvRow, StatusDot, Tag } from '../components/ui'

export default function Dashboard() {
  const engine = useForge((s) => s.engine)
  const gpu = useForge((s) => s.gpu)
  const lora = useForge((s) => s.lora)
  const runs = useForge((s) => s.runs)
  const tasks = useForge((s) => s.tasks)
  const agentRuns = useForge((s) => s.agentRuns)
  const progress = useForge((s) => s.engineProgress)
  const navigate = useNavigate()

  const liveRuns = runs.filter((r) => r.is_live)
  const liveTasks = tasks.filter((t) => t.is_live)
  const kind = engine.state === 'ready' ? 'ok'
    : engine.state === 'loading' ? 'warn'
    : engine.state === 'error' ? 'err' : 'idle'

  return (
    <div className="h-full overflow-y-auto p-5 space-y-4">
      {/* top strip: engine + gpu */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-2" title="Engine">
          <div className="flex items-start gap-6">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <StatusDot kind={kind} pulse={engine.state === 'loading'} />
                <span className="text-[16px] font-semibold capitalize">
                  {engine.state}
                </span>
                {engine.busy && <Tag kind="accent">busy</Tag>}
              </div>
              {engine.state === 'ready' && (
                <div className="grid grid-cols-2 gap-x-6">
                  <KvRow k="checkpoint" mono
                    v={engine.info.checkpoint?.split(/[\\/]/).pop()} />
                  <KvRow k="config" v={engine.info.config_name} />
                  <KvRow k="loaded in" v={`${engine.info.load_s?.toFixed(1)}s`} />
                  <KvRow k="device" v={engine.info.device} />
                  <KvRow k="dtype" v={engine.info.dtype} />
                  <KvRow k="compile" v={engine.info.use_compile ? 'on' : 'off'} />
                </div>
              )}
              {engine.state === 'loading' && (
                <div className="text-[12.5px] text-warn">
                  {progress ?? 'loading weights…'}
                </div>
              )}
              {engine.state === 'error' && (
                <div className="text-[12.5px] text-err">{engine.error}</div>
              )}
              {engine.state === 'idle' && (
                <div className="text-[12.5px] text-text-dim">
                  No model loaded — pick a checkpoint on the Models page.
                </div>
              )}
            </div>
            <div className="flex flex-col gap-1.5 shrink-0">
              {engine.state === 'ready' ? (
                <>
                  <Btn variant="subtle" onClick={() =>
                    api.post('/api/engine/unload').catch(() => undefined)}>
                    <Square size={12} /> Unload
                  </Btn>
                  <Btn variant="subtle" onClick={() =>
                    api.post('/api/engine/reactivate').catch(() => undefined)}>
                    Reactivate
                  </Btn>
                </>
              ) : engine.state === 'idle' || engine.state === 'error' ? (
                <Btn variant="primary" onClick={() => navigate('/models')}>
                  Load model
                </Btn>
              ) : null}
            </div>
          </div>
        </Card>

        <Card title="GPU">
          {gpu.available ? (
            <div className="flex items-center justify-around">
              <Gauge value={gpu.vram_pct} caption="vram" />
              <Gauge value={gpu.util_pct} caption="util" />
            </div>
          ) : (
            <div className="text-[12.5px] text-text-dim py-6 text-center">
              no GPU telemetry
            </div>
          )}
          {gpu.available && (
            <div className="grid grid-cols-2 gap-x-4 mt-1">
              <KvRow k="VRAM" v={`${(gpu.vram_used_mb / 1024).toFixed(1)}/${(gpu.vram_total_mb / 1024).toFixed(0)} GB`} />
              <KvRow k="temp" v={gpu.temp_c != null ? `${gpu.temp_c}°C` : '—'} />
            </div>
          )}
        </Card>
      </div>

      {/* quick actions */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <QuickAction icon={<Bot size={18} />} label="Run agent"
          desc="autonomous tool use" onClick={() => navigate('/agent')} />
        <QuickAction icon={<MessageSquare size={18} />} label="Chat"
          desc="test the model" onClick={() => navigate('/chat')} />
        <QuickAction icon={<Layers size={18} />} label="LoRA"
          desc={lora.current ?? 'adapters'} onClick={() => navigate('/lora')} />
        <QuickAction icon={<Cpu size={18} />} label="Tune engine"
          desc="activation settings" onClick={() => navigate('/engine')} />
      </div>

      {/* activity row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card title="Training runs">
          {liveRuns.length ? liveRuns.map((r) => (
            <div key={r.id} className="flex items-center gap-2 py-1">
              <StatusDot kind="ok" pulse />
              <span className="text-[12.5px] truncate flex-1">{r.name}</span>
              <span className="text-[11px] text-text-faint">
                {r.step}/{r.max_steps} · loss {r.loss.toFixed(3)}
              </span>
            </div>
          )) : (
            <div className="text-[12px] text-text-faint py-2">
              no live training runs
            </div>
          )}
        </Card>

        <Card title="Processes">
          {liveTasks.length ? liveTasks.map((t) => (
            <div key={t.id} className="flex items-center gap-2 py-1">
              <StatusDot kind="ok" pulse />
              <span className="text-[12.5px] truncate flex-1">{t.name}</span>
              <span className="text-[11px] text-text-faint">
                {Math.round(t.elapsed_s)}s
              </span>
            </div>
          )) : (
            <div className="text-[12px] text-text-faint py-2">
              no running processes
            </div>
          )}
        </Card>

        <Card title="Agent runs">
          {agentRuns.length ? agentRuns.slice(0, 5).map((r) => (
            <div key={r.run_id} className="flex items-center gap-2 py-1">
              <Tag kind={r.status === 'running' ? 'accent'
                : r.status === 'done' ? 'ok' : 'err'}>
                {r.status}
              </Tag>
              <span className="text-[12px] text-text-dim truncate flex-1">
                {r.task}
              </span>
            </div>
          )) : (
            <EmptyState icon={<Sparkles size={22} strokeWidth={1.4} />}
              title="No agent runs"
              desc="The agent is ready when the engine is." />
          )}
        </Card>
      </div>
    </div>
  )
}

function QuickAction({ icon, label, desc, onClick }: {
  icon: React.ReactNode
  label: string
  desc: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 bg-panel border border-border rounded-card px-4 py-3 text-left hover:border-accent/50 hover:bg-panel-alt transition-all cursor-pointer group">
      <span className="text-text-faint group-hover:text-accent-hi transition-colors">
        {icon}
      </span>
      <div className="min-w-0">
        <div className="text-[13px] font-medium">{label}</div>
        <div className="text-[11px] text-text-faint truncate">{desc}</div>
      </div>
    </button>
  )
}
