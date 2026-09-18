// Compute — GPU telemetry (live ~2s WS push) + torch/CUDA memory probe.
// Keeps a rolling 120-sample history of vram/util/temp/power for charts.

import { Cpu, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { useForge } from '../lib/store'
import type { GpuSnapshot } from '../lib/types'
import { Gauge, LineChart } from '../components/charts'
import {
  Card, IconBtn, KvRow, MetricCard, Progress, Tag,
} from '../components/ui'

const HIST = 120

interface TorchInfo {
  torch?: string
  cuda_available?: boolean
  device?: string
  capability?: string
  allocated_gb?: number
  reserved_gb?: number
  max_allocated_gb?: number
  error?: string
}

interface ComputeInfo extends GpuSnapshot {
  torch?: TorchInfo
}

interface Hist {
  vram: number[]
  util: number[]
  temp: number[]
  power: number[]
}

export default function Compute() {
  const gpu = useForge((s) => s.gpu)
  const [info, setInfo] = useState<ComputeInfo | null>(null)
  const [err, setErr] = useState('')
  const [hist, setHist] = useState<Hist>({
    vram: [], util: [], temp: [], power: [],
  })

  const load = async () => {
    try {
      setInfo(await api.get<ComputeInfo>('/api/compute'))
      setErr('')
    } catch (e) {
      setErr(String(e))
    }
  }
  useEffect(() => { void load() }, [])

  // rolling history — appended whenever a new gpu snapshot arrives
  useEffect(() => {
    if (!gpu.available) return
    setHist((h) => ({
      vram: [...h.vram, gpu.vram_pct].slice(-HIST),
      util: [...h.util, gpu.util_pct].slice(-HIST),
      temp: [...h.temp, gpu.temp_c ?? 0].slice(-HIST),
      power: [...h.power, gpu.power_w ?? 0].slice(-HIST),
    }))
  }, [gpu])

  const torch = info?.torch
  const gbUsed = gpu.vram_used_mb / 1024
  const gbTotal = gpu.vram_total_mb / 1024
  const hasGpu = gpu.available || !!info?.available

  return (
    <div className="h-full overflow-y-auto p-5 space-y-4">
      {/* gauges + metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card title="GPU" className="col-span-2">
          {gpu.available ? (
            <div className="flex items-center justify-around">
              <Gauge value={gpu.vram_pct} caption="vram" />
              <Gauge value={gpu.util_pct} caption="util" />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-text-faint">
              <Cpu size={28} strokeWidth={1.2} className="mb-2" />
              <span className="text-[12px]">no GPU telemetry</span>
            </div>
          )}
        </Card>
        <MetricCard
          title="Temp"
          value={gpu.temp_c != null ? gpu.temp_c : '—'}
          unit="°C"
          spark={hist.temp}
          color="#d29922" />
        <MetricCard
          title="Power"
          value={gpu.power_w != null ? Math.round(gpu.power_w) : '—'}
          unit={gpu.power_limit_w != null
            ? `/ ${Math.round(gpu.power_limit_w)} W` : 'W'}
          spark={hist.power}
          color="#a06bff" />
      </div>

      {/* history charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="VRAM % — history">
          <LineChart
            series={[{ name: 'vram', color: '#8aa3ff', data: hist.vram }]}
            minY={0} maxY={100} window={HIST} height={120} />
        </Card>
        <Card title="GPU util % — history">
          <LineChart
            series={[{ name: 'util', color: '#3ad9c9', data: hist.util }]}
            minY={0} maxY={100} window={HIST} height={120} />
        </Card>
      </div>

      {/* torch + gpu details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card
          title="PyTorch / CUDA"
          actions={
            <IconBtn title="Refresh" onClick={() => void load()}>
              <RefreshCw size={13} />
            </IconBtn>
          }>
          {err && (
            <div className="text-[11.5px] text-err mb-1">{err}</div>
          )}
          {torch?.error ? (
            <div className="text-[12px] text-err">{torch.error}</div>
          ) : (
            <>
              <KvRow k="torch" mono v={torch?.torch ?? '—'} />
              <KvRow k="cuda available" v={torch ? (
                <Tag kind={torch.cuda_available ? 'ok' : 'err'}>
                  {torch.cuda_available ? 'yes' : 'no'}
                </Tag>
              ) : '—'} />
              <KvRow k="device" v={torch?.device ?? '—'} />
              <KvRow k="capability" mono v={torch?.capability ?? '—'} />
              <KvRow k="allocated"
                v={torch?.allocated_gb != null
                  ? `${torch.allocated_gb.toFixed(2)} GB` : '—'} />
              <KvRow k="reserved"
                v={torch?.reserved_gb != null
                  ? `${torch.reserved_gb.toFixed(2)} GB` : '—'} />
              <KvRow k="max allocated"
                v={torch?.max_allocated_gb != null
                  ? `${torch.max_allocated_gb.toFixed(2)} GB` : '—'} />
            </>
          )}
        </Card>

        <Card title="GPU details">
          {hasGpu ? (
            <>
              <KvRow k="name" v={gpu.name || info?.name || '—'} />
              <KvRow k="vram"
                v={`${gbUsed.toFixed(1)} / ${gbTotal.toFixed(0)} GB`} />
              <div className="py-1.5">
                <Progress value={gpu.vram_pct} />
              </div>
              <KvRow k="util" v={`${Math.round(gpu.util_pct)}%`} />
              <KvRow k="power"
                v={gpu.power_w != null
                  ? `${Math.round(gpu.power_w)} / ${gpu.power_limit_w != null ? Math.round(gpu.power_limit_w) : '—'} W`
                  : '—'} />
              <KvRow k="temp"
                v={gpu.temp_c != null ? `${gpu.temp_c}°C` : '—'} />
            </>
          ) : (
            <div className="text-[12px] text-text-dim py-4 text-center">
              GPU not detected
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
