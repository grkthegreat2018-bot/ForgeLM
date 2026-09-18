// Lightweight SVG charts — LiveLineChart + CircularGauge equivalents.
// Custom (not recharts) keeps the bundle small and style fully ours.

import { useMemo } from 'react'

export interface Series {
  name: string
  color: string
  data: number[]
}

export function LineChart({ series, height = 140, window = 240, minY, maxY, yLabel }: {
  series: Series[]
  height?: number
  window?: number
  minY?: number
  maxY?: number
  yLabel?: string
}) {
  const W = 560
  const H = height
  const pad = { l: 34, r: 8, t: 8, b: 16 }
  const iw = W - pad.l - pad.r
  const ih = H - pad.t - pad.b

  const { lo, hi, paths } = useMemo(() => {
    const all = series.flatMap((s) => s.data.slice(-window))
    let lo = minY ?? Math.min(0, ...all)
    let hi = maxY ?? Math.max(1e-9, ...all)
    if (hi - lo < 1e-9) hi = lo + 1
    const paths = series.map((s) => {
      const d = s.data.slice(-window)
      if (d.length < 2) return { s, d: '' }
      const pts = d.map((v, i) => {
        const x = pad.l + (i / Math.max(d.length - 1, 1)) * iw
        const y = pad.t + ih - ((v - lo) / (hi - lo)) * ih
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      return { s, d: `M${pts.join(' L')}` }
    })
    return { lo, hi, paths }
  }, [series, window, minY, maxY, iw, ih])

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
        {/* grid */}
        {[0, 0.5, 1].map((f) => {
          const y = pad.t + ih * f
          const v = hi - f * (hi - lo)
          return (
            <g key={f}>
              <line x1={pad.l} x2={W - pad.r} y1={y} y2={y}
                stroke="#1c2333" strokeWidth="1" />
              <text x={pad.l - 4} y={y + 3} textAnchor="end"
                fontSize="9" fill="#5d687c">
                {v >= 100 ? v.toFixed(0) : v.toPrecision(2)}
              </text>
            </g>
          )
        })}
        {paths.map(({ s, d }) => d && (
          <path key={s.name} d={d} fill="none" stroke={s.color}
            strokeWidth="1.6" strokeLinejoin="round" />
        ))}
        {/* latest dots */}
        {paths.map(({ s, d }) => {
          if (!d) return null
          const last = s.data[s.data.length - 1]
          if (last === undefined) return null
          const x = pad.l + iw
          const y = pad.t + ih - ((last - lo) / (hi - lo)) * ih
          return <circle key={s.name} cx={x} cy={y} r="2.5" fill={s.color} />
        })}
      </svg>
      <div className="flex items-center gap-3 px-1">
        {yLabel && <span className="text-[10px] text-text-faint uppercase">{yLabel}</span>}
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1 text-[10.5px] text-text-dim">
            <span className="w-2 h-2 rounded-full" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  )
}

export function Gauge({ value, caption, size = 120, dangerAt = 85 }: {
  value: number          // 0..100
  caption: string
  size?: number
  dangerAt?: number
}) {
  const r = 42
  const cx = 60
  const cy = 60
  const start = 135
  const span = 270
  const frac = Math.max(0, Math.min(100, value)) / 100
  const end = start + span * frac

  const pt = (deg: number, rad: number) => {
    const a = ((deg - 90) * Math.PI) / 180
    return [cx + rad * Math.cos(a), cy + rad * Math.sin(a)]
  }
  const arc = (from: number, to: number, rad: number) => {
    const [x0, y0] = pt(from, rad)
    const [x1, y1] = pt(to, rad)
    const large = to - from > 180 ? 1 : 0
    return `M${x0.toFixed(1)},${y0.toFixed(1)} A${rad},${rad} 0 ${large} 1 ${x1.toFixed(1)},${y1.toFixed(1)}`
  }

  const color = value >= dangerAt ? '#f85149' : 'url(#gaugeGrad)'
  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <svg viewBox="0 0 120 120" width={size} height={size}>
        <defs>
          <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#8aa3ff" />
            <stop offset="100%" stopColor="#a06bff" />
          </linearGradient>
        </defs>
        <path d={arc(start, start + span, r)} fill="none"
          stroke="#1d2432" strokeWidth="9" strokeLinecap="round" />
        {frac > 0.005 && (
          <path d={arc(start, Math.min(end, start + span), r)} fill="none"
            stroke={color} strokeWidth="9" strokeLinecap="round" />
        )}
        <text x={cx} y={cy + 2} textAnchor="middle" fontSize="19"
          fontWeight="600" fill="#e8edf5">
          {Math.round(value)}
        </text>
        <text x={cx} y={cy + 15} textAnchor="middle" fontSize="8.5"
          fill="#5d687c">%</text>
      </svg>
      <div className="text-[10px] uppercase tracking-widest text-text-faint -mt-2">
        {caption}
      </div>
    </div>
  )
}
