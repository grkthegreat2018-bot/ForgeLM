// Small formatting helpers shared across pages.

/** "just now" | "4m ago" | "2h ago" | "Mar 4" — for unix-seconds ts. */
export function relTime(ts?: number): string {
  if (!ts) return ''
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 45) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  const d = new Date(ts * 1000)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

/** HH:MM for unix-seconds ts. */
export function fmtClock(ts?: number): string {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleTimeString(undefined, {
    hour: '2-digit', minute: '2-digit',
  })
}

/** "42s" | "3m 05s" | "1h 12m" for a duration in seconds. */
export function fmtDur(sec: number): string {
  const s = Math.max(0, Math.round(sec))
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

/** Date bucket for conversation list grouping. */
export function dayBucket(ts?: number): string {
  if (!ts) return 'Earlier'
  const d = new Date(ts * 1000)
  const now = new Date()
  const dayStart = (x: Date) =>
    new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const diffDays = Math.round((dayStart(now) - dayStart(d)) / 86400000)
  if (diffDays <= 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return 'This week'
  return 'Earlier'
}
