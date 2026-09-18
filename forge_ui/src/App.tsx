import { useEffect } from 'react'
import { Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import { PAGES, Sidebar } from './components/Sidebar'
import { ApprovalPopups, CommandPalette, PageHeader } from './components/shell'
import { connectWs, useForge } from './lib/store'
import { useUi } from './lib/ui'
import Agent from './pages/Agent'
import Chat from './pages/Chat'
import Compute from './pages/Compute'
import Dashboard from './pages/Dashboard'
import Engine from './pages/Engine'
import Finetune from './pages/Finetune'
import Generations from './pages/Generations'
import Launch from './pages/Launch'
import Logs from './pages/Logs'
import Lora from './pages/Lora'
import Models from './pages/Models'
import SelfPlay from './pages/SelfPlay'
import Tasks from './pages/Tasks'
import Training from './pages/Training'

const ROUTES: Record<string, React.ComponentType> = {
  '/': Dashboard,
  '/chat': Chat,
  '/agent': Agent,
  '/generations': Generations,
  '/engine': Engine,
  '/models': Models,
  '/lora': Lora,
  '/finetune': Finetune,
  '/selfplay': SelfPlay,
  '/training': Training,
  '/launch': Launch,
  '/tasks': Tasks,
  '/compute': Compute,
  '/logs': Logs,
}

export default function App() {
  const navigate = useNavigate()
  const toggle = useUi((s) => s.toggleCollapsed)
  const setPalette = useUi((s) => s.setPalette)
  const refresh = useForge((s) => s.refreshStatus)

  useEffect(() => {
    connectWs()
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return
      const k = e.key.toLowerCase()
      if (k === 'b') { e.preventDefault(); toggle() }
      else if (k === 'k') { e.preventDefault(); setPalette(true) }
      else if (k === 'r') { e.preventDefault(); refresh().catch(() => undefined) }
      else if (/^[1-9]$/.test(k)) {
        const idx = parseInt(k, 10) - 1
        if (idx < PAGES.length) { e.preventDefault(); navigate(PAGES[idx].path) }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navigate, toggle, setPalette, refresh])

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar />
      <main className="flex-1 min-w-0 h-full overflow-hidden flex flex-col bg-bg">
        <Routes>
          {Object.entries(ROUTES).map(([path, Comp]) => (
            <Route key={path} path={path} element={
              <PageFrame path={path}><Comp /></PageFrame>
            } />
          ))}
          <Route path="*" element={<Navigate to="/agent" replace />} />
        </Routes>
      </main>
      <ApprovalPopups />
      <CommandPalette />
    </div>
  )
}

function PageFrame({ path, children }: { path: string; children: React.ReactNode }) {
  const def = PAGES.find((p) => p.path === path)
  return (
    <div className="flex flex-col h-full min-h-0 fade-in">
      <PageHeader title={def?.label ?? ''} subtitle={def?.subtitle ?? ''} />
      <div className="flex-1 min-h-0 overflow-hidden">
        {children}
      </div>
    </div>
  )
}
