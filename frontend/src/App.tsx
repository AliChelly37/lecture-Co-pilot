import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

// ---------- types ----------
type Asr = {
  model: string
  device: string
  loaded: boolean
  downgraded: boolean
  backlog_s: number
  rtf_recent: number | null
  fatal?: string
}
type Status = {
  recording: boolean
  paused: boolean
  lecture: { id: string } | null
  elapsed_s: number
  badge: number
  asr: Asr | null
  hotkeys: boolean
  power: { state: 'ac' | 'battery'; battery: number | null }
  llm_available: boolean
}
type Course = { id: string; name: string; timezone: string }
type Segment = { id: number; t0: number; t1: number; text: string; conf: number; trigger: string[] }
type Flag = { id: number; t: number; window_t0: number; window_t1: number }
type LectureRow = { id: string; course_name: string; started_at: string; ended_at: string | null; status: string; asr_model: string }
type Deck = { id: string; filename: string; slide_count: number; indexed: boolean }
type Capture = { id: string; t_shutter: number | null; content_kind: string | null; text: string | null; legibility: number | null; status: string }
type Detail = {
  lecture: LectureRow & { deck_id: string | null; power_state: string; battery_start: number | null; battery_end: number | null; asr_rtf_p95: number | null }
  segments: { id: number; t0: number; t1: number; text: string; trigger_terms: string[]; trigger_score: number }[]
  flags: Flag[]
  gaps: { t0: number; t1: number | null; cause: string }[]
  deck: Deck | null
  captures: Capture[]
  alignment: { t0: number; t1: number; slide: number | null; score: number }[]
  usage: { stage: string; model: string; calls: number; cost_usd: number; cache_read: number; input_tokens: number }[]
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`
const when = (iso: string) => new Date(iso).toLocaleString()

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText)
  return res.json()
}
const json = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })

// ---------- app shell ----------
export default function App() {
  const [view, setView] = useState<'live' | 'lectures'>('live')
  const [status, setStatus] = useState<Status | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const refresh = useCallback(async () => setStatus(await api<Status>('/api/status')), [])

  return (
    <div className="app">
      <header>
        <h1>Lecture Co-Pilot</h1>
        <nav>
          <button className={view === 'live' ? 'tab active' : 'tab'} onClick={() => setView('live')}>
            Live
          </button>
          <button className={view === 'lectures' ? 'tab active' : 'tab'} onClick={() => setView('lectures')}>
            Lectures
          </button>
        </nav>
        <StatusPills status={status} connected={connected} />
      </header>
      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      {view === 'live' ? (
        <Live status={status} setStatus={setStatus} refresh={refresh} setConnected={setConnected} setError={setError} />
      ) : (
        <Lectures setError={setError} llmAvailable={!!status?.llm_available} />
      )}
      <footer className="muted">Audio never leaves this laptop. Nothing is sent anywhere during class.</footer>
    </div>
  )
}

function StatusPills({ status, connected }: { status: Status | null; connected: boolean }) {
  const asr = status?.asr
  return (
    <div className="pills">
      <span className={`pill ${connected ? 'ok' : 'bad'}`}>{connected ? 'connected' : 'reconnecting'}</span>
      {status && (
        <span className={`pill ${status.power.state === 'battery' ? 'warn' : ''}`}>
          {status.power.state}
          {status.power.battery !== null ? ` ${status.power.battery}%` : ''}
        </span>
      )}
      {status && <span className={`pill ${status.llm_available ? 'ok' : 'warn'}`}>{status.llm_available ? 'Claude ready' : 'no API key'}</span>}
      {asr && (
        <span className={`pill ${asr.fatal ? 'bad' : asr.loaded ? 'ok' : 'warn'}`}>
          {asr.fatal ? 'ASR failed' : asr.loaded ? `${asr.model} on ${asr.device}` : `loading ${asr.model}…`}
          {asr.loaded && asr.rtf_recent !== null ? ` · rtf ${asr.rtf_recent}` : ''}
          {asr.backlog_s > 5 ? ` · behind ${asr.backlog_s}s` : ''}
          {asr.downgraded ? ' · downgraded' : ''}
        </span>
      )}
      {status?.hotkeys && <span className="pill">F9 flag · F10 pause</span>}
    </div>
  )
}

// ---------- live view ----------
function Live({
  status,
  setStatus,
  refresh,
  setConnected,
  setError,
}: {
  status: Status | null
  setStatus: (s: Status) => void
  refresh: () => Promise<void>
  setConnected: (b: boolean) => void
  setError: (e: string | null) => void
}) {
  const [courses, setCourses] = useState<Course[]>([])
  const [courseId, setCourseId] = useState('')
  const [newCourse, setNewCourse] = useState('')
  const [segments, setSegments] = useState<Segment[]>([])
  const [flags, setFlags] = useState<Flag[]>([])
  const [badge, setBadge] = useState(0)
  const [busy, setBusy] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    refresh().catch((e) => setError(String(e)))
    api<Course[]>('/api/courses').then((c) => {
      setCourses(c)
      setCourseId((cur) => cur || (c[0]?.id ?? ''))
    })
  }, [refresh, setError])

  useEffect(() => {
    let ws: WebSocket | null = null
    let timer: number | undefined
    let closed = false
    const connect = () => {
      ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) timer = window.setTimeout(connect, 1500)
      }
      ws.onmessage = (m) => {
        const ev = JSON.parse(m.data)
        if (ev.type === 'hello') {
          setStatus(ev)
          setBadge(ev.badge)
        } else if (ev.type === 'segment') {
          setSegments((s) => [...s, ev.segment])
          if (ev.badge !== undefined) setBadge(ev.badge)
        } else if (ev.type === 'flag') setFlags((f) => [...f, ev.flag])
        else if (ev.type === 'lecture_started') {
          setSegments([])
          setFlags([])
          setBadge(0)
          refresh()
        } else refresh()
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(timer)
      ws?.close()
    }
  }, [refresh, setConnected, setStatus])

  useEffect(() => {
    if (!status?.recording) return
    const t = window.setInterval(() => refresh().catch(() => undefined), 5000)
    return () => window.clearInterval(t)
  }, [status?.recording, refresh])

  useEffect(() => listRef.current?.scrollTo({ top: listRef.current.scrollHeight }), [segments.length])

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    setError(null)
    try {
      await fn()
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  const recording = !!status?.recording

  return (
    <>
      <section className="controls">
        {!recording ? (
          <>
            <select value={courseId} onChange={(e) => setCourseId(e.target.value)} disabled={!courses.length}>
              {courses.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
            <input placeholder="New course name" value={newCourse} onChange={(e) => setNewCourse(e.target.value)} />
            <button
              onClick={() =>
                run(async () => {
                  const c = await api<Course>('/api/courses', json({ name: newCourse }))
                  setCourses((cs) => [...cs, c])
                  setCourseId(c.id)
                  setNewCourse('')
                })
              }
              disabled={!newCourse.trim() || busy}
            >
              Add course
            </button>
            <button className="primary" onClick={() => run(() => api('/api/lectures/start', json({ course_id: courseId })))} disabled={!courseId || busy}>
              Start lecture
            </button>
          </>
        ) : (
          <>
            <span className={`rec ${status?.paused ? 'paused' : ''}`}>
              {status?.paused ? 'PAUSED' : 'RECORDING'} · {fmt(status?.elapsed_s ?? 0)}
            </span>
            <button className="flag" onClick={() => run(() => api('/api/lectures/flag', { method: 'POST' }))} disabled={busy}>
              I didn't get that (F9)
            </button>
            <button onClick={() => run(() => api('/api/lectures/pause', { method: 'POST' }))} disabled={busy}>
              {status?.paused ? 'Resume (F10)' : 'Pause (F10)'}
            </button>
            <button className="danger" onClick={() => run(() => api('/api/lectures/stop', { method: 'POST' }))} disabled={busy}>
              Stop
            </button>
            <span className="badge" title="Possible deadline mentions (deterministic filter; confirmed after class)">
              {badge} possible deadline{badge === 1 ? '' : 's'}
            </span>
          </>
        )}
      </section>
      <main>
        <section className="transcript" ref={listRef}>
          {segments.length === 0 && <p className="muted">{recording ? 'Listening…' : 'Start a lecture to see the live transcript.'}</p>}
          {segments.map((s) => (
            <p key={s.id} className={s.trigger.length ? 'hit' : ''} title={`conf ${s.conf}`}>
              <span className="t">{fmt(s.t0)}</span>
              {s.text}
              {s.trigger.length > 0 && <span className="terms">{s.trigger.join(', ')}</span>}
            </p>
          ))}
        </section>
        <aside>
          <h2>Flags</h2>
          {flags.length === 0 && <p className="muted">No confusion flags yet.</p>}
          {flags.map((f) => (
            <p key={f.id}>
              <span className="t">{fmt(f.t)}</span> window {fmt(f.window_t0)}–{fmt(f.window_t1)}
            </p>
          ))}
        </aside>
      </main>
    </>
  )
}

// ---------- lectures view ----------
function Lectures({ setError, llmAvailable }: { setError: (e: string | null) => void; llmAvailable: boolean }) {
  const [rows, setRows] = useState<LectureRow[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  useEffect(() => {
    api<LectureRow[]>('/api/lectures').then(setRows).catch((e) => setError(String(e)))
  }, [setError])
  return (
    <main className="lectures">
      <aside>
        <h2>Past lectures</h2>
        {rows.length === 0 && <p className="muted">None yet.</p>}
        {rows.map((r) => (
          <p key={r.id} className={`row ${selected === r.id ? 'sel' : ''}`} onClick={() => setSelected(r.id)}>
            <b>{r.course_name}</b>
            <br />
            <span className="muted">
              {when(r.started_at)} · {r.status}
            </span>
          </p>
        ))}
      </aside>
      <section className="detail">{selected ? <LectureDetail id={selected} setError={setError} llmAvailable={llmAvailable} /> : <p className="muted">Select a lecture.</p>}</section>
    </main>
  )
}

function LectureDetail({ id, setError, llmAvailable }: { id: string; setError: (e: string | null) => void; llmAvailable: boolean }) {
  const [d, setD] = useState<Detail | null>(null)
  const [decks, setDecks] = useState<Deck[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [align, setAlign] = useState<{ coverage: number; off_slide: { t0: number; t1: number }[]; deck_mismatch?: boolean; note?: string } | null>(null)

  const load = useCallback(async () => {
    const detail = await api<Detail>(`/api/lectures/${id}`)
    setD(detail)
    setDecks(await api<Deck[]>('/api/decks'))
  }, [id])
  useEffect(() => {
    load().catch((e) => setError(String(e)))
  }, [load, setError])

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    setError(null)
    try {
      await fn()
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  if (!d) return <p className="muted">Loading…</p>
  const L = d.lecture
  const drain = L.battery_start !== null && L.battery_end !== null ? L.battery_start - L.battery_end : null
  const totalCost = d.usage.reduce((a, u) => a + u.cost_usd, 0)

  return (
    <>
      <h2>
        {L.course_name} · {when(L.started_at)}
      </h2>
      <p className="muted">
        {L.status} · {L.asr_model} · rtf p95 {L.asr_rtf_p95 ?? '–'} · {L.power_state}
        {drain !== null ? ` · battery −${drain}%` : ''} · {d.segments.length} segments · {d.flags.length} flags · {d.gaps.length} gaps · Claude ${totalCost.toFixed(3)}
      </p>

      <h3>Slides</h3>
      <div className="controls">
        <select
          value={L.deck_id ?? ''}
          onChange={(e) => run('deck', () => api(`/api/lectures/${id}/deck`, json({ deck_id: e.target.value || null })))}
          disabled={busy !== null}
        >
          <option value="">No deck</option>
          {decks.map((k) => (
            <option key={k.id} value={k.id}>
              {k.filename} ({k.slide_count} slides{k.indexed ? ', indexed' : ''})
            </option>
          ))}
        </select>
        <label className="upload">
          Upload PDF/PPTX
          <input
            type="file"
            accept=".pdf,.pptx"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (!f) return
              const fd = new FormData()
              fd.append('file', f)
              run('upload', async () => {
                const deck = await api<Deck>('/api/decks', { method: 'POST', body: fd })
                await api(`/api/lectures/${id}/deck`, json({ deck_id: deck.id }))
              })
            }}
          />
        </label>
        {d.deck && !d.deck.indexed && (
          <button onClick={() => run('index', () => api(`/api/decks/${d.deck!.id}/index`, { method: 'POST' }))} disabled={busy !== null || !llmAvailable}>
            Build slide index (Claude)
          </button>
        )}
        <button onClick={() => run('align', async () => setAlign(await api(`/api/lectures/${id}/align`, { method: 'POST' })))} disabled={busy !== null || !d.deck}>
          Align transcript to slides
        </button>
      </div>
      {align && (
        <p className="muted">
          {align.note ?? `coverage ${(align.coverage * 100).toFixed(0)}%`}
          {align.deck_mismatch ? ' · slides may not match this lecture' : ''}
          {align.off_slide.length ? ` · off-slide: ${align.off_slide.map((o) => `${fmt(o.t0)}–${fmt(o.t1)}`).join(', ')}` : ''}
        </p>
      )}

      <h3>Whiteboard photos</h3>
      <div className="controls">
        <label className="upload">
          Import photos
          <input
            type="file"
            accept="image/*"
            multiple
            hidden
            disabled={!llmAvailable}
            onChange={(e) => {
              const fs = Array.from(e.target.files ?? [])
              if (!fs.length) return
              const fd = new FormData()
              fs.forEach((f) => fd.append('files', f))
              run('photos', () => api(`/api/lectures/${id}/photos`, { method: 'POST', body: fd }))
            }}
          />
        </label>
        {!llmAvailable && <span className="muted">Needs the Claude API key (photos are read once and never stored).</span>}
        {busy === 'photos' && <span className="muted">Reading photos…</span>}
      </div>
      {d.captures.map((c) => (
        <p key={c.id} className="capture">
          <span className="t">{c.t_shutter !== null ? fmt(c.t_shutter) : 'unplaced'}</span>
          <b>{c.status}</b>
          {c.content_kind ? ` · ${c.content_kind}` : ''}
          {c.legibility !== null ? ` · legibility ${c.legibility}` : ''}
          {c.text ? <span className="pre">{c.text}</span> : null}
        </p>
      ))}

      <h3>Transcript</h3>
      <section className="transcript tall">
        {d.segments.map((s) => {
          const flagged = d.flags.some((f) => s.t1 > f.window_t0 && s.t0 < f.window_t1)
          const slide = d.alignment.find((w) => s.t0 >= w.t0 && s.t0 < w.t1)?.slide
          return (
            <p key={s.id} className={flagged ? 'flagged' : s.trigger_terms.length ? 'hit' : ''}>
              <span className="t">{fmt(s.t0)}</span>
              {slide ? <span className="slide">s{slide}</span> : null}
              {s.text}
            </p>
          )
        })}
      </section>
    </>
  )
}
