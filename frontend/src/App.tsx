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
  llm?: { provider: string; text_model: string; local: boolean }
}
type Course = { id: string; name: string; timezone: string }
type Segment = { id: number; t0: number; t1: number; text: string; conf: number; trigger: string[] }
type Flag = { id: number; t: number; window_t0: number; window_t1: number }
type LectureRow = { id: string; course_name: string; started_at: string; ended_at: string | null; status: string; asr_model: string }
type Deck = { id: string; filename: string; slide_count: number; indexed: boolean }
type Capture = { id: string; t_shutter: number | null; content_kind: string | null; text: string | null; legibility: number | null; status: string }
type Suggestion = {
  id: string
  tier: 'one_tap' | 'maybe'
  state: string
  payload: {
    title: string
    type: string
    intent: string
    date: string | null
    time: string | null
    date_expression: string
    resolution_note: string
    evidence_quote: string
    t0: number | null
    confidence: number
    asr_conf: number
    course_hint: string
    course_name: string
    lecture_id: string
    tier_reason?: string
  }
}
type FlagExplanationT = { flag_id: number; t: number; what_was_confusing?: string; explanation?: string; prerequisite?: string; sources?: string[]; error?: string }
type RecapT = {
  id: string
  version: number
  model: string
  rating: number | null
  flag_helpful: Record<string, boolean>
  sections: {
    title: string
    highlights: string[]
    concepts: { name: string; importance: 'high' | 'medium' | 'low'; explanation: string; sources: string[] }[]
    review_questions: { question: string; answer: string; sources: string[] }[]
    off_slide_notes: string[]
    gaps_note: string
    flag_explanations: FlagExplanationT[]
    detected_events: string[]
  }
}
type AnswerT = { answer: string; sources: string[]; coverage: 'answered_from_lecture' | 'partly_from_lecture' | 'not_in_lecture' }
type ExtractSummary = { chunks: number; candidates: number; surfaced: number; suggestions: { one_tap: number; maybe: number; log: number; existing: number }; note?: string }
type Detail = {
  lecture: LectureRow & { deck_id: string | null; power_state: string; battery_start: number | null; battery_end: number | null; asr_rtf_p95: number | null }
  segments: { id: number; t0: number; t1: number; text: string; trigger_terms: string[]; trigger_score: number }[]
  flags: Flag[]
  gaps: { t0: number; t1: number | null; cause: string }[]
  deck: Deck | null
  captures: Capture[]
  alignment: { t0: number; t1: number; slide: number | null; score: number }[]
  suggestions: Suggestion[]
  recap: RecapT | null
  usage: { stage: string; model: string; calls: number; cost_usd: number; cache_read: number; input_tokens: number }[]
}

// "t=12:34" -> 754 seconds; anything else is shown as a chip.
const sourceSeconds = (s: string): number | null => {
  const m = /^t=(\d+):(\d{2})$/.exec(s.trim())
  return m ? Number(m[1]) * 60 + Number(m[2]) : null
}

function Sources({ list, jump }: { list?: string[]; jump: (t: number) => void }) {
  if (!list?.length) return null
  return (
    <span className="sources">
      {list.map((s, i) => {
        const t = sourceSeconds(s)
        return t !== null ? (
          <button key={i} className="chip link" onClick={() => jump(t)} title="Show this moment in the transcript">
            {s.slice(2)}
          </button>
        ) : (
          <span key={i} className="chip">
            {s}
          </span>
        )
      })}
    </span>
  )
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
  type View = 'live' | 'lectures' | 'inbox' | 'dashboard'
  // Routes: #lectures, #lectures/<lecture id>, #inbox, #dashboard.
  const fromHash = (): View => {
    const h = location.hash.replace('#', '').split('/')[0]
    return h === 'lectures' || h === 'inbox' || h === 'dashboard' ? h : 'live'
  }
  const lectureFromHash = () => location.hash.replace('#', '').split('/')[1] ?? null
  const [view, setViewState] = useState<View>(fromHash)
  const setView = (v: View) => {
    setViewState(v)
    history.replaceState(null, '', v === 'live' ? location.pathname : `#${v}`)
  }
  useEffect(() => {
    const onHash = () => setViewState(fromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const [status, setStatus] = useState<Status | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [live, setLive] = useState<LiveState>({ segments: [], flags: [], badge: 0 })
  const refresh = useCallback(async () => setStatus(await api<Status>('/api/status')), [])

  // One socket for the whole app: status stays live on every tab, and the
  // transcript survives switching tabs during a lecture.
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
          setLive((l) => ({ ...l, badge: ev.badge }))
        } else if (ev.type === 'segment') {
          setLive((l) => ({ ...l, segments: [...l.segments, ev.segment], badge: ev.badge ?? l.badge }))
        } else if (ev.type === 'flag') {
          setLive((l) => ({ ...l, flags: [...l.flags, ev.flag] }))
        } else if (ev.type === 'lecture_started') {
          setLive({ segments: [], flags: [], badge: 0 })
          refresh()
        } else if (ev.type !== 'progress') {
          refresh()
        }
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(timer)
      ws?.close()
    }
  }, [refresh])

  useEffect(() => {
    refresh().catch((e) => setError(String(e)))
  }, [refresh])

  useEffect(() => {
    if (!status?.recording) return
    const t = window.setInterval(() => refresh().catch(() => undefined), 5000)
    return () => window.clearInterval(t)
  }, [status?.recording, refresh])

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
          <button className={view === 'inbox' ? 'tab active' : 'tab'} onClick={() => setView('inbox')}>
            Inbox
          </button>
          <button className={view === 'dashboard' ? 'tab active' : 'tab'} onClick={() => setView('dashboard')}>
            Dashboard
          </button>
        </nav>
        <StatusPills status={status} connected={connected} />
      </header>
      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      {view === 'live' && <Live status={status} live={live} refresh={refresh} setError={setError} />}
      {view === 'lectures' && <Lectures setError={setError} llmAvailable={!!status?.llm_available} initialId={lectureFromHash()} />}
      {view === 'inbox' && <Inbox setError={setError} />}
      {view === 'dashboard' && <Dashboard setError={setError} />}
      <footer className="muted">Audio never leaves this laptop. During class, nothing is sent anywhere.</footer>
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
      {status && (
        <span className={`pill ${status.llm_available ? 'ok' : 'warn'}`}>
          {status.llm_available ? `${status.llm?.text_model ?? 'model'} · ${status.llm?.local ? 'on this laptop' : status.llm?.provider}` : 'no model set up'}
        </span>
      )}
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
type LiveState = { segments: Segment[]; flags: Flag[]; badge: number }

function Live({
  status,
  live,
  refresh,
  setError,
}: {
  status: Status | null
  live: LiveState
  refresh: () => Promise<void>
  setError: (e: string | null) => void
}) {
  const { segments, flags, badge } = live
  const [courses, setCourses] = useState<Course[]>([])
  const [courseId, setCourseId] = useState('')
  const [newCourse, setNewCourse] = useState('')
  const [busy, setBusy] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api<Course[]>('/api/courses')
      .then((c) => {
        setCourses(c)
        setCourseId((cur) => cur || (c[0]?.id ?? ''))
      })
      .catch((e) => setError(String(e)))
  }, [setError])

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
            <button
              className="flag"
              onClick={() => run(() => api('/api/lectures/flag', { method: 'POST' }))}
              onPointerMove={(e) => {
                const r = e.currentTarget.getBoundingClientRect()
                e.currentTarget.style.setProperty('--mx', `${((e.clientX - r.left) / r.width) * 100}%`)
                e.currentTarget.style.setProperty('--my', `${((e.clientY - r.top) / r.height) * 100}%`)
              }}
              disabled={busy}
            >
              I didn't get that <kbd>F9</kbd>
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
          {segments.length === 0 && (
            <div className="empty">
              {recording ? (
                <>
                  <b>Listening</b>
                  <span>The transcript appears here as it is spoken.</span>
                </>
              ) : (
                <>
                  <b>Pick a course and start the lecture</b>
                  <span>
                    Everything stays on this laptop. <kbd>F9</kbd> marks a moment you didn't get, <kbd>F10</kbd> pauses.
                  </span>
                </>
              )}
            </div>
          )}
          {segments.map((s) => (
            <p key={s.id} className={s.trigger.length ? 'hit' : ''} title={`conf ${s.conf}`}>
              <span className="t">{fmt(s.t0)}</span>
              <span className="tx">{s.text}</span>
              {s.trigger.length > 0 && <span className="terms">might be a deadline: {s.trigger.join(', ')}</span>}
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

// ---------- dashboard ----------
type DashboardT = {
  courses: { id: string; name: string; lectures: number; last_lecture: string | null; open_flags: number; pending_suggestions: number; recaps: number }[]
  inbox: { one_tap: number; maybe: number }
  upcoming: Suggestion[]
  usage: { by_stage: { stage: string; model: string; calls: number; cost_usd: number; input_tokens: number; output_tokens: number }[]; total_cost_usd: number; calls: number }
  asr: { rtf_p95_avg: number | null; battery_drain_pct_per_hour: number | null; recorded_hours: number }
  storage: { db_bytes: number }
}

function Dashboard({ setError }: { setError: (e: string | null) => void }) {
  const [d, setD] = useState<DashboardT | null>(null)
  useEffect(() => {
    api<DashboardT>('/api/dashboard').then(setD).catch((e) => setError(String(e)))
  }, [setError])
  if (!d) return <p className="muted">Loading…</p>
  return (
    <main className="dash">
      <section className="cards">
        <div className="stat">
          <b>{d.inbox.one_tap}</b>
          <span>ready to confirm</span>
        </div>
        <div className="stat">
          <b>{d.inbox.maybe}</b>
          <span>to check</span>
        </div>
        <div className="stat">
          <b>{d.courses.reduce((a, c) => a + c.open_flags, 0)}</b>
          <span>open flags</span>
        </div>
        <div className="stat">
          <b>{d.asr.recorded_hours.toFixed(1)} h</b>
          <span>recorded</span>
        </div>
        <div className="stat">
          <b>${d.usage.total_cost_usd.toFixed(2)}</b>
          <span>model spend ({d.usage.calls} calls)</span>
        </div>
        <div className="stat">
          <b>{d.asr.battery_drain_pct_per_hour !== null ? `${d.asr.battery_drain_pct_per_hour}%/h` : '–'}</b>
          <span>battery drain on battery</span>
        </div>
        <div className="stat">
          <b>{d.asr.rtf_p95_avg ?? '–'}</b>
          <span>ASR real-time factor (p95 avg)</span>
        </div>
        <div className="stat">
          <b>{(d.storage.db_bytes / 1048576).toFixed(1)} MB</b>
          <span>local database</span>
        </div>
      </section>
      <section>
        <h2>Courses</h2>
        <table>
          <thead>
            <tr>
              <th>Course</th>
              <th>Lectures</th>
              <th>Recaps</th>
              <th>Open flags</th>
              <th>Pending</th>
              <th>Last</th>
            </tr>
          </thead>
          <tbody>
            {d.courses.map((c) => (
              <tr key={c.id}>
                <td>{c.name}</td>
                <td>{c.lectures}</td>
                <td>{c.recaps}</td>
                <td>{c.open_flags}</td>
                <td>{c.pending_suggestions}</td>
                <td className="muted">{c.last_lecture ? when(c.last_lecture) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h2>Upcoming deadlines</h2>
        {d.upcoming.length === 0 && <p className="muted">None confirmed yet.</p>}
        {d.upcoming.map((s) => (
          <p key={s.id} className="row">
            <b>{s.payload.date}</b>
            {s.payload.time ? ` ${s.payload.time}` : ''} · {s.payload.title} <span className="muted">· {s.payload.course_name} · {s.state}</span>
          </p>
        ))}
        <h2>Model calls by stage</h2>
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>Model</th>
              <th>Calls</th>
              <th>In</th>
              <th>Out</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {d.usage.by_stage.map((u, i) => (
              <tr key={i}>
                <td>{u.stage}</td>
                <td className="muted">{u.model}</td>
                <td>{u.calls}</td>
                <td>{u.input_tokens}</td>
                <td>{u.output_tokens}</td>
                <td>${u.cost_usd.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  )
}

// ---------- inbox view ----------
type TargetHealthT = { name: string; configured: boolean; connected: boolean; detail: string }

function Inbox({ setError }: { setError: (e: string | null) => void }) {
  const [proposed, setProposed] = useState<Suggestion[]>([])
  const [confirmed, setConfirmed] = useState<Suggestion[]>([])
  const [targets, setTargets] = useState<TargetHealthT[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [dates, setDates] = useState<Record<string, string>>({})

  const load = useCallback(async () => {
    setProposed(await api<Suggestion[]>('/api/suggestions?state_filter=proposed'))
    setConfirmed(await api<Suggestion[]>('/api/suggestions?state_filter=done'))
    setTargets(await api<TargetHealthT[]>('/api/targets'))
  }, [])

  const connectGoogle = async () => {
    setBusy('gcal')
    setError(null)
    try {
      await api('/api/targets/gcal/connect', { method: 'POST' })
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  useEffect(() => {
    load().catch((e) => setError(String(e)))
  }, [load, setError])

  const act = async (id: string, action: string, body?: unknown) => {
    setBusy(id)
    setError(null)
    try {
      await api(`/api/suggestions/${id}/${action}`, body ? json(body) : { method: 'POST' })
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const oneTap = proposed.filter((s) => s.tier === 'one_tap')
  const maybe = proposed.filter((s) => s.tier === 'maybe')

  const Card = ({ s, children }: { s: Suggestion; children: React.ReactNode }) => {
    const p = s.payload
    return (
      <div className={`card ${s.tier}`}>
        <div className="card-head">
          <span className="type">{p.type}</span>
          <b>{p.title}</b>
          <span className="muted"> · {p.course_name}</span>
        </div>
        <div className="card-date">
          {p.date ? (
            <>
              <b>{p.date}</b>
              {p.time ? ` ${p.time}` : ''}
            </>
          ) : (
            <span className="warn-text">no date resolved</span>
          )}
          <span className="muted"> — {p.resolution_note}</span>
        </div>
        <blockquote>
          {p.t0 !== null && <span className="t">{fmt(p.t0)}</span>}
          “{p.evidence_quote}”
        </blockquote>
        <div className="muted small">
          {s.tier === 'maybe' && p.tier_reason ? <span className="warn-text">Needs you because: {p.tier_reason}. </span> : null}
          intent {p.intent} · model confidence {p.confidence} · audio confidence {p.asr_conf}
          {p.course_hint ? ` · mentioned for: ${p.course_hint}` : ''}
        </div>
        <div className="card-actions">{children}</div>
      </div>
    )
  }

  return (
    <main className="inbox">
      <section>
        <h2>Ready ({oneTap.length})</h2>
        {oneTap.length === 0 && <p className="muted">Nothing waiting. Open a lecture and choose Find deadlines.</p>}
        {oneTap.map((s) => (
          <Card key={s.id} s={s}>
            <button className="primary" onClick={() => act(s.id, 'confirm')} disabled={busy === s.id}>
              Confirm
            </button>
            <button onClick={() => act(s.id, 'dismiss')} disabled={busy === s.id}>
              Dismiss
            </button>
          </Card>
        ))}

        <h2>Needs a date ({maybe.length})</h2>
        {maybe.length === 0 && <p className="muted">Nothing needs a date.</p>}
        {maybe.map((s) => (
          <Card key={s.id} s={s}>
            <input type="date" value={dates[s.id] ?? s.payload.date ?? ''} onChange={(e) => setDates((d) => ({ ...d, [s.id]: e.target.value }))} />
            <button onClick={() => act(s.id, 'date', { date: dates[s.id] ?? s.payload.date })} disabled={busy === s.id || !(dates[s.id] ?? s.payload.date)}>
              Set date
            </button>
            <button onClick={() => act(s.id, 'dismiss')} disabled={busy === s.id}>
              Dismiss
            </button>
          </Card>
        ))}
      </section>
      <aside>
        <h2>Targets</h2>
        {targets.length === 0 && <p className="muted small">Calendar and Notion aren't connected, so confirmed items are downloaded as .ics. The README explains how to connect them.</p>}
        {targets.map((t) => (
          <p key={t.name} className="small">
            <span className={`pill ${t.connected ? 'ok' : t.configured ? 'warn' : 'bad'}`}>{t.name === 'gcal' ? 'Google Calendar' : 'Notion'}</span> {t.detail}
            {t.name === 'gcal' && t.configured && !t.connected && (
              <button className="small" onClick={connectGoogle} disabled={busy !== null}>
                {busy === 'gcal' ? 'Waiting for sign-in…' : 'Connect Google'}
              </button>
            )}
          </p>
        ))}

        <h2>Confirmed ({confirmed.length})</h2>
        {confirmed.length > 0 && (
          <p>
            <a className="button" href="/api/suggestions/export.ics">
              Download .ics
            </a>
          </p>
        )}
        {confirmed.map((s) => {
          const ext = (s.payload as unknown as { external?: Record<string, { id: string; url: string | null }> }).external ?? {}
          const failed = s.state.startsWith('failed')
          return (
            <p key={s.id} className="row">
              <b>{s.payload.title}</b> · {s.payload.date}
              {s.payload.time ? ` ${s.payload.time}` : ''}
              <br />
              <span className={`small ${failed ? 'warn-text' : 'muted'}`}>
                {s.state}
                {(s as unknown as { error?: string }).error ? `: ${(s as unknown as { error?: string }).error}` : ''}
              </span>
              <br />
              {Object.entries(ext).map(([name, e]) =>
                e.url ? (
                  <a key={name} className="chip link" href={e.url} target="_blank" rel="noreferrer">
                    open in {name === 'gcal' ? 'Calendar' : name}
                  </a>
                ) : (
                  <span key={name} className="chip">
                    {name} ✓
                  </span>
                ),
              )}{' '}
              {s.state === 'failed_retryable' && (
                <button className="small" onClick={() => act(s.id, 'retry')} disabled={busy === s.id}>
                  Retry
                </button>
              )}
              <button className="small" onClick={() => act(s.id, 'undo')} disabled={busy === s.id}>
                Undo
              </button>
            </p>
          )
        })}
        <p className="muted small">Nothing is written anywhere until you confirm. Undo removes what was written.</p>
      </aside>
    </main>
  )
}

// ---------- lectures view ----------
function Lectures({ setError, llmAvailable, initialId }: { setError: (e: string | null) => void; llmAvailable: boolean; initialId: string | null }) {
  const [rows, setRows] = useState<LectureRow[]>([])
  const [selected, setSelectedState] = useState<string | null>(initialId)
  const setSelected = (id: string | null) => {
    setSelectedState(id)
    history.replaceState(null, '', id ? `#lectures/${id}` : '#lectures')
  }
  useEffect(() => {
    api<LectureRow[]>('/api/lectures')
      .then((r) => {
        setRows(r)
        setSelectedState((s) => s ?? r[0]?.id ?? null)
      })
      .catch((e) => setError(String(e)))
  }, [setError])
  return (
    <main className="lectures">
      <aside>
        <h2>Lectures</h2>
        {rows.length === 0 && <p className="muted">No lectures yet. Record one from Live.</p>}
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
      <section className="detail">{selected ? <LectureDetail id={selected} setError={setError} llmAvailable={llmAvailable} /> : <p className="muted">Pick a lecture on the left.</p>}</section>
    </main>
  )
}

function LectureDetail({ id, setError, llmAvailable }: { id: string; setError: (e: string | null) => void; llmAvailable: boolean }) {
  const [d, setD] = useState<Detail | null>(null)
  const [decks, setDecks] = useState<Deck[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [align, setAlign] = useState<{ coverage: number; off_slide: { t0: number; t1: number }[]; deck_mismatch?: boolean; note?: string } | null>(null)
  const [extract, setExtract] = useState<ExtractSummary | null>(null)
  const [progress, setProgress] = useState<{ done: number; total: number; stage: string } | null>(null)
  const [highlightT, setHighlightT] = useState<number | null>(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<AnswerT | null>(null)
  const [openQ, setOpenQ] = useState<Record<number, boolean>>({})
  const transcriptRef = useRef<HTMLElement>(null)

  // Progress events for a running recap job arrive over the event bus.
  useEffect(() => {
    if (busy !== 'recap') return
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
    ws.onmessage = (m) => {
      const ev = JSON.parse(m.data)
      if (ev.type === 'progress' && ev.lecture_id === id) setProgress({ done: ev.done, total: ev.total, stage: ev.stage })
    }
    return () => {
      ws.close()
      setProgress(null)
    }
  }, [busy, id])

  const jump = (t: number) => {
    setHighlightT(t)
    const target = d?.segments.reduce<(typeof d.segments)[number] | null>((best, s) => (s.t0 <= t + 0.5 && (!best || s.t0 > best.t0) ? s : best), null)
    if (target) document.getElementById(`seg-${target.id}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    window.setTimeout(() => setHighlightT(null), 2500)
  }

  const rate = (rating: number | null, flagHelpful?: Record<string, boolean>) =>
    d?.recap && run('rate', () => api(`/api/recaps/${d.recap!.id}/rating`, json({ rating, flag_helpful: flagHelpful ?? null })))

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
      <p className="muted small">
        {[
          L.status,
          L.asr_model ? `${L.asr_model} on ${L.power_state}` : null,
          L.asr_rtf_p95 ? `speech-to-text at ${(1 / L.asr_rtf_p95).toFixed(0)}× real time` : null,
          drain !== null && drain > 0 ? `battery −${drain}%` : null,
          `${d.segments.length} lines`,
          `${d.flags.length} flag${d.flags.length === 1 ? '' : 's'}`,
          d.gaps.length ? `${d.gaps.length} gap${d.gaps.length === 1 ? '' : 's'}` : null,
          totalCost > 0 ? `model spend $${totalCost.toFixed(3)}` : 'model spend $0',
        ]
          .filter(Boolean)
          .join(' · ')}
      </p>

      <div className="controls">
        <a className="button" href={`/api/lectures/${id}/export.md`}>
          Export Markdown
        </a>
        <a className="button" href={`/api/lectures/${id}/export.json`} download={`lecture-${id}.json`}>
          Export JSON
        </a>
        <button
          className="danger"
          disabled={busy !== null}
          onClick={() => {
            if (window.confirm('Delete this lecture and everything derived from it? This cannot be undone.'))
              run('delete', async () => {
                await api(`/api/lectures/${id}`, { method: 'DELETE' })
                window.location.reload()
              })
          }}
        >
          Delete lecture
        </button>
      </div>

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
          Add slides (PDF or PPTX)
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
            Index the slides
          </button>
        )}
        <button onClick={() => run('align', async () => setAlign(await api(`/api/lectures/${id}/align`, { method: 'POST' })))} disabled={busy !== null || !d.deck}>
          Match transcript to slides
        </button>
      </div>
      {align && (
        <p className="muted">
          {align.note ?? `coverage ${(align.coverage * 100).toFixed(0)}%`}
          {align.deck_mismatch ? ' · slides may not match this lecture' : ''}
          {align.off_slide.length ? ` · off-slide: ${align.off_slide.map((o) => `${fmt(o.t0)}–${fmt(o.t1)}`).join(', ')}` : ''}
        </p>
      )}

      <h3>Deadlines</h3>
      <div className="controls">
        <button onClick={() => run('extract', async () => setExtract(await api(`/api/lectures/${id}/extract`, { method: 'POST' })))} disabled={busy !== null || !llmAvailable}>
          {busy === 'extract' ? 'Looking…' : 'Find deadlines'}
        </button>
        {!llmAvailable && <span className="muted">No model is set up yet; see the README.</span>}
        {extract && (
          <span className="muted">
            {extract.note ??
              `${extract.chunks} window${extract.chunks === 1 ? '' : 's'}, ${extract.candidates} mentions → ${extract.suggestions.one_tap} ready, ${extract.suggestions.maybe} to check, ${extract.suggestions.log} ignored${extract.suggestions.existing ? `, ${extract.suggestions.existing} already in inbox` : ''}`}
          </span>
        )}
        {d.suggestions.length > 0 && <span className="muted">{d.suggestions.length} suggestion(s) from this lecture are in the Inbox.</span>}
      </div>

      <h3>Recap</h3>
      <div className="controls">
        <button className={d.recap ? '' : 'primary'} onClick={() => run('recap', () => api(`/api/lectures/${id}/recap`, { method: 'POST' }))} disabled={busy !== null || !llmAvailable}>
          {busy === 'recap' ? (progress ? `Writing… ${progress.done}/${progress.total} (${progress.stage})` : 'Starting…') : d.recap ? 'Write recap again' : 'Write recap'}
        </button>
        {d.recap && (
          <span className="muted">
            v{d.recap.version} · {d.recap.model} · rate it:
            {[1, 2, 3, 4, 5].map((n) => (
              <button key={n} className={`small ${d.recap!.rating === n ? 'primary' : ''}`} onClick={() => rate(n)} disabled={busy !== null}>
                {n}
              </button>
            ))}
          </span>
        )}
      </div>
      {d.recap && (
        <div className="recap">
          <h4>{d.recap.sections.title}</h4>
          {d.recap.sections.gaps_note && <p className="warn-text small">Recording gap: {d.recap.sections.gaps_note}</p>}
          <h5>Highlights</h5>
          <ul>
            {d.recap.sections.highlights.map((h, i) => (
              <li key={i}>
                {h.replace(/\s*\(t=\d+:\d{2}\)\s*$/, '')}
                <Sources list={(h.match(/t=\d+:\d{2}/g) ?? []) as string[]} jump={jump} />
              </li>
            ))}
          </ul>
          {d.recap.sections.flag_explanations.length > 0 && (
            <>
              <h5>What you flagged</h5>
              {d.recap.sections.flag_explanations.map((f) => (
                <div key={f.flag_id} className="flagx">
                  <div>
                    <button className="chip link" onClick={() => jump(f.t)}>
                      {fmt(f.t)}
                    </button>{' '}
                    <b>{f.what_was_confusing ?? f.error}</b>
                  </div>
                  {f.explanation && <p>{f.explanation}</p>}
                  {f.prerequisite && <p className="muted small">Check first: {f.prerequisite}</p>}
                  <Sources list={f.sources} jump={jump} />
                  <span className="helpful">
                    Helpful?
                    <button className={`small ${d.recap!.flag_helpful[String(f.flag_id)] === true ? 'primary' : ''}`} onClick={() => rate(null, { ...d.recap!.flag_helpful, [f.flag_id]: true })}>
                      Yes
                    </button>
                    <button className={`small ${d.recap!.flag_helpful[String(f.flag_id)] === false ? 'danger' : ''}`} onClick={() => rate(null, { ...d.recap!.flag_helpful, [f.flag_id]: false })}>
                      No
                    </button>
                  </span>
                </div>
              ))}
            </>
          )}
          <h5>Key concepts</h5>
          {d.recap.sections.concepts.map((c, i) => (
            <div key={i} className={`concept ${c.importance}`}>
              <b>{c.name}</b> <span className="chip">{c.importance}</span>
              <p>{c.explanation}</p>
              <Sources list={c.sources} jump={jump} />
            </div>
          ))}
          {d.recap.sections.off_slide_notes.length > 0 && (
            <>
              <h5>Off the slides</h5>
              <ul>
                {d.recap.sections.off_slide_notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </>
          )}
          <h5>Review questions</h5>
          {d.recap.sections.review_questions.map((q, i) => (
            <div key={i} className="q">
              <button className="linkish" onClick={() => setOpenQ((o) => ({ ...o, [i]: !o[i] }))}>
                {openQ[i] ? '▾' : '▸'} {q.question}
              </button>
              {openQ[i] && (
                <p>
                  {q.answer} <Sources list={q.sources} jump={jump} />
                </p>
              )}
            </div>
          ))}
          <h5>Ask this lecture</h5>
          <div className="controls">
            <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. why does the fugacity coefficient approach 1 at low pressure?" style={{ flex: 1 }} />
            <button onClick={() => run('ask', async () => setAnswer(await api(`/api/lectures/${id}/ask`, json({ question }))))} disabled={busy !== null || !question.trim()}>
              {busy === 'ask' ? 'Thinking…' : 'Ask'}
            </button>
          </div>
          {answer && (
            <div className="answer">
              <p>{answer.answer}</p>
              {answer.coverage === 'not_in_lecture' && <p className="warn-text small">The lecture material doesn't cover this.</p>}
              {answer.coverage === 'partly_from_lecture' && <p className="muted small">Only partly covered by the lecture.</p>}
              <Sources list={answer.sources} jump={jump} />
            </div>
          )}
        </div>
      )}

      <h3>Whiteboard photos</h3>
      <div className="controls">
        <label className="upload">
          Add board photos
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
        {!llmAvailable && <span className="muted">Needs a model. Photos are read once and never stored.</span>}
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
      <section className="transcript tall" ref={transcriptRef}>
        {d.segments.map((s) => {
          const flagged = d.flags.some((f) => s.t1 > f.window_t0 && s.t0 < f.window_t1)
          const slide = d.alignment.find((w) => s.t0 >= w.t0 && s.t0 < w.t1)?.slide
          const lit = highlightT !== null && s.t0 <= highlightT + 0.5 && s.t1 >= highlightT - 30
          return (
            <p key={s.id} id={`seg-${s.id}`} className={`${flagged ? 'flagged' : s.trigger_terms.length ? 'hit' : ''} ${lit ? 'lit' : ''}`}>
              <span className="t">{fmt(s.t0)}</span>
              {slide ? <span className="slide">slide {slide}</span> : null}
              <span className="tx">{s.text}</span>
            </p>
          )
        })}
      </section>
    </>
  )
}
