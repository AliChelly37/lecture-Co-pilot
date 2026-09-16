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
  const [view, setView] = useState<'live' | 'lectures' | 'inbox'>('live')
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
          <button className={view === 'inbox' ? 'tab active' : 'tab'} onClick={() => setView('inbox')}>
            Inbox
          </button>
        </nav>
        <StatusPills status={status} connected={connected} />
      </header>
      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      {view === 'live' && <Live status={status} setStatus={setStatus} refresh={refresh} setConnected={setConnected} setError={setError} />}
      {view === 'lectures' && <Lectures setError={setError} llmAvailable={!!status?.llm_available} />}
      {view === 'inbox' && <Inbox setError={setError} />}
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
        <h2>Ready to confirm ({oneTap.length})</h2>
        {oneTap.length === 0 && <p className="muted">Nothing waiting. Run “Extract deadlines” on a lecture.</p>}
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

        <h2>Check these ({maybe.length})</h2>
        {maybe.length === 0 && <p className="muted">Nothing uncertain.</p>}
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
        {targets.length === 0 && <p className="muted small">No external targets enabled (set LC_TARGETS=gcal,notion). Confirmed items export as .ics.</p>}
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

      <h3>Deadlines</h3>
      <div className="controls">
        <button onClick={() => run('extract', async () => setExtract(await api(`/api/lectures/${id}/extract`, { method: 'POST' })))} disabled={busy !== null || !llmAvailable}>
          {busy === 'extract' ? 'Extracting…' : 'Extract deadlines (model)'}
        </button>
        {!llmAvailable && <span className="muted">No model provider available.</span>}
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
          {busy === 'recap' ? (progress ? `Working… ${progress.done}/${progress.total} (${progress.stage})` : 'Starting…') : d.recap ? 'Regenerate recap' : 'Generate recap (model)'}
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
          {d.recap.sections.gaps_note && <p className="warn-text small">Recording note: {d.recap.sections.gaps_note}</p>}
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
      <section className="transcript tall" ref={transcriptRef}>
        {d.segments.map((s) => {
          const flagged = d.flags.some((f) => s.t1 > f.window_t0 && s.t0 < f.window_t1)
          const slide = d.alignment.find((w) => s.t0 >= w.t0 && s.t0 < w.t1)?.slide
          const lit = highlightT !== null && s.t0 <= highlightT + 0.5 && s.t1 >= highlightT - 30
          return (
            <p key={s.id} id={`seg-${s.id}`} className={`${flagged ? 'flagged' : s.trigger_terms.length ? 'hit' : ''} ${lit ? 'lit' : ''}`}>
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
