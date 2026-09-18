import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
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
type ReplayStatus = {
  filename: string
  duration_s: number
  transcribed_s: number
  done: boolean
  heard_s: number
}
type Status = {
  recording: boolean
  source: 'mic' | 'replay' | 'youtube' | null
  paused: boolean
  lecture: { id: string } | null
  elapsed_s: number
  badge: number
  asr: Asr | null
  replay: ReplayStatus | null
  slides: 'reading' | 'ready' | 'none' | null
  hotkeys: boolean
  power: { state: 'ac' | 'battery'; battery: number | null }
  llm_available: boolean
  llm?: { provider: string; text_model: string; local: boolean }
}
type Course = { id: string; name: string; timezone: string }
type Segment = {
  id: number
  t0: number
  t1: number
  text: string
  conf: number
  trigger: string[]
}
type Flag = { id: number; t: number; window_t0: number; window_t1: number }
type LectureRow = {
  id: string
  course_name: string
  started_at: string
  ended_at: string | null
  status: string
  asr_model: string
  source: 'mic' | 'replay' | 'youtube'
}
type Deck = {
  id: string
  filename: string
  slide_count: number
  indexed: boolean
}
type Capture = {
  id: string
  t_shutter: number | null
  content_kind: string | null
  text: string | null
  legibility: number | null
  status: string
}
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
type FlagExplanationT = {
  flag_id: number
  t: number
  what_was_confusing?: string
  explanation?: string
  prerequisite?: string
  sources?: string[]
  error?: string
}
type RecapT = {
  id: string
  version: number
  model: string
  rating: number | null
  flag_helpful: Record<string, boolean>
  sections: {
    title: string
    highlights: string[]
    concepts: {
      name: string
      importance: 'high' | 'medium' | 'low'
      explanation: string
      sources: string[]
    }[]
    review_questions: { question: string; answer: string; sources: string[] }[]
    off_slide_notes: string[]
    gaps_note: string
    flag_explanations: FlagExplanationT[]
    detected_events: string[]
  }
}
type AnswerT = {
  answer: string
  sources: string[]
  coverage: 'answered_from_lecture' | 'partly_from_lecture' | 'not_in_lecture'
}
type ExtractSummary = {
  chunks: number
  candidates: number
  surfaced: number
  suggestions: {
    one_tap: number
    maybe: number
    log: number
    existing: number
  }
  note?: string
}
type Detail = {
  lecture: LectureRow & {
    deck_id: string | null
    power_state: string
    battery_start: number | null
    battery_end: number | null
    asr_rtf_p95: number | null
    source_url: string | null
  }
  segments: {
    id: number
    t0: number
    t1: number
    text: string
    asr_conf: number | null
    trigger_terms: string[]
    trigger_score: number
  }[]
  flags: Flag[]
  gaps: { t0: number; t1: number | null; cause: string }[]
  deck: (Deck & { slides_text: string[] }) | null
  captures: Capture[]
  alignment: { t0: number; t1: number; slide: number | null; score: number }[]
  suggestions: Suggestion[]
  recap: RecapT | null
  usage: {
    stage: string
    model: string
    calls: number
    cost_usd: number
    cache_read: number
    input_tokens: number
  }[]
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
const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

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
  const [live, setLive] = useState<LiveState>({
    segments: [],
    flags: [],
    badge: 0,
  })
  const [ytProgress, setYtProgress] = useState<{
    stage: string
    done: number
    total: number
  } | null>(null)
  const refresh = useCallback(async () => setStatus(await api<Status>('/api/status')), [])
  const replay = useReplay(status)
  // Connected in the middle of a lecture (a reload, a second tab): bring back what was already said and flagged.
  const restore = useCallback(async (id: string) => {
    const d = await api<Detail>(`/api/lectures/${id}`).catch(() => null)
    if (!d) return
    setLive((l) => {
      const segIds = new Set(l.segments.map((x) => x.id))
      const flagIds = new Set(l.flags.map((x) => x.id))
      const old = d.segments
        .filter((x) => !segIds.has(x.id))
        .map((x) => ({ id: x.id, t0: x.t0, t1: x.t1, text: x.text, conf: x.asr_conf ?? 1, trigger: x.trigger_terms }))
      return {
        ...l,
        segments: [...old, ...l.segments].sort((a, b) => a.t0 - b.t0),
        flags: [...d.flags.filter((x) => !flagIds.has(x.id)), ...l.flags].sort((a, b) => a.t - b.t),
      }
    })
  }, [])

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
          if (ev.recording && ev.lecture) restore(ev.lecture.id)
        } else if (ev.type === 'segment') {
          setLive((l) => ({
            ...l,
            segments: [...l.segments, ev.segment],
            badge: ev.badge ?? l.badge,
          }))
        } else if (ev.type === 'flag') {
          setLive((l) => ({ ...l, flags: [...l.flags, ev.flag] }))
        } else if (ev.type === 'lecture_started') {
          setLive({ segments: [], flags: [], badge: 0 })
          setYtProgress(null)
          refresh()
        } else if (ev.type === 'progress' && ev.job === 'youtube') {
          setYtProgress({ stage: ev.stage, done: ev.done, total: ev.total })
        } else if (ev.type === 'slides_ready') {
          setYtProgress(null)
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
  }, [refresh, restore])

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
      {view === 'live' && <Live status={status} live={live} refresh={refresh} setError={setError} replay={replay} ytProgress={ytProgress} />}
      {view === 'lectures' && <Lectures setError={setError} llmAvailable={!!status?.llm_available} initialId={lectureFromHash()} />}
      {view === 'inbox' && <Inbox setError={setError} />}
      {view === 'dashboard' && <Dashboard setError={setError} />}
      <footer className="muted">Audio never leaves this laptop. During class, nothing is sent anywhere.</footer>
      {/* Lives in the shell so a recording keeps playing while you look at another tab. */}
      <audio ref={replay.audioRef} src={replay.file?.url} preload="auto" />
    </div>
  )
}

// ---------- replay: the browser plays the file, the backend keeps the clock ----------
const AUDIO_TYPES = 'audio/*,video/*,.m4a,.mp3,.wav,.ogg,.opus,.webm,.mp4,.aac,.flac'

type ReplayFile = { url: string; name: string }
type ReplayCtl = {
  file: ReplayFile | null
  open: (f: File, opts: { autoplay: boolean; startAt?: number }) => void
  close: () => void
  audioRef: React.RefObject<HTMLAudioElement | null>
  pos: number
  heard: number
  playing: boolean
  ended: boolean
  toggle: () => void
  seek: (t: number) => void
}

// Resolves with the duration when this browser can play the file. Checked before anything is uploaded.
function playable(f: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const a = document.createElement('audio')
    const url = URL.createObjectURL(f)
    const settle = (fn: () => void) => {
      window.clearTimeout(timer)
      a.onloadedmetadata = null
      a.onerror = null
      a.removeAttribute('src')
      a.load()
      URL.revokeObjectURL(url)
      fn()
    }
    const timer = window.setTimeout(() => settle(() => reject(new Error(`${f.name} did not open in this browser.`))), 20000)
    a.preload = 'metadata'
    a.onloadedmetadata = () => {
      const d = a.duration
      settle(() => resolve(d))
    }
    a.onerror = () => settle(() => reject(new Error(`This browser can't play ${f.name}. Convert it to mp3 or m4a and open it again.`)))
    a.src = url
  })
}

function useReplay(status: Status | null): ReplayCtl {
  const audioRef = useRef<HTMLAudioElement>(null)
  const urlRef = useRef<string | null>(null)
  const pending = useRef<{ autoplay: boolean; startAt: number } | null>(null)
  const lastReport = useRef(0)
  const [file, setFile] = useState<ReplayFile | null>(null)
  const [pos, setPos] = useState(0)
  const [heard, setHeard] = useState(0) // furthest point reached: lines stay visible after a seek back
  const [playing, setPlaying] = useState(false)
  const [ended, setEnded] = useState(false)
  const inReplay = !!status?.recording && (status.source === 'replay' || status.source === 'youtube')
  const backendPaused = status?.paused
  const backendClock = status?.elapsed_s

  // The file stays in this browser; only play, pause and the position reach the backend.
  const report = useCallback((isPlaying: boolean, t: number) => {
    lastReport.current = performance.now()
    api('/api/lectures/playback', json({ playing: isPlaying, t })).catch(() => undefined)
  }, [])

  const setSource = useCallback((f: File | null) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    urlRef.current = f ? URL.createObjectURL(f) : null
    setFile(f && urlRef.current ? { url: urlRef.current, name: f.name } : null)
  }, [])

  const open = useCallback(
    (f: File, opts: { autoplay: boolean; startAt?: number }) => {
      const startAt = opts.startAt ?? 0
      pending.current = { autoplay: opts.autoplay, startAt }
      setPos(startAt)
      setHeard(startAt)
      setEnded(false)
      setSource(f)
    },
    [setSource],
  )
  const close = useCallback(() => {
    audioRef.current?.pause()
    pending.current = null
    setSource(null)
    setPos(0)
    setHeard(0)
    setPlaying(false)
    setEnded(false)
  }, [setSource])

  // The audio element is the truth for play, pause and position.
  useEffect(() => {
    const a = audioRef.current
    if (!a || !file) return
    const onPlay = () => {
      setPlaying(true)
      setEnded(false)
      report(true, a.currentTime)
    }
    const onPause = () => {
      setPlaying(false)
      report(false, a.currentTime)
    }
    const onSeeked = () => {
      setPos(a.currentTime)
      report(!a.paused, a.currentTime)
    }
    const onTime = () => {
      setPos(a.currentTime)
      setHeard((h) => Math.max(h, a.currentTime))
      if (!a.paused && performance.now() - lastReport.current > 2000) report(true, a.currentTime)
    }
    const onEnded = () => setEnded(true)
    const onMeta = () => {
      const p = pending.current
      pending.current = null
      if (!p) return
      if (p.startAt > 0) a.currentTime = p.startAt
      if (p.autoplay) a.play().catch(() => undefined) // a blocked autoplay leaves the Play button
    }
    // Closing or reloading the page stops the audio; stop the backend clock with it.
    const onHide = () => {
      if (a.paused) return
      navigator.sendBeacon(
        '/api/lectures/playback',
        new Blob([JSON.stringify({ playing: false, t: a.currentTime })], {
          type: 'application/json',
        }),
      )
    }
    const media: [string, () => void][] = [
      ['play', onPlay],
      ['pause', onPause],
      ['seeked', onSeeked],
      ['timeupdate', onTime],
      ['ended', onEnded],
      ['loadedmetadata', onMeta],
    ]
    media.forEach(([name, fn]) => a.addEventListener(name, fn))
    window.addEventListener('pagehide', onHide)
    if (a.readyState >= 1) onMeta() // the metadata can arrive before this effect runs
    return () => {
      media.forEach(([name, fn]) => a.removeEventListener(name, fn))
      window.removeEventListener('pagehide', onHide)
    }
  }, [file, report])

  // F10 is a global hotkey the backend handles; mirror its pause state here.
  useEffect(() => {
    const a = audioRef.current
    if (!a || !file || !inReplay || backendPaused === undefined) return
    if (backendPaused && !a.paused) a.pause()
    else if (!backendPaused && a.paused && !a.ended && !pending.current) a.play().catch(() => undefined)
  }, [backendPaused, inReplay, file])

  // After a reload the audio is gone, but the backend clock may still be running: stop it until the file is back.
  useEffect(() => {
    if (inReplay && !file && backendPaused === false) report(false, backendClock ?? 0)
  }, [inReplay, file, backendPaused, backendClock, report])

  // The session ended (Finish here, or anywhere else): let go of the file.
  useEffect(() => {
    if (status && !inReplay && file) close()
  }, [status, inReplay, file, close])

  const toggle = useCallback(() => {
    const a = audioRef.current
    if (!a) return
    if (a.paused) a.play().catch(() => undefined)
    else a.pause()
  }, [])
  const seek = useCallback((t: number) => {
    const a = audioRef.current
    if (a && Number.isFinite(t)) a.currentTime = Math.max(0, t)
  }, [])

  return {
    file,
    open,
    close,
    audioRef,
    pos,
    heard,
    playing,
    ended,
    toggle,
    seek,
  }
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
  replay,
  ytProgress,
}: {
  status: Status | null
  live: LiveState
  refresh: () => Promise<void>
  setError: (e: string | null) => void
  replay: ReplayCtl
  ytProgress: { stage: string; done: number; total: number } | null
}) {
  const { segments, flags, badge } = live
  const [courses, setCourses] = useState<Course[]>([])
  const [courseId, setCourseId] = useState('')
  const [newCourse, setNewCourse] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [opening, setOpening] = useState<string | null>(null)
  const [slides, setSlides] = useState<{
    lectureId: string
    texts: string[]
    windows: { t0: number; t1: number; slide: number | null }[]
  } | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api<Course[]>('/api/courses')
      .then((c) => {
        setCourses(c)
        setCourseId((cur) => cur || (c[0]?.id ?? ''))
      })
      .catch((e) => setError(String(e)))
  }, [setError])

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
  const lectureId = status?.lecture?.id ?? null
  const isYoutube = recording && status?.source === 'youtube'
  const isPlayback = recording && (status?.source === 'replay' || status?.source === 'youtube')
  const rs = status?.replay ?? null
  const dur = rs?.duration_s ?? 0
  // Playback: the transcript runs ahead of the audio, so a line (and its badge tick)
  // appears once the recording has reached it, and stays after a seek back.
  const reach = Math.max(replay.heard, rs?.heard_s ?? 0) // the backend remembers it across a reload
  const shown = isPlayback ? segments.filter((s) => s.t0 <= reach + 0.3) : segments
  const playedTo = shown.filter((s) => s.t0 <= replay.pos + 0.3)
  const nowId = isPlayback && replay.file ? playedTo[playedTo.length - 1]?.id : undefined
  const shownBadge = isPlayback ? shown.filter((s) => s.trigger.length > 0).length : badge
  const pct = (t: number) => `${dur ? Math.min(100, (t / dur) * 100) : 0}%`

  // A YouTube import already has its deck: fetch it once (no attach/align click needed) and
  // work out which slide covers the played-to position, so the slide "follows the sound".
  useEffect(() => {
    if (!isYoutube || !lectureId || status?.slides !== 'ready' || slides?.lectureId === lectureId) return
    api<Detail>(`/api/lectures/${lectureId}`)
      .then((d) => {
        if (d.deck)
          setSlides({
            lectureId,
            texts: d.deck.slides_text,
            windows: d.alignment,
          })
      })
      .catch(() => undefined)
  }, [isYoutube, lectureId, status?.slides, slides])
  useEffect(() => {
    if (!recording) setSlides(null)
  }, [recording])
  const atPos = replay.file ? replay.pos : (status?.elapsed_s ?? 0)
  const currentSlide = slides?.windows.find((w) => atPos >= w.t0 && atPos < w.t1)?.slide ?? null
  const currentSlideText = currentSlide ? slides?.texts[currentSlide - 1] : null

  // A page reload drops the audio Blob; a replayed upload asks for the file back, but a
  // YouTube import can just be re-fetched from the backend, which still has it in memory.
  useEffect(() => {
    if (!isYoutube || !lectureId || replay.file || busy) return
    let cancelled = false
    fetch(`/api/lectures/${lectureId}/audio`)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error('audio no longer available'))))
      .then((blob) => {
        if (!cancelled)
          replay.open(new File([blob], 'lecture-audio', { type: blob.type }), {
            autoplay: false,
            startAt: status?.elapsed_s ?? 0,
          })
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [isYoutube, lectureId, replay, busy, status?.elapsed_s])

  useEffect(() => {
    const box = listRef.current
    if (!box) return
    if (!isPlayback) {
      box.scrollTo({ top: box.scrollHeight })
      return
    }
    // Keep the line being played in view, including after a seek back.
    const el = box.querySelector<HTMLElement>('p.now')
    if (!el) return
    const b = box.getBoundingClientRect()
    const r = el.getBoundingClientRect()
    if (r.top < b.top || r.bottom > b.bottom)
      box.scrollTo({
        top: box.scrollTop + (r.top - b.top) - box.clientHeight / 3,
      })
  }, [shown.length, nowId, isPlayback])

  const openRecording = (f: File) =>
    run(async () => {
      setOpening(f.name)
      try {
        await playable(f) // a file this browser can't play is refused before anything starts
        const q = new URLSearchParams({
          course_id: courseId,
          filename: f.name,
        })
        await api(`/api/lectures/replay?${q}`, {
          method: 'POST',
          headers: { 'content-type': 'application/octet-stream' },
          body: f,
        })
        await refresh()
        replay.open(f, { autoplay: true })
      } finally {
        setOpening(null)
      }
    })
  const reopen = (f: File) =>
    run(async () => {
      const d = await playable(f)
      if (dur && Number.isFinite(d) && Math.abs(d - dur) > Math.max(5, dur * 0.02)) {
        throw new Error(`${f.name} is ${fmt(d)} long, but the recording you were listening to is ${fmt(dur)}. Open that one.`)
      }
      replay.open(f, { autoplay: false, startAt: status?.elapsed_s ?? 0 })
    })
  const importYoutube = () =>
    run(async () => {
      const url = youtubeUrl.trim()
      setOpening('the video')
      try {
        const r = await api<{ lecture: { id: string }; duration_s: number }>('/api/lectures/youtube', json({ course_id: courseId, url }))
        setYoutubeUrl('')
        await refresh()
        const blob = await (await fetch(`/api/lectures/${r.lecture.id}/audio`)).blob()
        replay.open(new File([blob], 'lecture-audio', { type: blob.type }), {
          autoplay: true,
        })
      } finally {
        setOpening(null)
      }
    })
  const finish = () =>
    run(async () => {
      const ended = await api<{ id: string }>('/api/lectures/stop', {
        method: 'POST',
      })
      replay.close()
      location.hash = `#lectures/${ended.id}`
    })
  const flagNow = () => run(() => api('/api/lectures/flag', isPlayback ? json({ t: replay.audioRef.current?.currentTime ?? replay.pos }) : { method: 'POST' }))
  const liquid = (e: React.PointerEvent<HTMLButtonElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    e.currentTarget.style.setProperty('--mx', `${((e.clientX - r.left) / r.width) * 100}%`)
    e.currentTarget.style.setProperty('--my', `${((e.clientY - r.top) / r.height) * 100}%`)
  }
  const pick = (then: (f: File) => void) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    e.target.value = '' // picking the same file again must still fire
    if (f) then(f)
  }

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
            <label className={`upload ${!courseId || busy ? 'off' : ''}`} title="Missed the class? Play someone's recording here and use the same buttons.">
              {opening === 'the video' ? 'Opening…' : opening ? `Opening ${opening}…` : 'Catch up on a recording'}
              <input type="file" accept={AUDIO_TYPES} hidden disabled={!courseId || busy} onChange={pick(openRecording)} />
            </label>
            <div className="yt-import">
              <input type="url" placeholder="Or paste a YouTube link…" value={youtubeUrl} onChange={(e) => setYoutubeUrl(e.target.value)} disabled={!courseId || busy} />
              <button onClick={importYoutube} disabled={!courseId || !youtubeUrl.trim() || busy}>
                {opening === 'the video' ? (ytProgress ? `${ytProgress.stage} ${ytProgress.total ? `${ytProgress.done}/${ytProgress.total}` : '…'}` : 'Downloading…') : 'Import'}
              </button>
            </div>
          </>
        ) : isPlayback ? (
          <>
            <span className={`rec replay ${replay.playing ? '' : 'paused'}`}>
              {replay.ended ? 'ENDED' : replay.playing ? 'PLAYING' : 'PAUSED'} · {fmt(atPos)} / {fmt(dur)}
            </span>
            {replay.file ? (
              <>
                <button className="flag" onClick={flagNow} onPointerMove={liquid} disabled={busy}>
                  I didn't get that <kbd>F9</kbd>
                </button>
                <button onClick={replay.toggle} disabled={busy}>
                  {replay.playing ? 'Pause (F10)' : replay.ended ? 'Play again' : 'Play (F10)'}
                </button>
              </>
            ) : isYoutube ? (
              <span className="muted">Reconnecting the audio…</span>
            ) : (
              <label className={`upload ${busy ? 'off' : ''}`}>
                Open {rs?.filename ?? 'the recording'} again to keep listening
                <input type="file" accept={AUDIO_TYPES} hidden disabled={busy} onChange={pick(reopen)} />
              </label>
            )}
            <button className="danger" onClick={finish} disabled={busy}>
              Finish
            </button>
            <span className="badge" title="Possible deadline mentions (deterministic filter; confirmed after class)">
              {shownBadge} possible deadline{shownBadge === 1 ? '' : 's'}
            </span>
            <div
              className="tape"
              style={
                {
                  '--pos': pct(atPos),
                  '--done': pct(rs?.transcribed_s ?? 0),
                } as CSSProperties
              }
            >
              <div className="ticks" aria-hidden>
                {dur > 0 && flags.map((f) => <i key={f.id} style={{ left: pct(f.t) }} />)}
              </div>
              <input
                className="scrub"
                type="range"
                min={0}
                max={dur || 1}
                step={0.1}
                value={Math.min(replay.pos, dur || 1)}
                onChange={(e) => replay.seek(Number(e.target.value))}
                disabled={!replay.file}
                aria-label="Position in the recording"
                aria-valuetext={`${fmt(replay.pos)} of ${fmt(dur)}`}
              />
              <span className="tape-note">
                {rs?.filename}
                {rs ? (rs.done ? ' · fully transcribed' : ` · transcribed to ${fmt(rs.transcribed_s)}`) : ''}
                {replay.ended
                  ? ' · Finish to find deadlines and write the recap'
                  : status?.slides === 'reading'
                    ? ` · reading the slides${ytProgress?.total ? ` ${ytProgress.done}/${ytProgress.total}` : '…'}`
                    : isYoutube
                    ? ' · downloaded once, kept only in this browser; only the transcript is saved'
                    : ' · the file stays in this browser; only the transcript is saved'}
              </span>
            </div>
            {currentSlideText !== null && currentSlideText !== undefined && (
              <div className="slide-now">
                <b>Slide {currentSlide}</b>
                <span>{currentSlideText}</span>
              </div>
            )}
          </>
        ) : (
          <>
            <span className={`rec ${status?.paused ? 'paused' : ''}`}>
              {status?.paused ? 'PAUSED' : 'RECORDING'} · {fmt(status?.elapsed_s ?? 0)}
            </span>
            <button className="flag" onClick={flagNow} onPointerMove={liquid} disabled={busy}>
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
          {shown.length === 0 && (
            <div className="empty">
              {isPlayback ? (
                <>
                  <b>{!replay.file ? 'Opening…' : replay.playing ? 'Listening' : 'Press play'}</b>
                  <span>Each line appears when the recording reaches it. Flag and pause work the same way they do in class.</span>
                </>
              ) : recording ? (
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
                  <span>Missed the class? Open a recording, or paste a public YouTube link, and use the same buttons while you listen.</span>
                </>
              )}
            </div>
          )}
          {shown.map((s) => (
            <p key={s.id} className={`${s.trigger.length ? 'hit' : ''} ${s.id === nowId ? 'now' : ''}`} title={`conf ${s.conf}`}>
              <span className="t">{fmt(s.t0)}</span>
              <span className="tx">{s.text}</span>
              {s.trigger.length > 0 && <span className="terms">might be a deadline: {s.trigger.join(', ')}</span>}
            </p>
          ))}
        </section>
        <aside>
          <h2>Flags</h2>
          {flags.length === 0 && <p className="muted">No confusion flags yet.</p>}
          {flags.map((f) =>
            isPlayback && replay.file ? (
              <p key={f.id}>
                <button className="linkish" onClick={() => replay.seek(Math.max(0, f.t - 10))} title="Hear the 10 seconds before this flag again">
                  <span className="t">{fmt(f.t)}</span> window {fmt(f.window_t0)}–{fmt(f.window_t1)}
                </button>
              </p>
            ) : (
              <p key={f.id}>
                <span className="t">{fmt(f.t)}</span> window {fmt(f.window_t0)}–{fmt(f.window_t1)}
              </p>
            ),
          )}
        </aside>
      </main>
    </>
  )
}

// ---------- dashboard ----------
type DashboardT = {
  courses: {
    id: string
    name: string
    lectures: number
    last_lecture: string | null
    open_flags: number
    pending_suggestions: number
    recaps: number
  }[]
  inbox: { one_tap: number; maybe: number }
  upcoming: Suggestion[]
  usage: {
    by_stage: {
      stage: string
      model: string
      calls: number
      cost_usd: number
      input_tokens: number
      output_tokens: number
    }[]
    total_cost_usd: number
    calls: number
  }
  asr: {
    rtf_p95_avg: number | null
    battery_drain_pct_per_hour: number | null
    recorded_hours: number
  }
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
type TargetHealthT = {
  name: string
  configured: boolean
  connected: boolean
  detail: string
}

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
          const ext =
            (
              s.payload as unknown as {
                external?: Record<string, { id: string; url: string | null }>
              }
            ).external ?? {}
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
              {r.source === 'replay' ? ' · from a recording' : r.source === 'youtube' ? ' · from a YouTube video' : ''}
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
  const [align, setAlign] = useState<{
    coverage: number
    off_slide: { t0: number; t1: number }[]
    deck_mismatch?: boolean
    note?: string
  } | null>(null)
  const [extract, setExtract] = useState<ExtractSummary | null>(null)
  const [progress, setProgress] = useState<{
    done: number
    total: number
    stage: string
  } | null>(null)
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
          L.source === 'replay' ? 'listened from a recording' : L.source === 'youtube' ? 'imported from YouTube' : null,
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
      {L.source_url && (
        <p className="muted small">
          Source:{' '}
          <a href={L.source_url} target="_blank" rel="noreferrer">
            {L.source_url}
          </a>
        </p>
      )}

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
                const deck = await api<Deck>('/api/decks', {
                  method: 'POST',
                  body: fd,
                })
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
      {d.deck && d.deck.slides_text.length > 0 && (
        <details className="slides-list">
          <summary>
            {d.deck.slides_text.length} slides read from {d.deck.filename}
          </summary>
          {d.deck.slides_text.map((text, i) => (
            <p key={i}>
              <span className="slide">slide {i + 1}</span>
              <span className="tx">{text}</span>
            </p>
          ))}
        </details>
      )}
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
                    <button
                      className={`small ${d.recap!.flag_helpful[String(f.flag_id)] === true ? 'primary' : ''}`}
                      onClick={() =>
                        rate(null, {
                          ...d.recap!.flag_helpful,
                          [f.flag_id]: true,
                        })
                      }
                    >
                      Yes
                    </button>
                    <button
                      className={`small ${d.recap!.flag_helpful[String(f.flag_id)] === false ? 'danger' : ''}`}
                      onClick={() =>
                        rate(null, {
                          ...d.recap!.flag_helpful,
                          [f.flag_id]: false,
                        })
                      }
                    >
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
              {slide ? (
                <span className="slide" title={d.deck?.slides_text[slide - 1]}>
                  slide {slide}
                </span>
              ) : null}
              <span className="tx">{s.text}</span>
            </p>
          )
        })}
      </section>
    </>
  )
}
