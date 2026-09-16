# M0 — Whisper benchmark on the target laptop

Machine: Lenovo, i5-12450HX (8C/12T), RTX 4050 Laptop 6 GB, 16 GB RAM,
Windows 11, NVIDIA driver 580.97. faster-whisper 1.2.1 / CTranslate2 4.8.2 with
pip-installed CUDA 12 + cuDNN 9 runtime DLLs (no system CUDA install).

## Run 1 — 2026-09-16, on battery, greedy decoding (beam 1), VAD filter on

Audio: `eval/fixtures/tts_lecture.wav`, 139.9 s of Windows TTS speech from
`tts_lecture.txt` (thermodynamics jargon, one deadline, one hypothetical, one
correction). **Caveat:** TTS audio is not lecture audio; WER here is only a
relative signal between models. Real WER comes from the OCW set (M6).

| Config | Load (s)* | RTF | Speed-up | WER (TTS) | Lang detected |
|---|---|---|---|---|---|
| **cuda / large-v3-turbo / int8_float16** | 143 | **0.077** | 13x | **0.109** | en |
| cuda / distil-large-v3 / int8_float16 | 163 | 0.024 | 42x | 0.206 | en |
| cuda / small / int8_float16 | 54 | 0.025 | 39x | 0.190 | fr (!) |
| cpu / small / int8 | 1.4 | 0.134 | 7.4x | 0.181 | fr (!) |
| cpu / base / int8 | 14 | 0.201 | 5.0x | 0.645 | fr (!) |
| cpu / distil-small.en / int8 | 29 | 0.077 | 13x | 0.234 | en |

\* Load times in run 1 include the one-off model download. Warm load is a few
seconds (re-measure in run 2).

## What the numbers decide

1. **Default on AC: `large-v3-turbo` on the GPU.** RTF 0.077 leaves >10x
   headroom over real time, with the best accuracy by a wide margin. The GPU
   is busy under 10% of the time.
2. **Language must be pinned.** Auto-detect picked French on three configs for
   English audio. The worker passes the course language (`en` default).
3. **`base` is unusable** (WER 0.65). The fallback model is `small`, not
   `base`. `small` on CPU keeps up at RTF 0.134, so a GPU failure still works.
4. **Battery default: `small`.** Chosen for power, not speed; whether turbo on
   the GPU drains the battery meaningfully is an open measurement (the app
   records battery % at lecture start/end for exactly this).
5. **Distil models** are fast but cost accuracy on this sample; they stay as
   eval candidates for the OCW run, not defaults.

## Next measurements
- Run 2: warm load times; same table on AC power.
- OCW lectures (M6): real WER, jargon error rate with and without `hotwords`,
  phantom segments per hour of silence/noise, and downstream date-extraction
  precision per model size.
- Battery drain per lecture-hour: turbo/GPU vs small/CPU, from real lectures.
