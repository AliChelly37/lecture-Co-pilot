# Extraction eval runs

| run | provider:model | cases | one-tap precision | one-tap recall | surfaced recall | traps → one-tap | false one-taps | needs-a-date → one-tap | dates exact | maybe surfaced | time | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260916-152305 | ollama:gemma3:4b+guards3 | 5 | 1.00 | 0.60 | 0.92 | 0 | 0 | 0 | 13/15 | 13/15 | 458s | $0.0000 |
| 20260916-153523 | ollama:gemma3:4b+guards4 | 5 | 1.00 | 0.68 | 0.92 | 0 | 0 | 0 | 16/17 | 12/15 | 456s | $0.0000 |

## Earlier runs (before the surfaced-recall and needs-a-date columns)

| run | provider:model | cases | one-tap precision | one-tap recall | traps → one-tap | false one-taps | dates exact | maybe surfaced | time | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 20260916-143753 | ollama:gemma3:4b | 5 | 0.71 | 0.48 | 1 | 4 | 9/12 | 15/15 | 576s | $0.0000 |
| 20260916-144711 | ollama:gemma3:4b | 5 | 0.60 | 0.48 | 1 | 7 | 9/12 | 15/15 | 518s | $0.0000 |
| 20260916-145745 | ollama:gemma3:4b+guards | 5 | 0.85 | 0.44 | 1 | 1 | 9/11 | 14/15 | 462s | $0.0000 |
| 20260916-151029 | ollama:gemma3:4b+guards2 | 5 | 1.00 | 0.68 | 0 | 0 | 13/17 | 14/15 | 474s | $0.0000 |
