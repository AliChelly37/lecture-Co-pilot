"""Runtime settings. Every value can be overridden with an LC_* environment
variable or a .env file in the working directory (e.g. LC_ASR_MODEL_AC=small).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LC_", env_file=".env", extra="ignore")

    # Where the SQLite database and exports live. Text only; never raw media.
    data_dir: Path = Path("data")

    host: str = "127.0.0.1"  # localhost only: nothing is reachable from the network
    port: int = 8765
    open_browser: bool = True

    # Speech-to-text (D12/D13). Model choice is a measured decision (see docs/ASR_BENCHMARK.md).
    asr_device: str = "auto"  # auto | cuda | cpu
    asr_model_ac: str = "large-v3-turbo"
    asr_model_battery: str = "small"
    asr_model_fallback: str = "small"  # used when the worker falls behind real time (`base` measured unusable: WER 0.65)
    asr_compute_cuda: str = "int8_float16"
    asr_compute_cpu: str = "int8"
    asr_beam_size: int = 1  # greedy for live use
    asr_max_chunk_s: float = 20.0  # force a cut when speech runs this long
    asr_min_silence_ms: int = 500  # a pause this long closes a chunk
    asr_backlog_downgrade_s: float = 30.0  # un-transcribed audio beyond this, together with...
    asr_downgrade_rtf: float = 0.7  # ...a recent real-time factor above this, triggers a downgrade
    sample_rate: int = 16000

    # Hallucination filters applied to every segment (D12).
    asr_no_speech_threshold: float = 0.6
    asr_log_prob_threshold: float = -1.0
    asr_compression_ratio_threshold: float = 2.4

    # Global hotkeys (D14), pynput syntax.
    hotkey_flag: str = "<f9>"
    hotkey_pause: str = "<f10>"

    # Confusion flag window: the confusing content usually precedes the tap.
    flag_lookback_s: float = 90.0
    flag_lookahead_s: float = 15.0

    # Post-lecture extraction (D15/D21): window size for the local provider; cloud providers use one window.
    extract_window_s: float = 600.0
    extract_overlap_s: float = 60.0

    # Privacy defaults (D19).
    transcript_retention_days: int = 30

    # Action targets (D1, M5): comma list of "gcal", "notion". Empty = confirm stays local (.ics export).
    targets: str = ""
    google_client_secret: Path = Path("data/google_client_secret.json")
    notion_token: str | None = Field(default=None, validation_alias="NOTION_TOKEN")
    notion_database_id: str | None = Field(default=None, validation_alias="NOTION_DATABASE_ID")
    notion_prop_title: str = "Name"
    notion_prop_date: str = "Due"
    notion_prop_status: str = ""  # optional status/select property
    notion_status_value: str = ""  # value to set on creation, e.g. "To do"
    notion_prop_course: str = ""  # optional select/rich_text property for the course name

    # Model provider (D21). "ollama" is fully local and free; "cloudflare" is the
    # Workers AI free tier (10k neurons/day); "anthropic" needs a paid key.
    llm_provider: str = "ollama"  # ollama | cloudflare | anthropic
    ollama_url: str = "http://localhost:11434"
    # gemma3:4b measured best of the local candidates (docs/LLM_BENCHMARK.md): 100% GPU, ~54 tok/s.
    ollama_model_text: str = "gemma3:4b"
    ollama_model_vision: str = "gemma3:4b"
    ollama_num_ctx: int = 16384  # measured: gemma3:4b stays 100% on the 6 GB GPU at 16K, spills at 32K
    cloudflare_api_token: str | None = Field(default=None, validation_alias="CLOUDFLARE_API_TOKEN")
    cloudflare_account_id: str | None = Field(default=None, validation_alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_model_text: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    cloudflare_model_vision: str = "@cf/meta/llama-3.2-11b-vision-instruct"
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-sonnet-5"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "copilot.sqlite3"


settings = Settings()
