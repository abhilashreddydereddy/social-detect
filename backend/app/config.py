from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SOCIAL_DETECT_", extra="ignore")

    app_name: str = "Social Detect API"
    version: str = "0.1.0"
    environment: str = "development"

    # CORS: dashboard dev server + the extension's origin (extensions call
    # from chrome-extension://<id>, which is added dynamically in main.py).
    cors_allow_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    max_upload_mb: int = 50
    max_video_frames: int = 24
    download_timeout_seconds: int = 20

    database_url: str = "postgresql+asyncpg://social_detect:social_detect@localhost:5432/social_detect"
    redis_url: str = "redis://localhost:6379/0"
    use_redis_cache: bool = False

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_probe_checkpoint_path: str | None = None

    device: str = "auto"  # "auto" | "cpu" | "cuda"
    ensemble_enable_face_branch: bool = True
    ensemble_face_max_crops: int = 3
    ensemble_video_frame_samples: int = 12
    image_model_checkpoint_path: str | None = None
    video_model_checkpoint_path: str | None = None
    fusion_model_path: str | None = None
    mfad_net_checkpoint_path: str | None = None
    asvspoof5_audio_checkpoint_path: str | None = None
    # WeDefense deploys a detector as a pair of TorchScript modules: a raw
    # waveform frontend and a backend/projection classifier.  Leave unset to
    # keep this optional detector disabled.
    wedefense_audio_frontend_path: str | None = None
    wedefense_audio_backend_path: str | None = None
    wedefense_audio_spoof_class_index: int = 1
    wedefense_audio_window_seconds: float = 12.0
    wedefense_audio_hop_seconds: float = 6.0


settings = Settings()
