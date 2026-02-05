from functools import lru_cache
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "moonshotai/kimi-k2.5"


class Settings(BaseSettings):
    # api key
    nvidia_nim_api_key: str = ""

    # rate limiting
    rate_limit: int = 40
    rate_window: int = 60

    # model params
    temperature: float = 1.0
    max_tokens: int = 81920

    # optimizations - skip unnecessary requests
    skip_quota_check: bool = True
    skip_title_generation: bool = True
    skip_suggestion_mode: bool = True
    skip_filepath_extraction: bool = True
    fast_prefix_detection: bool = True

    @field_validator("nvidia_nim_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v):
        if v == "":
            return ""
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
