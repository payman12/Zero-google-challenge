"""Read model settings from config.toml and secrets from the git-ignored .env."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"
ENV_PATH = ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    model_id: str
    temperature: float
    max_tokens: int
    base_url: str
    api_key: str = ""


def load_settings(config_path: Path = CONFIG_PATH, require_key: bool = True) -> Settings:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    load_dotenv(ENV_PATH)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if require_key and not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY is empty: set it in {ENV_PATH}")

    return Settings(
        model_id=config["model"]["id"],
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"],
        base_url=config["openrouter"]["base_url"],
        api_key=api_key,
    )
