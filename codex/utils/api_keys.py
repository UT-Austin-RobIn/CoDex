"""API keys: environment variables first, then keys saved from the app.

Saved keys live in <repo>/.api_keys.json (override with CODEX_KEYS_FILE):
readable only by the owner and gitignored, so they are never committed.
"""

import json
import os
from pathlib import Path

KEYS = {"OPENAI_API_KEY": "OpenAI", "TRIPO_API_KEY": "Tripo"}
DEFAULT_FILE = Path(__file__).resolve().parents[2] / ".api_keys.json"


def keys_file():
    return Path(os.environ.get("CODEX_KEYS_FILE", DEFAULT_FILE))


def saved_keys():
    try:
        with open(keys_file()) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k in KEYS and v}
    except (OSError, ValueError):
        return {}


def get_key(name):
    """The environment variable if set, otherwise the saved key (or None)."""
    return os.environ.get(name) or saved_keys().get(name)


def load_saved_keys():
    """Export saved keys for variables the environment does not already set."""
    for name, value in saved_keys().items():
        os.environ.setdefault(name, value)


def save_key(name, value):
    """Save (or with an empty value, remove) a key and apply it to this process."""
    if name not in KEYS:
        raise KeyError(name)
    data = saved_keys()
    if value:
        data[name] = value.strip()
        os.environ[name] = value.strip()
    else:
        data.pop(name, None)
    path = keys_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(path, 0o600)


def masked(value):
    """sk-proj-abc...wxyz -> sk-************wxyz"""
    if not value:
        return ""
    return value[:3] + "*" * 12 + value[-4:] if len(value) > 10 else "*" * len(value)


def key_source(name):
    if os.environ.get(name) and os.environ.get(name) != saved_keys().get(name):
        return "environment"
    return "saved" if saved_keys().get(name) else None
