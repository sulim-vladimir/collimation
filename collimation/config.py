from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            out[key] = _merge(base[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None):
    """config.toml layered over config.example.toml, so a key added to the example has a default
    in every older config."""
    with open(ROOT / "config.example.toml", "rb") as f:
        defaults = tomllib.load(f)
    path = Path(path) if path else ROOT / "config.toml"
    if not path.exists():
        return defaults
    with open(path, "rb") as f:
        return _merge(defaults, tomllib.load(f))
