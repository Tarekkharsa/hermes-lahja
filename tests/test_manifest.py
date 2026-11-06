"""Static checks on plugin.yaml that `hermes plugins validate` does not make."""

import re
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "lahja" / "plugin.yaml"
RESERVED_ROOTS = {"model", "plugins", "security", "settings"}  # hermes_cli/plugins_state.py


def _schema_keys():
    text = MANIFEST.read_text(encoding="utf-8")
    block = text.split("config_schema:", 1)[1]
    return re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", block, re.M)


def test_config_keys_are_not_reserved():
    keys = _schema_keys()
    assert keys, "no config_schema keys found"
    assert not RESERVED_ROOTS & {k.lower() for k in keys}


def test_every_config_key_read_by_code_is_declared():
    code = "".join(p.read_text(encoding="utf-8") for p in MANIFEST.parent.glob("*.py"))
    read = set(re.findall(r'(?:_cfg|get_config)\("([A-Za-z0-9_.-]+)"', code))
    assert read <= set(_schema_keys()), read - set(_schema_keys())
