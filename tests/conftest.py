"""Load the plugin the way hermes_cli.plugins does (as ``hermes_plugins.lahja``) on top of a
minimal stand-in for ``agent.transcription_provider`` so the suite runs without Hermes."""

from __future__ import annotations

import abc
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "lahja"


class _StubTranscriptionProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def display_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return True

    def list_models(self):
        return []

    def default_model(self):
        ms = self.list_models()
        return ms[0]["id"] if ms else None

    def get_setup_schema(self):
        return {}

    @abc.abstractmethod
    def transcribe(self, file_path, *, model=None, language=None, **extra): ...


def _install_stubs() -> None:
    if "agent.transcription_provider" in sys.modules:
        return
    agent = types.ModuleType("agent")
    agent.__path__ = []
    tp = types.ModuleType("agent.transcription_provider")
    tp.TranscriptionProvider = _StubTranscriptionProvider
    sys.modules["agent"] = agent
    sys.modules["agent.transcription_provider"] = tp


def _load_plugin():
    _install_stubs()
    if "hermes_plugins.lahja" in sys.modules:
        return sys.modules["hermes_plugins.lahja"]
    pkg = types.ModuleType("hermes_plugins")
    pkg.__path__ = []
    sys.modules.setdefault("hermes_plugins", pkg)
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.lahja", PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.lahja"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeState:
    def __init__(self, data_dir: Path):
        self._d: dict = {}
        self.data_dir = data_dir

    def get(self, key, default=None):
        return self._d.get(key, default)

    def set(self, key, value):
        self._d[key] = value


class FakeCtx:
    def __init__(self, data_dir: Path, config: dict | None = None):
        self.state = FakeState(data_dir)
        self.config = config or {}
        self.registered: dict = {}

    # Mirrors hermes_cli.plugins_state: these roots raise ValueError in real Hermes.
    RESERVED_ROOTS = frozenset({"model", "plugins", "security", "settings"})

    def get_config(self, key, default=None):
        if key.split(".")[0].lower() in self.RESERVED_ROOTS:
            raise ValueError(f"reserved plugin config key: {key}")
        return self.config.get(key, default)

    def register_transcription_provider(self, p):
        self.registered["provider"] = p

    def register_command(self, name, handler, **kw):
        self.registered[f"/{name}"] = handler

    def register_cli_command(self, name, **kw):
        self.registered[f"cli:{name}"] = kw


@pytest.fixture(scope="session")
def lahja():
    return _load_plugin()


@pytest.fixture
def ctx(tmp_path):
    return FakeCtx(tmp_path)
