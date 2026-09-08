"""One warm runner for shadow mode, shared by every command that needs the
configured model: `delegate`, `sync`, and the base half of `adapt`.

Starting a runner costs a model load every time; a second delegation a
minute after the first should not pay it again. The first command that
needs the model starts a runner detached, with the runner's own idle
unload (`--ttl`) and without a parent watch, and records where it is in
``~/.xyntetik/shadow/server.json``. The next command asks that port for
its capabilities; only a live runner whose pid and model match is reused,
anything else is replaced. `shadow server --stop` ends it; so does
`uninstall`. The user can always see what is running with `shadow server`.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xyntetik_runner.endpoint import RunnerEndpoint
from xyntetik_runner.process import ManagedRunner, ServerLaunch

STATE_REL = Path(".xyntetik") / "shadow" / "server.json"
DEFAULT_TTL = 900


@dataclass(frozen=True)
class ServerState:
    port: int
    pid: int
    model: str
    adapter: str
    runner: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def read_state(home: Path) -> ServerState | None:
    path = home / STATE_REL
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return ServerState(int(d["port"]), int(d["pid"]), str(d.get("model") or ""),
                           str(d.get("adapter") or ""), str(d.get("runner") or ""))
    except (ValueError, KeyError, TypeError):
        return None


def write_state(home: Path, state: ServerState) -> Path:
    path = home / STATE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.__dict__, indent=2) + "\n", encoding="utf-8")
    return path


def clear_state(home: Path) -> None:
    path = home / STATE_REL
    if path.is_file():
        path.unlink()


def alive(state: ServerState, *, endpoint_factory: Any = None, timeout: float = 3.0) -> bool:
    """True when the recorded runner answers on its port with its own pid
    and the recorded model; a different process on that port is not ours."""
    factory = endpoint_factory or RunnerEndpoint
    try:
        caps = factory(state.base_url).capabilities(timeout=timeout)
    except (OSError, RuntimeError, ValueError):
        return False
    if caps.get("pid") != state.pid:
        return False
    models = [m.get("id") for m in caps.get("models", []) if isinstance(m, dict)]
    return bool(models) and str(models[0]) == Path(state.model).name


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def served_runner(home: Path, cfg: dict[str, object], *, runner: str = "", model: str = "",
                  adapter: str | None = None, ttl: int = DEFAULT_TTL, start_timeout: float = 600.0,
                  request_timeout: float = 900.0, managed_factory: Any = ManagedRunner,
                  endpoint_factory: Any = RunnerEndpoint) -> tuple[RunnerEndpoint, bool]:
    """An endpoint for the configured model: the warm runner when it is
    alive and serving the same model and adapter, else a fresh one, started
    detached and recorded. Returns (endpoint, started_now). Raises
    RuntimeError when nothing can be served."""
    model = model or str(cfg.get("model") or "")
    runner = runner or str(cfg.get("runner") or "runner")
    if adapter is None:
        adapter = str(cfg.get("adapter") or "")
        if adapter and not Path(adapter).is_file():
            adapter = ""
    if not model:
        raise RuntimeError("no model in the shadow config; run 'runner --shadow-mode -m MODEL.gguf' first")
    state = read_state(home)
    if state is not None:
        if (state.model == model and state.adapter == adapter
                and alive(state, endpoint_factory=endpoint_factory)):
            return endpoint_factory(state.base_url, timeout=request_timeout), False
        stop_server(home)
    extra = ("--lora", adapter) if adapter else ()
    launch = ServerLaunch(executable=runner, model=model, port=_free_port(),
                          context_size=int(str(cfg.get("ctx") or 8192)),
                          gpu=str(cfg.get("gpu") or "auto"),
                          threads=int(str(cfg.get("threads") or 0)) or None,
                          parent_pid=None, ttl=ttl, extra_args=extra)
    managed = managed_factory(launch)
    if not managed.start(timeout=start_timeout):
        raise RuntimeError("the runner did not start; check the model path and 'runner --fit'")
    pid = int(getattr(getattr(managed, "process", None), "pid", 0) or 0)
    write_state(home, ServerState(launch.port, pid, model, adapter, runner))
    # the process object is dropped on purpose: the runner outlives this
    # command and unloads by itself after ttl seconds idle
    if hasattr(managed, "process"):
        managed.process = None
    return endpoint_factory(managed.base_url, timeout=request_timeout), True


def stop_server(home: Path) -> bool:
    """End the recorded runner if it is ours, and forget it either way."""
    state = read_state(home)
    clear_state(home)
    if state is None or state.pid <= 0:
        return False
    if not alive(state):
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/pid", str(state.pid), "/f"], capture_output=True)
        else:
            os.kill(state.pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def render_status(home: Path) -> str:
    state = read_state(home)
    if state is None:
        return "no warm runner recorded"
    live = alive(state)
    return (f"runner {state.runner} on {state.base_url} (pid {state.pid}): "
            f"{'alive' if live else 'gone'}; model {Path(state.model).name}"
            f"{' + adapter ' + Path(state.adapter).name if state.adapter else ''}")


__all__ = ["ServerState", "read_state", "write_state", "clear_state", "alive", "served_runner",
           "stop_server", "render_status", "DEFAULT_TTL", "STATE_REL"]
