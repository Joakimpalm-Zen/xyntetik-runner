"""The runtime chooses only models its own fit probe accepts."""
from pathlib import Path
from typing import Any
import subprocess
from xyntetik_runner.shadow.install import Candidate, fit_verdict, read_config, suggest_model

def test_install_picks_the_largest_gguf_that_fits(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli, install as inst
    main = cli.main
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / "models").mkdir()
    (home / "models" / "small.gguf").write_bytes(b"0" * 10)
    (home / "models" / "big.gguf").write_bytes(b"0" * 20)
    (home / "models" / "huge.gguf").write_bytes(b"0" * 30)
    verdicts = {"huge.gguf": "PAGES", "big.gguf": "FITS WITH --kv q8", "small.gguf": "FITS"}
    monkeypatch.setattr(inst, "fit_verdict", lambda runner, m, ctx: verdicts[m.name])
    best, cands = suggest_model("runner", home)
    assert best is not None and best.path.name == "big.gguf" and [c.path.name for c in cands] == ["huge.gguf", "big.gguf", "small.gguf"]
    monkeypatch.setattr(cli, "suggest_model", lambda runner, home, ctx=8192: suggest_model(runner, home, ctx=ctx))
    assert main(["install", "--home", str(home), "--yes"]) == 0
    text = capsys.readouterr().out
    assert "big.gguf (found on disk" in text and "huge.gguf" in text and "PAGES" in text
    assert read_config(home)["model"] == str((home / "models" / "big.gguf").resolve())
    # nothing fits: installed without a model, said plainly
    verdicts.update({"big.gguf": "PAGES", "small.gguf": "PAGES"})
    home2 = tmp_path / "home2"
    (home2 / ".codex").mkdir(parents=True)
    (home2 / "models").mkdir()
    (home2 / "models" / "small.gguf").write_bytes(b"0" * 10)
    assert main(["install", "--home", str(home2), "--yes"]) == 0
    text = capsys.readouterr().out
    assert "no model for offloading yet: none of the GGUF files found fits" in text
    assert read_config(home2) == {}
    # the verdict parser reads the runner's own line
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "fit: x\n  weights 1 GiB\n  verdict       FITS — 2.37 GiB to spare at ctx 8192\n", ""))
    assert fit_verdict("runner", Path("x.gguf"), 8192) == "FITS"
    assert Candidate(Path("x"), 1, "FITS WITH --kv q8").fits and not Candidate(Path("x"), 1, "PAGES").fits
