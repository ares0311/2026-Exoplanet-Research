"""Run CNN inference outside processes that have loaded XGBoost.

The macOS wheels used by this project ship incompatible native OpenMP
runtimes: loading XGBoost before PyTorch blocks, while loading PyTorch before
XGBoost terminates the interpreter.  Full-ensemble scoring therefore keeps
XGBoost in the pipeline process and runs CNN inference in a short-lived child.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

RunFn = Callable[..., subprocess.CompletedProcess[str]]


class IsolatedCnnScorer:
    """CNN scorer proxy whose PyTorch runtime lives in a child process."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        run_fn: RunFn = subprocess.run,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._run_fn = run_fn
        self._timeout_seconds = timeout_seconds
        self._training_mission = self._read_training_mission()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> IsolatedCnnScorer:
        """Create a proxy for a saved CNN checkpoint."""
        return cls(Path(path))

    @property
    def checkpoint_path(self) -> Path:
        """Path passed to the isolated inference worker."""
        return self._checkpoint_path

    @property
    def training_mission(self) -> str | None:
        """Mission declared by the checkpoint's sibling config, if present."""
        return self._training_mission

    @property
    def is_available(self) -> bool:
        """Whether the checkpoint and PyTorch module are locally available."""
        return self._checkpoint_path.is_file() and importlib.util.find_spec("torch") is not None

    def _read_training_mission(self) -> str | None:
        config_path = self._checkpoint_path.with_name("config.json")
        try:
            value = json.loads(config_path.read_text(encoding="utf-8")).get(
                "training_mission"
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        return str(value) if value is not None else None

    def predict_proba(self, snippet: list[float]) -> float:
        """Return CNN probability from an isolated Python interpreter."""
        completed = self._run_fn(
            [
                sys.executable,
                "-m",
                "exo_toolkit.ml.cnn_inference_worker",
                str(self._checkpoint_path),
            ],
            input=json.dumps({"snippet": snippet}),
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-500:]
            raise RuntimeError(
                f"isolated CNN inference failed with exit {completed.returncode}: {detail}"
            )
        try:
            probability = float(json.loads(completed.stdout)["probability"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("isolated CNN inference returned invalid JSON") from exc
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("isolated CNN inference returned an invalid probability")
        return probability
