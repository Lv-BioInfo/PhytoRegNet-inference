"""Load a trained checkpoint and predict the two signal heads."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from .model import SeqUNet1D
from .sequence import one_hot

MODEL_NAMES = ("arabidopsis", "rice_nip")


def load_config(model_name: str) -> dict:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}.")
    config = json.loads(
        files("phytoregnet_minimal")
        .joinpath("configs")
        .joinpath(model_name + ".json")
        .read_text(encoding="utf-8")
    )
    params = config["model_params"]
    tracks = config["tracks"]
    expected = (8192, 4096, 8, 8)
    actual = tuple(
        params[key]
        for key in ("inputlen", "target_region_len", "bin_size", "num_targets")
    )
    if actual != expected or len(tracks) != params["num_targets"]:
        raise ValueError(
            "The model configuration has inconsistent input/output dimensions."
        )
    if len({track["name"] for track in tracks}) != len(tracks):
        raise ValueError("The model configuration has duplicate tissue names.")
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Predictor:
    """Single-checkpoint, float32 forward inference in the supplied orientation.

    `profile` is the nonnegative per-bin training signal. `log_count` is
    the independent count head's log(1 + total signal) prediction.
    """

    def __init__(self, model_name: str, checkpoint: str | Path, device: str = "cpu"):
        self.config = load_config(model_name)
        self.model_name = model_name
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "Use device 'cpu' or 'cuda:0' (or another CUDA device index)."
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable in this environment; use --device cpu."
            )
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        if (
            not isinstance(state, Mapping)
            or not state
            or not all(isinstance(value, torch.Tensor) for value in state.values())
        ):
            raise ValueError("Checkpoint must contain a tensor state_dict.")
        self.model = SeqUNet1D(self.config["model_params"])
        # An architecture mismatch must fail rather than silently skip weights.
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.track_names = [track["name"] for track in self.config["tracks"]]
        self.track_labels = [track["label"] for track in self.config["tracks"]]
        params = self.config["model_params"]
        self.input_length = params["inputlen"]
        self.output_length = params["target_region_len"]
        self.bin_size = params["bin_size"]
        self.output_offset = (self.input_length - self.output_length) // 2

    def predict(
        self, sequences: Sequence[str] | str, batch_size: int = 4
    ) -> dict[str, np.ndarray]:
        if isinstance(sequences, str):
            sequences = [sequences]
        if not sequences:
            raise ValueError("Supply at least one sequence.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        profiles, log_counts = [], []
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                inputs = one_hot(
                    sequences[start : start + batch_size], self.input_length
                )
                profile, log_count = self.model(
                    torch.from_numpy(inputs).to(self.device)
                )
                expected = (
                    len(inputs),
                    len(self.track_names),
                    self.output_length // self.bin_size,
                )
                if (
                    tuple(profile.shape) != expected
                    or tuple(log_count.shape) != expected[:2]
                ):
                    raise ValueError(
                        "The model returned unexpected prediction dimensions."
                    )
                if (
                    not torch.isfinite(profile).all()
                    or not torch.isfinite(log_count).all()
                ):
                    raise ValueError("The model produced nonfinite predictions.")
                profiles.append(profile.cpu().numpy())
                log_counts.append(log_count.cpu().numpy())
        profile = np.concatenate(profiles)
        log_count = np.concatenate(log_counts)
        with np.errstate(over="raise", invalid="raise"):
            count = np.maximum(np.expm1(log_count.astype(np.float64)), 0)
        return {"profile": profile, "log_count": log_count, "count": count}
