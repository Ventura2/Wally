from __future__ import annotations

import torch
from pydantic import BaseModel, field_validator

from wally.planner.protocols import WorldModelProtocol


class SubgoalDetectorConfig(BaseModel):
    threshold: float = 1.0
    smoothing_window: int = 5
    min_segment_length: int = 8

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("threshold must be greater than 0")
        return v

    @field_validator("smoothing_window")
    @classmethod
    def _check_smoothing_window(cls, v: int) -> int:
        if v < 1:
            raise ValueError("smoothing_window must be at least 1")
        return v

    @field_validator("min_segment_length")
    @classmethod
    def _check_min_segment_length(cls, v: int) -> int:
        if v < 2:
            raise ValueError("min_segment_length must be at least 2")
        return v

    @classmethod
    def default(cls) -> SubgoalDetectorConfig:
        return cls()


class SubgoalDetector:
    # consumes prediction error, not the raw predicted tensor; unaffected by
    # residual-loss contract change.
    def __init__(self, config: SubgoalDetectorConfig | None = None) -> None:
        self._config = config if config is not None else SubgoalDetectorConfig.default()

    @property
    def config(self) -> SubgoalDetectorConfig:
        return self._config

    def compute_prediction_errors(
        self,
        model: WorldModelProtocol,
        frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if frames.dim() == 5:
            return self._compute_errors_batched(model, frames, actions)
        return self._compute_errors_single(model, frames, actions)

    def _compute_errors_single(
        self,
        model: WorldModelProtocol,
        frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        T = frames.shape[0]
        with torch.no_grad():
            encoded = model.encode(frames)
        errors = []
        for t in range(T - 1):
            z_t = encoded[t : t + 1]
            a_t = actions[t : t + 1]
            z_pred = model.predict(z_t, a_t)
            z_actual = encoded[t + 1 : t + 2]
            err = torch.norm(z_pred - z_actual, p=2, dim=-1).squeeze(0)
            errors.append(err)
        return torch.stack(errors)

    def _compute_errors_batched(
        self,
        model: WorldModelProtocol,
        frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        B, T = frames.shape[0], frames.shape[1]
        with torch.no_grad():
            flat_frames = frames.reshape(B * T, *frames.shape[2:])
            encoded_flat = model.encode(flat_frames)
            encoded = encoded_flat.reshape(B, T, -1)
        errors = []
        for t in range(T - 1):
            z_t = encoded[:, t]
            a_t = actions[:, t]
            z_pred = model.predict(z_t, a_t)
            z_actual = encoded[:, t + 1]
            err = torch.norm(z_pred - z_actual, p=2, dim=-1)
            errors.append(err)
        return torch.stack(errors, dim=1)

    def smooth_errors(self, errors: torch.Tensor) -> torch.Tensor:
        w = self._config.smoothing_window
        if w <= 1:
            return errors.clone()
        kernel = torch.ones(w, dtype=errors.dtype, device=errors.device) / w
        if errors.dim() == 1:
            padded = errors.unsqueeze(0).unsqueeze(0)
            padded = torch.nn.functional.pad(padded, (w // 2, w // 2), mode="replicate")
            smoothed = torch.conv1d(
                padded,
                kernel.unsqueeze(0).unsqueeze(0),
            ).squeeze(0).squeeze(0)
            return smoothed[: errors.shape[0]]
        B = errors.shape[0]
        smoothed_list = []
        for b in range(B):
            row = errors[b]
            padded = row.unsqueeze(0).unsqueeze(0)
            padded = torch.nn.functional.pad(padded, (w // 2, w // 2), mode="replicate")
            smoothed = torch.conv1d(
                padded,
                kernel.unsqueeze(0).unsqueeze(0),
            ).squeeze(0).squeeze(0)
            smoothed_list.append(smoothed[: row.shape[0]])
        return torch.stack(smoothed_list)

    def detect_change_points(self, smoothed_errors: torch.Tensor) -> list[int]:
        if smoothed_errors.dim() == 2:
            all_cps: list[int] = []
            for b in range(smoothed_errors.shape[0]):
                all_cps.extend(self._detect_1d(smoothed_errors[b]).tolist())
            return sorted(set(all_cps))
        return self._detect_1d(smoothed_errors).tolist()

    def _detect_1d(self, errors: torch.Tensor) -> torch.Tensor:
        min_seg = self._config.min_segment_length
        threshold = self._config.threshold
        T = errors.shape[0]

        candidates: list[int] = []
        for i in range(1, T - 1):
            is_local_max = errors[i] > errors[i - 1] and errors[i] > errors[i + 1]
            if is_local_max and errors[i] > threshold:
                candidates.append(i)

        if not candidates:
            return torch.tensor([], dtype=torch.long)

        filtered = list(candidates)
        changed = True
        while changed:
            changed = False
            new_filtered: list[int] = []
            skip: set[int] = set()
            for idx in range(len(filtered)):
                if idx in skip:
                    continue
                too_close = (
                    idx + 1 < len(filtered)
                    and filtered[idx + 1] - filtered[idx] < min_seg
                )
                if too_close:
                    i1, i2 = filtered[idx], filtered[idx + 1]
                    if errors[i1] >= errors[i2]:
                        new_filtered.append(i1)
                        skip.add(idx + 1)
                    else:
                        new_filtered.append(i2)
                        skip.add(idx)
                    changed = True
                else:
                    new_filtered.append(filtered[idx])
            filtered = new_filtered

        return torch.tensor(sorted(filtered), dtype=torch.long)

    def extract_abstract_transitions(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        change_points: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        T = latents.shape[0]
        frame_boundaries = [cp + 1 for cp in sorted(change_points)]
        boundaries = [0] + frame_boundaries + [T]
        transitions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end - start < 2:
                continue
            start_latent = latents[start]
            end_latent = latents[end - 1]
            action_start = max(0, start - 1) if i > 0 else 0
            seg_actions = actions[action_start : end - 1]
            macro_action = seg_actions.float().mean(dim=0)
            transitions.append((start_latent, end_latent, macro_action))
        return transitions
