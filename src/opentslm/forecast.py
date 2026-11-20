# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Direct multihorizon regression using the Qwen3.5 text decoder."""

import torch
from torch import Tensor, nn
from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5TextModel

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


class QwenForecaster(nn.Module):
    def __init__(self, decoder: Qwen3_5TextModel, *, channels=1, targets=1, horizon=6, patch_size=4):
        super().__init__()
        if min(channels, targets, horizon, patch_size) < 1:
            raise ValueError("Forecast dimensions must be positive")
        self.decoder = decoder
        self.channels = channels
        self.horizon = horizon
        self.targets = targets
        self.patch_size = patch_size
        width = decoder.config.hidden_size
        self.project = nn.Linear(2 * patch_size * channels, width)
        self.readout = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.readout, std=decoder.config.initializer_range)
        self.head = nn.Linear(width, horizon * targets)

    @classmethod
    def from_pretrained(cls, *, local_files_only=False, checkpoint_path=None, **dimensions):
        checkpoint, report = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint_path or MODEL_ID,
            revision=None if checkpoint_path else MODEL_REVISION,
            local_files_only=local_files_only,
            dtype=torch.float32,
            output_loading_info=True,
        )
        if report["missing_keys"] or report["mismatched_keys"] or report["error_msgs"]:
            raise ValueError("Incomplete Qwen checkpoint")
        return cls(checkpoint.model.language_model, **dimensions)

    def encode(self, history: Tensor, observed: Tensor) -> Tensor:
        """Use only historical values and their availability mask, both shaped [B,T,C]."""
        if history.ndim != 3 or history.shape[-1] != self.channels:
            raise ValueError("Expected history with shape [batch, time, channels]")
        if observed.shape != history.shape or observed.dtype != torch.bool:
            raise ValueError("Expected a matching boolean observation mask")
        batch, hours, _ = history.shape
        if hours < self.patch_size or hours % self.patch_size:
            raise ValueError("History length must be a positive multiple of patch size")
        values = history.masked_fill(~observed, 0)
        patches = torch.cat((values, observed.to(values.dtype)), dim=-1)
        tokens = self.project(patches.reshape(batch, hours // self.patch_size, -1))
        tokens = torch.cat((tokens, self.readout.expand(batch, -1, -1)), dim=1)
        state = self.decoder(inputs_embeds=tokens, use_cache=False).last_hidden_state
        return state[:, -1]

    def forward(self, history: Tensor, observed: Tensor) -> Tensor:
        return self.head(self.encode(history, observed)).reshape(len(history), self.horizon, self.targets)


class PersistenceForecaster(QwenForecaster):
    """Univariate forecasts anchored to the observed value at the forecast origin."""

    def __init__(self, decoder: Qwen3_5TextModel, *, horizon=6, patch_size=4):
        super().__init__(decoder, horizon=horizon, patch_size=patch_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history: Tensor, observed: Tensor) -> Tensor:
        correction = super().forward(history, observed)
        if not observed[:, -1].all() or not history[:, -1].isfinite().all():
            raise ValueError("Persistence requires a finite observed value at the origin")
        return history[:, -1:] + correction


QUANTILES = (0.1, 0.5, 0.9)


class QuantileForecaster(QwenForecaster):
    """Median plus two nonnegative distances, one interval per horizon."""

    def __init__(self, decoder):
        super().__init__(decoder, targets=3)

    def forward(self, history, observed):
        left, median, right = super().forward(history, observed).unbind(dim=-1)
        return torch.stack((median - nn.functional.softplus(left), median, median + nn.functional.softplus(right)), dim=-1)


class EpisodeForecaster(QwenForecaster):
    """One raw logit for a threshold crossing anywhere in the six-hour horizon."""

    def __init__(self, decoder):
        super().__init__(decoder)
        self.head = nn.Linear(decoder.config.hidden_size + 1, 1)

    def forward(self, history, observed, normalized_threshold):
        state = self.encode(history, observed)
        features = torch.cat((state, normalized_threshold.reshape(len(history), 1)), dim=-1)
        return self.head(features).squeeze(-1)


FORECASTERS = {
    "direct": QwenForecaster,
    "persistence": PersistenceForecaster,
    "quantile": QuantileForecaster,
    "episode": EpisodeForecaster,
}


def task_loss(head, predicted, target, center, scale):
    if head == "episode":
        return nn.functional.binary_cross_entropy_with_logits(predicted, target)
    normalized = (target - center) / scale
    if head == "quantile":
        residual = normalized - predicted
        levels = predicted.new_tensor(QUANTILES)
        return torch.maximum(levels * residual, (levels - 1) * residual).mean()
    return nn.functional.mse_loss(predicted, normalized)


def smoke_step(model: QwenForecaster, history: Tensor, observed: Tensor, targets: Tensor) -> dict:
    """Run one FP32 AdamW step and check an update in both trainable components."""
    model.train()
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    probes = [model.decoder.layers[0].mlp.down_proj.weight, model.head.weight]
    before = [p.detach().clone() for p in probes]
    predicted = model(history, observed)
    if predicted.shape != targets.shape:
        raise ValueError("Forecast and target shapes must match")
    loss = nn.functional.mse_loss(predicted, targets)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    updates = [(p.detach() - old).abs().max().item() for p, old in zip(probes, before, strict=True)]
    if not all(0 < change < float("inf") for change in updates):
        raise RuntimeError("Expected finite updates to both decoder and head")
    return {
        "prediction_before_step": predicted.detach().cpu().tolist(),
        "mse": loss.item(),
        "gradient_norm_before_clip": grad_norm.item(),
        "learning_rate": learning_rate,
        "decoder_max_update": updates[0],
        "head_max_update": updates[1],
    }
