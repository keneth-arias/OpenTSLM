# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Six-hour PM2.5 regression on the pinned per-series Flamingo checkpoint."""

from contextlib import contextmanager
import hashlib
import json

import torch
from torch import nn

WEIGHTS_MANIFEST_SHA256 = "7f8562992f1cf3852ced581f331247c6420fc90292d45453424c77f0a29c199f"
MODEL_ID = "opentime/tslm-nov26"
MODEL_REVISION = "5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983"


def forecast_schema():
    from opentslm.model.llm.per_series import FORECAST_PROMPT

    return {
        "schema": "flamingo-pm25-direct-v1",
        "model": MODEL_ID,
        "history_hours": 24,
        "horizon_hours": 6,
        "readout": "last_nonpadding_token",
        "prompt_sha256": hashlib.sha256(FORECAST_PROMPT.encode()).hexdigest(),
        "transform": "history_mean_population_std_flat_one_v1",
        "lora_targets": ["self_attn.q_proj", "self_attn.v_proj"],
        "lora_rank": 4,
        "lora_alpha": 8,
    }


class FlamingoForecaster(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.head = nn.Linear(backbone.hidden_size, 6)

    @property
    def decoder(self):
        return self.backbone.flamingo.lang_encoder.language_model

    @classmethod
    def from_pretrained(cls, *, checkpoint_path, local_files_only):
        from pathlib import Path
        from opentslm.model.llm.per_series import PerSeriesFlamingo

        if not local_files_only or not Path(checkpoint_path).is_dir():
            raise ValueError("A staged local Flamingo snapshot is required")
        return cls(PerSeriesFlamingo.from_pretrained(checkpoint_path, device="cpu", dtype=torch.float32))

    def add_lora(self):
        from peft import LoraConfig, inject_adapter_in_model

        targets = [
            f"layers.{i}.decoder_layer.self_attn.{name}"
            for i in range(len(self.decoder.layers))
            for name in ("q_proj", "v_proj")
        ]
        if not targets or any(not isinstance(self.decoder.get_submodule(name), nn.Linear) for name in targets):
            raise ValueError("Unexpected Gemma self-attention layout")
        config = LoraConfig(r=4, lora_alpha=8, target_modules=targets)
        inject_adapter_in_model(config, self.decoder)
        expected = {f"{target}.lora_{side}.default.weight" for target in targets for side in ("A", "B")}
        actual = {name for name, param in self.decoder.named_parameters() if param.requires_grad}
        if actual != expected:
            raise ValueError("Unexpected Flamingo LoRA trainable parameters")
        return sorted(name for name in actual if ".lora_B." in name)

    def frozen_digest(self):
        """Digest named frozen weights, including aliases, with bounded host memory."""
        digest = hashlib.sha256()
        for name, param in self.named_parameters(remove_duplicate=False):
            if param.requires_grad:
                continue
            header = [name, list(param.shape), str(param.dtype)]
            digest.update(json.dumps(header, separators=(",", ":")).encode() + b"\n")
            for part in param.detach().view(-1).split(2**22):
                digest.update(memoryview(part.cpu().numpy()).cast("B"))
        return digest.hexdigest()

    def adapter_state(self, state):
        # DCP can retain frozen alias keys for tied Flamingo modules. Save only the
        # unique trainable tensors, and fail if DCP lost one of the required adapters.
        return {name: state[name] for name, param in self.named_parameters() if param.requires_grad}

    @contextmanager
    def generation_mode(self):
        """Use the immutable language/QA weights for generation and restore adapter state on exit."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        enabled = [m for m in self.decoder.modules() if isinstance(m, BaseTunerLayer) and not m.disable_adapters]
        training = self.training
        try:
            self.eval()
            for module in enabled:
                module.enable_adapters(False)
            with torch.inference_mode():
                yield self.backbone
        finally:
            for module in enabled:
                module.enable_adapters(True)
            self.train(training)

    def forward(self, history, observed):
        if history.ndim != 3 or history.shape[1:] != (24, 1):
            raise ValueError("Flamingo forecasting expects [batch, 24, 1] raw histories")
        if observed.shape != history.shape or observed.dtype != torch.bool:
            raise ValueError("Expected a matching boolean observation mask")
        features = self.backbone.forecast_features(history.squeeze(-1), observed.squeeze(-1))
        return self.head(features).unsqueeze(-1)


def verify_base(weights):
    from .forecast_inputs import verify_weight_snapshot

    verify_weight_snapshot(weights, WEIGHTS_MANIFEST_SHA256, strict=True)


def restore_selected(weights_dir, checkpoint_dir, receipt_path, *, expected_receipt_sha256, device="cpu"):
    """Restore only the selected adapter from a fingerprinted terminal direct-fit receipt."""
    from pathlib import Path
    from .forecast_checkpoint import files, read_manifest, sha256
    from .forecast_inference import checkpoint_head, load_checkpoint_model

    weights, checkpoint, receipt = map(Path, (weights_dir, checkpoint_dir, receipt_path))
    raw_receipt = receipt.read_bytes()
    if receipt.is_symlink() or hashlib.sha256(raw_receipt).hexdigest() != expected_receipt_sha256:
        raise ValueError("Selected Flamingo receipt checksum mismatch")
    result = json.loads(raw_receipt)
    manifest = read_manifest(checkpoint)
    if (
        not manifest
        or result.get("outcome") not in ("early-stopped", "completed")
        or result.get("identity") != manifest["identity"]
        or result.get("best_step") != manifest["step"]
        or result.get("selected_manifest_sha256") != sha256(checkpoint / "manifest.json")
        or manifest["files"] != files(checkpoint)
    ):
        raise ValueError("Expected the complete selected generation from the Flamingo receipt")
    identity = manifest["identity"]
    if identity.get("backbone") != "flamingo":
        raise ValueError("Expected a Flamingo forecast checkpoint")
    checkpoint_head(identity, None, "direct", None)
    verify_base(weights)
    model = FlamingoForecaster.from_pretrained(checkpoint_path=weights, local_files_only=True)
    model.add_lora()
    load_checkpoint_model(model, checkpoint, True)
    if model.frozen_digest() != identity.get("frozen_parameters_sha256"):
        raise ValueError("Restored frozen parameters differ from the training base")
    return model.to(device).eval()
