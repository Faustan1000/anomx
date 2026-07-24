"""PyTorch Lightning autoencoder model (requires the `torch` extra)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.components.base import (
    ComponentCapabilities,
    DataStructure,
    ModelSignature,
    NormalityModel,
    Reconstructive,
)
from anomx.data.base.characteristics import Modality


class TorchAutoencoderModel(NormalityModel, Reconstructive):
    """Small fully-connected autoencoder trained with PyTorch Lightning.

    Reconstruction error per observation lands in `model_score`. The component
    exists to establish the torch/lightning integration pattern; the network
    is intentionally tiny.
    """

    component_key = "torch_autoencoder"
    component_icon = "CpuChip01"
    signature = ModelSignature(
        structures={DataStructure.TABULAR, DataStructure.TEMPORAL_SEQUENCE},
        modalities={Modality.NUMERIC},
    )
    component_name = "Torch Autoencoder"
    component_default_config = {
        "feature_columns": [],
        "hidden_size": 8,
        "latent_size": 2,
        "max_epochs": 5,
    }
    component_config_schema = {
        "feature_columns": {"type": "array"},
        "hidden_size": {"type": "integer"},
        "latent_size": {"type": "integer"},
        "max_epochs": {"type": "integer"},
    }
    capabilities: ClassVar[ComponentCapabilities] = ComponentCapabilities(
        supports_sequences=True,
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.hidden_size = max(1, int(self.config.get("hidden_size", 8)))
        self.latent_size = max(1, int(self.config.get("latent_size", 2)))
        self.max_epochs = max(1, int(self.config.get("max_epochs", 5)))
        self.feature_columns: list[str] = []
        self.state_dict_payload: dict[str, Any] | None = None
        self._module: Any | None = None

    @staticmethod
    def _require_torch() -> tuple[Any, Any]:
        try:
            import lightning
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("The Torch Autoencoder model requires `pip install anomx[torch]`.") from exc
        return torch, lightning

    def _build_module(self, feature_count: int) -> Any:
        torch, lightning = self._require_torch()

        class AutoencoderModule(lightning.LightningModule):
            def __init__(self, input_size: int, hidden_size: int, latent_size: int) -> None:
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, latent_size),
                )
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(latent_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, input_size),
                )

            def forward(self, batch: Any) -> Any:
                return self.decoder(self.encoder(batch))

            def training_step(self, batch: Any, batch_index: int) -> Any:
                (values,) = batch
                reconstruction = self(values)
                return torch.nn.functional.mse_loss(reconstruction, values)

            def configure_optimizers(self) -> Any:
                return torch.optim.Adam(self.parameters(), lr=1e-2)

        return AutoencoderModule(feature_count, self.hidden_size, self.latent_size)

    def fit(self, data: Any) -> None:
        torch, lightning = self._require_torch()
        frame = ensure_dataframe(data)
        configured_columns = self.config.get("feature_columns")
        self.feature_columns = (
            [str(column) for column in configured_columns]
            if configured_columns
            else frame.select_dtypes(include=["number"]).columns.tolist()
        )
        values = torch.tensor(frame[self.feature_columns].astype(float).to_numpy(), dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(values)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(64, max(1, len(dataset))), shuffle=True)
        self._module = self._build_module(len(self.feature_columns))
        trainer = lightning.Trainer(
            accelerator="cpu",
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(self._module, loader)
        self.state_dict_payload = {key: value.tolist() for key, value in self._module.state_dict().items()}

    def predict(self, data: Any) -> pd.DataFrame:
        torch, _ = self._require_torch()
        if self._module is None:
            if self.state_dict_payload is None or not self.feature_columns:
                raise RuntimeError("Model must be fit or loaded before predict() is called.")
            self._module = self._build_module(len(self.feature_columns))
            self._module.load_state_dict({key: torch.tensor(value) for key, value in self.state_dict_payload.items()})

        frame = ensure_dataframe(data)
        values = torch.tensor(frame[self.feature_columns].astype(float).to_numpy(), dtype=torch.float32)
        self._module.eval()
        with torch.no_grad():
            reconstruction = self._module(values)
            errors = torch.mean(torch.abs(reconstruction - values), dim=1)

        result = frame.copy()
        result["model_score"] = [round(float(error), 6) for error in errors.tolist()]
        return result

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "config": self.config,
                    "feature_columns": self.feature_columns,
                    "hidden_size": self.hidden_size,
                    "latent_size": self.latent_size,
                    "max_epochs": self.max_epochs,
                    "state_dict": self.state_dict_payload,
                },
                handle,
            )

    def load(self, path: str) -> TorchAutoencoderModel:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.config = dict(payload["config"])
        self.feature_columns = list(payload["feature_columns"])
        self.hidden_size = int(payload["hidden_size"])
        self.latent_size = int(payload["latent_size"])
        self.max_epochs = int(payload["max_epochs"])
        self.state_dict_payload = payload["state_dict"]
        self._module = None
        return self
