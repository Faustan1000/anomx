"""Train, export, and run a tiny model: pip install 'anomx[ml]'.

Runs as a standalone script; inside an Anomx compute block use work.metric and
work.publish_model to persist the same model and KPIs in the platform.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from anomx import Dataset, ONNXModel, PyTorchModel
from anomx.detectors import QuantileThresholdDetector
from anomx.scorers import AbsoluteErrorScorer


def main() -> None:
    series = Dataset.from_values(np.sin(np.arange(160, dtype=np.float32) / 8))
    train, validation = series.split_after(0.8)
    model = PyTorchModel(input_chunk_length=12, n_epochs=5, random_state=42)
    model.fit(train)
    forecast = model.predict(len(validation))
    scores = AbsoluteErrorScorer().score_from_prediction(validation, forecast)
    # Fit a threshold on a calibration segment, then detect on later observations.
    calibration, evaluation = scores.split_after(0.5)
    labels = QuantileThresholdDetector(0.95).fit(calibration).detect(evaluation)
    with TemporaryDirectory(prefix="anomx-training-") as folder:
        artifact = model.export_onnx(Path(folder) / "model.onnx")
        output = ONNXModel(artifact).predict(train.values()[-12:][None])
        print({"loss": model.training_history[-1]["loss"], "forecast": output.tolist()})
        print({"detected_samples": int(labels.values().sum())})


if __name__ == "__main__":
    main()
