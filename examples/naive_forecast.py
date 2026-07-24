"""Run a synchronous job definition with the anomx runner."""

from anomx import JobRunner
from anomx.data import make_sine_anomaly_dataset


def main() -> None:
    frame = make_sine_anomaly_dataset().to_dataframe().reset_index(drop=True)
    job_definition = {
        "nodes": [
            {"id": "start", "type": "start", "last": [], "next": ["infer"], "config": {}},
            {
                "id": "infer",
                "type": "model_inference",
                "last": ["start"],
                "next": ["score"],
                "config": {"component": "anomx.components.models.constant.ConstantBaselineModel"},
            },
            {
                "id": "score",
                "type": "scorer",
                "last": ["infer"],
                "next": ["detect"],
                "config": {"component": "anomx.components.detection.scorers.absolute_error.AbsoluteErrorScorer"},
            },
            {
                "id": "detect",
                "type": "detector",
                "last": ["score"],
                "next": ["end"],
                "config": {
                    "component": "anomx.components.detection.detectors.threshold.ThresholdDetector",
                    "config": {"source_column": "score", "threshold": 3.0},
                },
            },
            {"id": "end", "type": "end", "last": ["detect"], "next": [], "config": {}},
        ]
    }
    result = JobRunner().run(job_definition, frame)
    print(result.to_dict())
    print([record for record in result.context["decisions"] if record["is_anomaly"]][:5])


if __name__ == "__main__":
    main()
