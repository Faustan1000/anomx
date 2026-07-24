import pandas as pd
import pytest

from anomx import (
    ConstantBaselineModel,
    DataStructure,
    JobDefinition,
    JobRunner,
    ModelSignature,
    NormalityModel,
    Predictive,
    discover_component_payloads,
)
from anomx.runner import JobDefinitionError


def build_job_definition_payload():
    return {
        "nodes": [
            {"id": "start", "type": "start", "last": [], "next": ["infer"], "config": {}, "_meta": {"x": 0, "y": 0}},
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
                    "config": {"source_column": "score", "threshold": 2.0},
                },
            },
            {"id": "end", "type": "end", "last": ["detect"], "next": [], "config": {}},
        ]
    }


def test_component_payloads_expose_capabilities_and_signature():
    payloads = {payload["key"]: payload for payload in discover_component_payloads()}

    constant_payload = payloads["constant_baseline"]
    assert "predictive" in constant_payload["capabilities"]
    assert "tabular" in constant_payload["signature"]["structures"]
    assert constant_payload["import_path"].endswith("ConstantBaselineModel")
    assert any(parameter["name"] == "feature_columns" for parameter in constant_payload["parameters"])

    assert payloads["zscore"]["component_type"] == "scorer"
    assert payloads["threshold"]["component_type"] == "detector"


def test_capability_mixins_and_signature_are_inspectable():
    assert issubclass(ConstantBaselineModel, NormalityModel)
    assert issubclass(ConstantBaselineModel, Predictive)
    signature = ConstantBaselineModel.get_component_signature()
    assert isinstance(signature, ModelSignature)
    assert signature.supports_structure(DataStructure.TABULAR)

    model = ConstantBaselineModel({"feature_columns": ["value"]})
    assert model.get_config()["feature_columns"] == ["value"]


def test_job_definition_round_trip_preserves_meta():
    definition = JobDefinition.from_dict(build_job_definition_payload())
    assert definition.validate() == []
    payload = definition.to_dict()
    assert payload["nodes"][0]["_meta"] == {"x": 0, "y": 0}


def test_job_definition_validation_reports_problems():
    definition = JobDefinition.from_dict({
        "nodes": [
            {"id": "start", "type": "start", "last": [], "next": ["missing"], "config": {}},
        ]
    })
    problems = definition.validate()
    assert any("unknown node" in problem for problem in problems)
    assert any("end node" in problem for problem in problems)

    with pytest.raises(JobDefinitionError):
        definition.node("nope")


def test_job_runner_executes_pipeline_synchronously():
    frame = pd.DataFrame({"value": [1.0] * 20 + [9.0] + [1.0] * 4})
    result = JobRunner().run(build_job_definition_payload(), frame)

    assert result.status == "completed"
    assert result.visited_node_ids[0] == "start"
    assert result.visited_node_ids[-1] == "end"
    decisions = result.context["decisions"]
    assert isinstance(decisions, list)
    assert any(record["is_anomaly"] for record in decisions)


def test_job_runner_context_stays_json_encodable():
    import json

    frame = pd.DataFrame({"value": [1.0] * 10 + [5.0]})
    result = JobRunner().run(build_job_definition_payload(), frame)
    assert json.dumps(result.context)


def test_job_runner_runs_single_nodes_with_propagated_context():
    from anomx.runner import NodeExecutor, frame_to_records

    frame = pd.DataFrame({"value": [1.0] * 20 + [9.0]})
    definition = build_job_definition_payload()
    executor = NodeExecutor()
    context = {"data": frame_to_records(frame)}
    from anomx.runner import JobDefinition as Definition

    job_definition = Definition.from_dict(definition)
    node_id = "start"
    visited = []
    while True:
        execution = executor.execute(job_definition, node_id, context)
        visited.append(execution.node_id)
        context = execution.context
        if not execution.next_node_ids:
            break
        node_id = execution.next_node_ids[0]

    assert visited == ["start", "infer", "score", "detect", "end"]
    assert any(record["is_anomaly"] for record in context["decisions"])


def test_job_runner_if_else_and_python_nodes():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    definition = {
        "nodes": [
            {"id": "start", "type": "start", "last": [], "next": ["branch"], "config": {}},
            {
                "id": "branch",
                "type": "if_else",
                "last": ["start"],
                "next": ["train", "note"],
                "config": {"condition_key": "should_train", "branches": {"true": "train", "false": "note"}},
            },
            {
                "id": "train",
                "type": "model_training",
                "last": ["branch"],
                "next": ["end"],
                "config": {"component": "anomx.components.models.constant.ConstantBaselineModel"},
            },
            {
                "id": "note",
                "type": "python",
                "last": ["branch"],
                "next": ["end"],
                "config": {"body": "context['note'] = 'skipped'"},
            },
            {"id": "end", "type": "end", "last": ["train", "note"], "next": [], "config": {}},
        ]
    }
    runner = JobRunner()

    trained = runner.run(definition, frame, context={"flags": {"should_train": True}})
    assert "train" in trained.visited_node_ids
    assert trained.context["model_ref"].endswith("ConstantBaselineModel")
    assert trained.context["flags"]["trained"] is True

    skipped = runner.run(definition, frame, context={"flags": {"should_train": False}})
    assert "note" in skipped.visited_node_ids
    assert skipped.context["note"] == "skipped"


def test_python_body_receives_and_returns_context():
    from anomx.runner import run_python_body

    context = run_python_body("context['doubled'] = context['value'] * 2\nreturn context", {"value": 21})
    assert context["doubled"] == 42
