import base64
import hashlib

import pytest

from anomx import ONNXModel, WorkContext


@pytest.fixture
def artifact():
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tensor = onnx.TensorProto.FLOAT
    value = onnx.helper.make_tensor_value_info
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["left", "right"], ["sum"]),
            onnx.helper.make_node("Sub", ["left", "right"], ["difference"]),
        ],
        "two-input arithmetic",
        [value("left", tensor, [None, 2]), value("right", tensor, [None, 2])],
        [value("sum", tensor, [None, 2]), value("difference", tensor, [None, 2])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def test_stored_model_runs_named_inputs_and_outputs(artifact):
    calls = []

    def load(reference):
        calls.append(reference)
        return {
            "artifact_base64": base64.b64encode(artifact).decode(),
            "checksum_sha256": hashlib.sha256(artifact).hexdigest(),
        }

    model = WorkContext({"model_artifact": load}).load_model("models_model-example")
    assert calls == ["models_model-example"]
    assert model.run({"left": [[3, 4], [5, 6]], "right": [[1, 2], [3, 4]]}) == {
        "sum": [[4.0, 6.0], [8.0, 10.0]],
        "difference": [[2.0, 2.0], [2.0, 2.0]],
    }


def test_named_inference_validates_tensors(artifact):
    model = ONNXModel(artifact)
    with pytest.raises(ValueError, match="named input"):
        model.run({"left": [[1, 2]]})
    with pytest.raises(ValueError, match="finite"):
        model.run({"left": [[float("nan"), 2]], "right": [[1, 2]]})


def test_loading_requires_a_host_and_verified_artifact():
    with pytest.raises(RuntimeError, match="model_artifact"):
        WorkContext().load_model("models_model-example")
    with pytest.raises(ValueError, match="invalid model artifact"):
        WorkContext({"model_artifact": lambda **_: {}}).load_model("models_model-example")
    response = {"artifact_base64": base64.b64encode(b"tampered").decode(), "checksum_sha256": "bad"}
    with pytest.raises(ValueError, match="checksum"):
        WorkContext({"model_artifact": lambda **_: response}).load_model("models_model-example")
