# Tests for openwakeword.train (require the `train` extra)
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchinfo")
pytest.importorskip("torchmetrics")
pytest.importorskip("yaml")
pytest.importorskip("audiomentations")
pytest.importorskip("torch_audiomentations")
pytest.importorskip("pronouncing")

from openwakeword import train  # noqa: E402

FAKE_GENERATOR = """
calls = []
def generate_samples(**kwargs):
    calls.append(("pt", kwargs))
def generate_samples_onnx(**kwargs):
    calls.append(("onnx", kwargs))
"""


@pytest.fixture
def fake_piper(tmp_path, monkeypatch):
    pkg = tmp_path / "psg" / "piper_sample_generator"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_GENERATOR)
    (tmp_path / "psg" / "models").mkdir()
    (tmp_path / "psg" / "models" / "en_US-libritts_r-medium.pt").write_text("")
    for name in [m for m in sys.modules if m.startswith("piper_sample_generator") or m == "generate_samples"]:
        monkeypatch.delitem(sys.modules, name)
    yield tmp_path / "psg"
    for name in [m for m in sys.modules if m.startswith("piper_sample_generator")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(tmp_path))]


def test_load_sample_generator_default_model(fake_piper):
    generate = train.load_sample_generator({"piper_sample_generator_path": str(fake_piper)})
    generate(text=["hey test"], output_dir="out", batch_size=10, auto_reduce_batch_size=True)
    calls = sys.modules["piper_sample_generator.__main__"].calls
    kind, kwargs = calls[-1]
    assert kind == "pt"
    assert kwargs["model"].endswith("en_US-libritts_r-medium.pt")
    assert kwargs["batch_size"] == 10


def test_load_sample_generator_onnx_voice(fake_piper):
    generate = train.load_sample_generator({"piper_sample_generator_path": str(fake_piper),
                                            "piper_sample_generator_model": "de_DE-thorsten-medium.onnx"})
    generate(text=["hey test"], output_dir="out", batch_size=10, auto_reduce_batch_size=True)
    kind, kwargs = sys.modules["piper_sample_generator.__main__"].calls[-1]
    assert kind == "onnx"
    assert kwargs["model"] == "de_DE-thorsten-medium.onnx"
    assert "batch_size" not in kwargs and "auto_reduce_batch_size" not in kwargs


@pytest.mark.parametrize("model_type", ["dnn", "rnn"])
def test_export_and_convert(tmp_path, model_type):
    model = train.Model(n_classes=1, input_shape=(16, 96), model_type=model_type, layer_dim=32)
    model.model.eval()
    model.export_model(model.model, "test_model", str(tmp_path))
    onnx_path = tmp_path / "test_model.onnx"
    assert onnx_path.exists()

    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 16, 96).astype(np.float32)
    with torch.no_grad():
        expected = model.model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(session.run(None, {session.get_inputs()[0].name: x})[0], expected, atol=1e-5)

    pytest.importorskip("litert_torch")
    tflite_path = tmp_path / "test_model.tflite"
    train.convert_torch_to_tflite(model.model, (16, 96), str(tflite_path))
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()
    np.testing.assert_allclose(interpreter.get_tensor(interpreter.get_output_details()[0]["index"]), expected, atol=1e-5)
