# Tests for openwakeword.utils
import os

import numpy as np
import pytest

import openwakeword
from openwakeword.utils import AudioFeatures, bulk_predict

openwakeword.utils.download_models()

TEST_CLIPS = [os.path.join("tests", "data", i) for i in ["alexa_test.wav", "hey_mycroft_test.wav", "hey_jane.wav"]]


@pytest.mark.parametrize("inference_framework", ["onnx", "tflite"])
def test_embed_clips(inference_framework):
    features = AudioFeatures(inference_framework=inference_framework)
    x = (np.random.randn(3, 32000)*1000).astype(np.int16)
    embeddings = features.embed_clips(x, batch_size=2, ncpu=2)
    assert embeddings.shape == (3, 16, 96)
    assert features.get_embedding_shape(2.0) == (16, 96)


def test_embed_clips_frameworks_agree():
    x = (np.random.randn(2, 32000)*1000).astype(np.int16)
    onnx = AudioFeatures(inference_framework="onnx").embed_clips(x)
    tflite = AudioFeatures(inference_framework="tflite").embed_clips(x)
    np.testing.assert_allclose(onnx, tflite, atol=0.05)


@pytest.mark.parametrize("ncpu", [1, 2])
def test_bulk_predict(ncpu):
    results = bulk_predict(TEST_CLIPS, wakeword_models=["alexa", "hey_mycroft"], ncpu=ncpu, inference_framework="onnx")
    assert set(results.keys()) == set(TEST_CLIPS)
    scores = {os.path.basename(clip): {k: max(i[k] for i in predictions) for k in predictions[0]}
              for clip, predictions in results.items()}
    assert scores["alexa_test.wav"]["alexa"] >= 0.5
    assert scores["hey_mycroft_test.wav"]["hey_mycroft"] >= 0.5
    assert scores["hey_jane.wav"]["alexa"] < 0.5 and scores["hey_jane.wav"]["hey_mycroft"] < 0.5
