# Change Log

## v0.7.0 - 2026/11/20

### Added

* Support for Python 3.10 - 3.14; the LiteRT runtime (`ai-edge-litert`) is installed on all platforms
* New extras: `train` (training dependencies), `tflite` (tflite export with `litert-torch`), `speex` (Speex noise suppression)
* `openwakeword.train` supports the current [rhasspy/piper-sample-generator](https://github.com/rhasspy/piper-sample-generator),
  including Piper voices (`.onnx`) via the new `piper_sample_generator_model` config key, which allows training non-English wake words
* `convert_torch_to_tflite` exports trained models to tflite without Tensorflow; `convert_onnx_to_tflite` uses `onnx2tf` when available
* Audio helpers in `openwakeword.data`: `load_audio`, `reverberate`, `generate_colored_noise`, `AddBackgroundNoise`
* Tests for the training data pipeline, model export and the false positive metric

### Changed

* Packaging uses `pyproject.toml` only; `speexdsp-ns` is now optional (`pip install openwakeword[speex]`)
* Removed the dependencies on speechbrain, acoustics, torchaudio I/O, onnx-tf and Tensorflow for training
* The DeepPhonemizer model location is configurable (`OPENWAKEWORD_PHONEMIZER_MODEL`, `OPENWAKEWORD_PHONEMIZER_URL`),
  as the original download has been disabled; without it, training continues without phoneme predictions for unknown words
* The streaming melspectrogram only copies the newest audio samples (upstream PR #356)
* Validation data for training is windowed with a strided view instead of a copy, greatly reducing memory use
* CI uses current GitHub Actions and tests Linux, Windows and macOS

### Fixed

* `augmentation_rounds` > 1 had no effect, because the feature arrays were sized for one round
* An interrupted feature computation left a partial `.npy` file that was silently used for training
* Gradients of small batches were discarded before gradient accumulation could use them
* `custom_negative_phrases` was modified in place, leaking training phrases into the test set
* Adversarial phrases were misaligned for words with several pronunciations or without similar words
* `get_false_positives` counted a run of positive frames more than once (upstream PR #353)
* Boolean command line flags of `train.py` defaulted to the string "False" (upstream PRs #351, #308)
* Custom verifier models were applied repeatedly and with the features of the wrong model when several models were loaded
* `mix_clips_batch` crashed with the default `start_index` and when converting to 16-bit PCM
* `apply_reverb` used the first character of a single path; augmented clips could overflow 16-bit PCM
* `detect_from_microphone.py` defaults to ONNX on Windows and downloads the models (upstream PR #237)
* `bulk_predict` failed with the "spawn"/"forkserver" multiprocessing start methods (Windows, macOS, Linux with Python >= 3.14)
* `AudioFeatures.embed_clips` crashed with the tflite framework
* A custom verifier model for any but the first loaded model raised a `ValueError`
* `Model.reset()` did not reset the VAD and Speex noise suppression state, so results depended on previous audio
* `predict_clip` silently produced wrong scores for WAV files that are not 16-bit, 16 khz, mono; it now raises a `ValueError`
* `train_custom_verifier` did not reset the model between clips
* `trim_mmap` mangled relative paths and file names ending in "n", "p" or "y" (`str.strip(".npy")`)
* `convert_clips` passed file names through a shell and ignored the `sr` argument
* `auto_train` assumed an 11.3 hour false positive validation set, and never updated the best false positive rate,
  so the negative class weight was always doubled for the 2nd and 3rd training sequence

## v0.6.0 - 2023/06/15

### Added

* Various bug fixes, and some new functionality in `model.py` to control repeated detections

### Changed

* Models are no longer included in the PyPi package, and must be downloaded separately

### Removed

## v0.5.0 - 2023/06/15

### Added

* A new wakeword model, "hey rhasspy"
* Added support for tflite versions of the melspectrogram model, embedding model, and pre-trained wakeword models
* Added an inference framework argument to allow users to select either ONNX or tflite as the inference framework
* The `detect_from_microphone.py` example now supports additional arguments and has improved console formatting

### Changed

* Made tflite the default inference framework for linux platforms due to improved efficiency, with windows still using ONNX as the default given the lack of pre-built Windows WHLs for the tflite runtime (https://pypi.org/project/tflite/)
* Adjusted the default provider arguments for onnx models to avoid warnings (https://github.com/dscripka/openWakeWord/issues/27)

### Removed