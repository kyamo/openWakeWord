# Tests for the training data utilities in openwakeword.data (require the `train` extra)
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
pytest.importorskip("audiomentations")
pytest.importorskip("torch_audiomentations")
pytest.importorskip("pronouncing")

from openwakeword import data  # noqa: E402


def _write_wav(path, x, sr=16000):
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype="PCM_16")
    return str(path)


def test_load_audio_shapes(tmp_path):
    mono = _write_wav(tmp_path / "mono.wav", np.zeros(1600))
    stereo = _write_wav(tmp_path / "stereo.wav", np.zeros((1600, 2)))
    x, sr = data.load_audio(mono)
    assert sr == 16000 and x.shape == (1, 1600) and x.dtype == torch.float32
    x, sr = data.load_audio(stereo)
    assert x.shape == (2, 1600)
    assert data.read_audio(mono).shape == (1600,)
    assert data.load_rir(stereo).shape == (1600,)
    assert data.get_clip_duration(mono) == pytest.approx(0.1)
    assert data.get_clip_duration(str(tmp_path / "missing.wav")) == 0


def test_reverberate_identity_and_alignment():
    x = torch.randn(3, 4000)
    # A delayed unit impulse only shifts the signal, which is compensated by the direct path alignment
    rir = torch.zeros(500)
    rir[100] = 1.0
    y = data.reverberate(x, rir, rescale_amp="avg")
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4)

    # Echoes keep the average amplitude and the 1D input shape
    rir[300] = 0.5
    y = data.reverberate(x[0], rir, rescale_amp="avg")
    assert y.shape == (4000,)
    assert torch.isclose(y.abs().mean(), x[0].abs().mean(), rtol=1e-4)


@pytest.mark.parametrize("color, beta", [("white", 0), ("pink", 1), ("brown", 2), ("blue", -1), ("violet", -2)])
def test_generate_colored_noise_spectrum(color, beta):
    np.random.seed(0)
    n = 2**16
    noise = data.generate_colored_noise(n, color)
    assert noise.shape == (n,)
    assert np.abs(noise).max() == pytest.approx(1.0)

    # The log-log slope of the power spectrum is approximately -beta
    psd = np.abs(np.fft.rfft(noise))**2
    f = np.fft.rfftfreq(n)
    sel = (f > 0.001) & (f < 0.4)
    slope = np.polyfit(np.log(f[sel]), np.log(psd[sel]), 1)[0]
    assert slope == pytest.approx(-beta, abs=0.15)


def test_generate_colored_noise_invalid_color():
    with pytest.raises(ValueError):
        data.generate_colored_noise(100, "green")


def test_add_background_noise_snr(tmp_path):
    random_state = np.random.RandomState(0)
    paths = [_write_wav(tmp_path / f"bg{i}.wav", random_state.normal(0, 0.1, n))
             for i, n in enumerate([500, 16000, 40000])]
    augment = data.AddBackgroundNoise(paths, min_snr_in_db=10, max_snr_in_db=10, p=1.0)
    x = torch.from_numpy(random_state.normal(0, 0.2, (4, 8000)).astype(np.float32))
    y = augment(x)
    assert y.shape == x.shape
    noise = y - x
    snr = 20*torch.log10(x.pow(2).mean(dim=1).sqrt()/noise.pow(2).mean(dim=1).sqrt())
    assert torch.allclose(snr, torch.full_like(snr, 10.0), atol=0.01)

    # p=0 and an empty list of paths do nothing
    assert torch.equal(data.AddBackgroundNoise(paths, p=0.0)(x), x)
    assert torch.equal(data.AddBackgroundNoise([], p=1.0)(x), x)


def test_augment_clips(tmp_path):
    t = np.arange(12000)/16000
    clips = [_write_wav(tmp_path / f"clip{i}.wav", 0.5*np.sin(2*np.pi*(300 + 100*i)*t)) for i in range(5)]
    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    background = [_write_wav(bg_dir / "bg.wav", np.random.normal(0, 0.5, (20000, 2)))]
    rir = [_write_wav(tmp_path / "rir.wav", np.r_[np.zeros(10), 1.0, 0.2*np.random.randn(800)*np.exp(-np.arange(800)/100)])]
    probabilities = {k: 1.0 for k in ["SevenBandParametricEQ", "TanhDistortion", "PitchShift", "BandStopFilter",
                                      "AddColoredNoise", "AddBackgroundNoise", "Gain", "RIR"]}
    batches = list(data.augment_clips(clips, total_length=32000, batch_size=2, augmentation_probabilities=probabilities,
                                      background_clip_paths=background, RIR_paths=rir))
    assert [b.shape for b in batches] == [(2, 32000), (2, 32000), (1, 32000)]
    assert all(b.dtype == np.int16 for b in batches)


def test_mix_clips_batch(tmp_path):
    fg = [_write_wav(tmp_path / f"fg{i}.wav", 0.3*np.random.randn(8000)) for i in range(4)]
    bg = [_write_wav(tmp_path / f"bg{i}.wav", 0.1*np.random.randn(n)) for i, n in enumerate([16000, 32000, 50000])]
    rir = [_write_wav(tmp_path / "rir.wav", np.r_[1.0, np.zeros(100), 0.3])]
    batches = list(data.mix_clips_batch(fg, bg, combined_size=32000, batch_size=2, snr_low=0, snr_high=10,
                                        rirs=rir, generated_noise_augmentation=1.0, return_background_clips=True))
    assert len(batches) == 2
    for clips, labels, background in batches:
        assert clips.shape == (2, 32000) and clips.dtype == np.int16
        assert background.shape == (2, 32000)


def test_generate_adversarial_texts():
    np.random.seed(0)
    texts = data.generate_adversarial_texts("hey jarvis", N=20, include_partial_phrase=0, include_input_words=0)
    assert len(texts) > 0
    assert all(t != "hey jarvis" for t in texts)
    assert all(len(t.split()) >= 2 for t in texts)


def test_generate_adversarial_texts_without_phonemizer(monkeypatch):
    # Words that are not in the pronunciation dictionary are kept as-is when no phonemizer is available
    monkeypatch.setattr(data, "_load_deep_phonemizer", lambda: False)
    np.random.seed(0)
    texts = data.generate_adversarial_texts("hey zorblaxian", N=10, include_partial_phrase=0, include_input_words=0)
    assert len(texts) > 0
    assert all(t.split()[-1] == "zorblaxian" for t in texts)


def test_compute_features_is_atomic(tmp_path):
    from openwakeword.utils import compute_features_from_generator, download_models
    download_models(model_names=["alexa"])  # the feature models are always downloaded

    def generator():
        for _ in range(3):
            yield (np.random.randn(4, 32000)*1000).astype(np.int16)

    output = str(tmp_path / "features.npy")
    compute_features_from_generator(generator(), n_total=12, clip_duration=32000, output_file=output)
    assert os.path.exists(output)
    assert not os.path.exists(str(tmp_path / "features.incomplete.npy"))
    assert np.load(output).shape == (12, 16, 96)

    def failing_generator():
        yield (np.random.randn(4, 32000)*1000).astype(np.int16)
        raise RuntimeError("interrupted")

    output = str(tmp_path / "features2.npy")
    with pytest.raises(RuntimeError):
        compute_features_from_generator(failing_generator(), n_total=12, clip_duration=32000, output_file=output)
    assert not os.path.exists(output)


@pytest.mark.parametrize("name", ["features.npy", "happy.npy"])
def test_trim_mmap(tmp_path, monkeypatch, name):
    from numpy.lib.format import open_memmap
    monkeypatch.chdir(tmp_path)
    path = os.path.join(".", name)  # relative path, which was previously mangled by str.strip(".npy")
    arr = open_memmap(path, mode="w+", dtype=np.float32, shape=(10, 4, 3))
    arr[:7] = 1.0
    arr.flush()
    del arr
    data.trim_mmap(path)
    assert np.load(path).shape == (7, 4, 3)
    assert sorted(os.listdir(tmp_path)) == [name]


def test_trim_mmap_all_empty(tmp_path):
    path = str(tmp_path / "empty.npy")
    np.save(path, np.zeros((5, 2, 2), dtype=np.float32))
    data.trim_mmap(path)
    assert np.load(path).shape == (0, 2, 2)


def test_create_fixed_size_clip():
    x = np.ones(1000)
    assert data.create_fixed_size_clip(x, 4000).shape == (4000,)
    assert data.create_fixed_size_clip(x, 500).shape == (500,)
    assert data.create_fixed_size_clip(torch.ones(1000), 500).shape == (500,)


def test_resample_wav_files(tmp_path):
    t = np.arange(22050)/22050
    path = _write_wav(tmp_path / "a.wav", 0.5*np.sin(2*np.pi*440*t), sr=22050)
    stereo = str(tmp_path / "b.wav")
    sf.write(stereo, np.zeros((16000, 2), dtype=np.float32), 16000, subtype="PCM_16")
    ok = _write_wav(tmp_path / "c.wav", np.zeros(16000))
    assert data.resample_wav_files([path, stereo, ok]) == 2
    for p in (path, stereo, ok):
        info = sf.info(p)
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert sf.info(path).frames == 16000


def test_augment_clips_resamples(tmp_path):
    clip = _write_wav(tmp_path / "clip.wav", 0.5*np.sin(2*np.pi*300*np.arange(22050)/22050), sr=22050)
    probabilities = {k: 0.0 for k in ["SevenBandParametricEQ", "TanhDistortion", "PitchShift", "BandStopFilter",
                                      "AddColoredNoise", "AddBackgroundNoise", "Gain", "RIR"]}
    batch = next(data.augment_clips([clip], total_length=32000, batch_size=1, augmentation_probabilities=probabilities))
    assert batch.shape == (1, 32000)
    assert np.count_nonzero(batch) == pytest.approx(16000, abs=300)  # 1 s of audio at 16 khz
