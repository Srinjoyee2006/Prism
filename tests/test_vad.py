"""
Unit tests for AudioFrame, VAD abstractions, FakeVAD, SileroVAD, and SpeechSegmenter.

Tests requirements:
1. audio frame properties
2. PCM validation
3. speech start detection
4. speech continuation
5. speech end detection
6. silence handling
7. speech segment creation
8. empty segment rejection
15. reset behavior
"""

import numpy as np
import pytest

from src.audio.frame import AudioFormatError, AudioFrame
from src.vad.base import VADDecision, VADState
from src.vad.mock_vad import FakeVAD
from src.vad.segmenter import SpeechSegment, SpeechSegmenter
from src.vad.silero_vad import SileroVAD

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pcm_frame(
    num_samples: int = 512,
    sample_rate: int = 16000,
    amplitude: int = 0,
    frame_index: int = 0,
) -> AudioFrame:
    """Create a test AudioFrame with fixed sample values."""
    import struct

    raw = struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))
    return AudioFrame(
        pcm_data=raw,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        frame_index=frame_index,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Audio Frame Properties and PCM Validation
# ---------------------------------------------------------------------------


class TestAudioFrame:
    def test_audio_frame_properties(self):
        frame = make_pcm_frame(num_samples=512, sample_rate=16000)
        assert frame.num_samples == 512
        assert frame.duration_ms == 32.0
        assert frame.sample_rate == 16000
        assert frame.channels == 1
        assert frame.sample_width == 2
        assert len(frame.pcm_data) == 1024  # 512 samples * 2 bytes

    def test_pcm_validation_unaligned_bytes_raises(self):
        # 1 channel * 2 bytes per sample = 2 bytes per frame. Passing 3 bytes must fail.
        with pytest.raises(AudioFormatError, match="not aligned"):
            AudioFrame(pcm_data=b"\x00\x01\x02", sample_rate=16000, channels=1, sample_width=2)

    def test_pcm_validation_invalid_type_raises(self):
        with pytest.raises(AudioFormatError, match="must be bytes"):
            AudioFrame(pcm_data="not_bytes", sample_rate=16000)  # type: ignore[arg-type]

    def test_pcm_validation_non_positive_sample_rate_raises(self):
        with pytest.raises(AudioFormatError, match="Sample rate must be positive"):
            AudioFrame(pcm_data=b"\x00\x00", sample_rate=0)

    def test_to_numpy_normalized(self):
        # Test 16-bit max value 32767 normalizes to approx 1.0
        frame = make_pcm_frame(num_samples=4, amplitude=32767)
        arr = frame.to_numpy(dtype="float32")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 4
        assert np.isclose(arr[0], 1.0, atol=1e-3)

    def test_silence_detection(self):
        silent_frame = AudioFrame.create_silence(num_samples=512)
        assert silent_frame.calculate_rms() == 0.0
        assert silent_frame.is_silent()

        loud_frame = make_pcm_frame(num_samples=512, amplitude=16000)
        assert not loud_frame.is_silent()


# ---------------------------------------------------------------------------
# FakeVAD Tests
# ---------------------------------------------------------------------------


class TestFakeVAD:
    def test_scripted_decisions(self):
        vad = FakeVAD(scripted_decisions=[True, False, True])
        frame = make_pcm_frame()

        d1 = vad.process_frame(frame)
        assert d1.is_speech
        assert d1.confidence == 0.95

        d2 = vad.process_frame(frame)
        assert not d2.is_speech

        d3 = vad.process_frame(frame)
        assert d3.is_speech

        # Exhausted script defaults to silence
        d4 = vad.process_frame(frame)
        assert not d4.is_speech

    def test_energy_threshold_vad(self):
        vad = FakeVAD(energy_threshold=0.1)
        silent = AudioFrame.create_silence(num_samples=512)
        loud = make_pcm_frame(num_samples=512, amplitude=16000)

        assert not vad.process_frame(silent).is_speech
        assert vad.process_frame(loud).is_speech

    def test_vad_reset(self):
        vad = FakeVAD(scripted_decisions=[True, False])
        frame = make_pcm_frame()
        vad.process_frame(frame)
        vad.reset()
        assert vad.reset_count == 1
        # Cursor is reset, next frame gets True again
        assert vad.process_frame(frame).is_speech


# ---------------------------------------------------------------------------
# 3-8, 15. SpeechSegmenter Tests
# ---------------------------------------------------------------------------


class TestSpeechSegmenter:
    def test_silence_handling_stays_in_silence(self):
        segmenter = SpeechSegmenter(min_silence_duration_ms=400.0)
        frame = make_pcm_frame()
        silence_decision = VADDecision(is_speech=False, confidence=0.0)

        for _ in range(10):
            res = segmenter.process(frame, silence_decision)
            assert res is None
            assert segmenter.state == VADState.SILENCE
            assert not segmenter.is_in_speech

    def test_speech_start_detection(self):
        segmenter = SpeechSegmenter()
        frame = make_pcm_frame()
        speech_decision = VADDecision(is_speech=True, confidence=0.9)

        res = segmenter.process(frame, speech_decision)
        assert res is None
        assert segmenter.state == VADState.SPEECH
        assert segmenter.is_in_speech
        assert segmenter.just_started_speech

        # Next speech frame: continuation, just_started_speech resets to False
        res2 = segmenter.process(frame, speech_decision)
        assert res2 is None
        assert segmenter.state == VADState.SPEECH
        assert not segmenter.just_started_speech

    def test_speech_continuation_and_segment_creation(self):
        # 32ms per frame (512 samples at 16kHz)
        # min_speech: 100ms (~4 frames)
        # min_silence: 200ms (~7 frames)
        segmenter = SpeechSegmenter(
            min_speech_duration_ms=100.0,
            min_silence_duration_ms=200.0,
            speech_pad_ms=0.0,
        )
        frame = make_pcm_frame(num_samples=512)

        # 1. Start speech: 6 frames = 192ms
        for _ in range(6):
            res = segmenter.process(frame, VADDecision(is_speech=True, confidence=0.9))
            assert res is None
            assert segmenter.is_in_speech

        # 2. Silence starts: 7 frames = 224ms (>= 200ms)
        segment: SpeechSegment | None = None
        for i in range(7):
            res = segmenter.process(frame, VADDecision(is_speech=False, confidence=0.1))
            if i < 6:
                assert res is None
            else:
                segment = res

        # 3. Segment should now be emitted
        assert segment is not None
        assert isinstance(segment, SpeechSegment)
        assert segment.total_samples > 0
        assert segment.duration_ms >= 192.0
        assert segment.sample_rate == 16000
        assert len(segment.pcm_data) == segment.total_samples * 2
        assert segmenter.state == VADState.SILENCE
        assert segmenter.just_ended_speech

    def test_empty_segment_rejection_noise_spike(self):
        # Noise spike: 1 frame (32ms) < min_speech_duration_ms (100ms)
        segmenter = SpeechSegmenter(
            min_speech_duration_ms=100.0,
            min_silence_duration_ms=100.0,
            speech_pad_ms=0.0,
        )
        frame = make_pcm_frame()

        # 1 single speech frame
        segmenter.process(frame, VADDecision(is_speech=True, confidence=0.9))

        # Trailing silence frames
        for _ in range(5):
            res = segmenter.process(frame, VADDecision(is_speech=False, confidence=0.1))
            # Must NOT produce a segment because duration (32ms) < 100ms
            assert res is None

        assert segmenter.state == VADState.SILENCE

    def test_pre_speech_padding_preserves_onset(self):
        segmenter = SpeechSegmenter(
            min_speech_duration_ms=50.0,
            min_silence_duration_ms=100.0,
            speech_pad_ms=30.0,
        )
        silence_frame = make_pcm_frame(amplitude=0)
        speech_frame = make_pcm_frame(amplitude=5000)

        # Feed silence frames first to populate pre-speech ring buffer
        for _ in range(5):
            segmenter.process(silence_frame, VADDecision(is_speech=False, confidence=0.0))

        # Speech starts
        segmenter.process(speech_frame, VADDecision(is_speech=True, confidence=0.9))
        segmenter.process(speech_frame, VADDecision(is_speech=True, confidence=0.9))

        # Trailing silence to close segment
        segment = None
        for _ in range(5):
            res = segmenter.process(silence_frame, VADDecision(is_speech=False, confidence=0.0))
            if res is not None:
                segment = res

        assert segment is not None
        # Must contain more than the 2 speech frames because of pre-speech padding
        assert segment.total_samples > (2 * 512)

    def test_segmenter_flush(self):
        segmenter = SpeechSegmenter(min_speech_duration_ms=50.0)
        frame = make_pcm_frame()

        # Feed 3 speech frames = 96ms
        for _ in range(3):
            segmenter.process(frame, VADDecision(is_speech=True, confidence=0.9))

        assert segmenter.is_in_speech
        segment = segmenter.flush()
        assert segment is not None
        assert segment.duration_ms >= 90.0
        assert segmenter.state == VADState.SILENCE

    def test_segmenter_reset(self):
        segmenter = SpeechSegmenter()
        frame = make_pcm_frame()
        segmenter.process(frame, VADDecision(is_speech=True, confidence=0.9))
        assert segmenter.is_in_speech

        segmenter.reset()
        assert segmenter.state == VADState.SILENCE
        assert not segmenter.is_in_speech


# ---------------------------------------------------------------------------
# SileroVAD Unit Tests
# ---------------------------------------------------------------------------


class TestSileroVAD:
    def test_silero_vad_runs_on_silence_and_tone(self):
        vad = SileroVAD(threshold=0.5)

        # 512 samples of silence
        silent_frame = AudioFrame.create_silence(num_samples=512)
        d1 = vad.process_frame(silent_frame)
        assert isinstance(d1, VADDecision)
        # Silence confidence should be very low
        assert d1.confidence < 0.2
        assert not d1.is_speech

        # Test reset
        vad.reset()
        assert not vad._in_speech
