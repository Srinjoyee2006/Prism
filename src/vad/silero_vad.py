"""
Local Silero VAD v6 implementation using ONNX Runtime.

Runs local, zero-cost, neural voice activity detection on CPU without PyTorch.
Maintains stateful RNN recurrent hidden states across streaming audio frames.
"""

import logging
from pathlib import Path

import numpy as np

from src.audio.frame import AudioFrame
from src.vad.base import VADDecision, VoiceActivityDetector

logger = logging.getLogger(__name__)


class SileroVAD(VoiceActivityDetector):
    """Voice Activity Detector powered by Silero VAD v6 ONNX model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        sample_rate: int = 16000,
    ) -> None:
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.sample_rate = sample_rate

        self._session = None
        self._init_session(model_path)

        # Internal RNN states: shape [1, 1, 128]
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        # Context buffer: 64 samples
        self._context_size = 64
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

        # Hysteresis state tracking
        self._in_speech: bool = False

    def _init_session(self, model_path: str | Path | None) -> None:
        """Initialize the ONNX Runtime session."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for SileroVAD. Please install it."
            ) from exc

        # Find the bundled model if no path is provided
        resolved_path = None
        if model_path is not None:
            resolved_path = Path(model_path)
        else:
            try:
                import faster_whisper.vad as fw_vad

                bundled = Path(fw_vad.get_assets_path()) / "silero_vad_v6.onnx"
                if bundled.exists():
                    resolved_path = bundled
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not resolve bundled faster-whisper assets: %s", exc)

        if resolved_path is None or not resolved_path.exists():
            raise FileNotFoundError(
                f"Silero VAD ONNX model not found at {resolved_path}. "
                "Ensure faster-whisper or silero_vad_v6.onnx is installed."
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 3  # Warning level only

        self._session = ort.InferenceSession(
            str(resolved_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        logger.debug("SileroVAD initialized from %s", resolved_path)

    def process_frame(self, frame: AudioFrame) -> VADDecision:
        """Evaluate a frame of audio through Silero VAD ONNX.

        Expects 512 samples at 16kHz (32ms).
        """
        # Convert int16 PCM to float32 normalized in [-1.0, 1.0]
        audio_float = frame.to_numpy(dtype="float32")

        # If audio is not 512 samples, pad or truncate
        target_samples = 512
        if len(audio_float) < target_samples:
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[: len(audio_float)] = audio_float
            audio_float = padded
        elif len(audio_float) > target_samples:
            audio_float = audio_float[:target_samples]

        # Concatenate 64 context samples with 512 audio samples = 576 samples
        input_tensor = np.concatenate(
            [self._context, audio_float.reshape(1, target_samples)], axis=1
        ).astype(np.float32)

        # Update context buffer with the trailing 64 samples of current audio
        self._context = audio_float[-self._context_size :].reshape(1, self._context_size)

        # Run ONNX session
        outputs = self._session.run(
            None,
            {"input": input_tensor, "h": self._h, "c": self._c},
        )
        prob = float(outputs[0][0])
        self._h = outputs[1]
        self._c = outputs[2]

        # Apply hysteresis thresholding
        if not self._in_speech and prob >= self.threshold:
            self._in_speech = True
        elif self._in_speech and prob < self.neg_threshold:
            self._in_speech = False

        return VADDecision(
            is_speech=self._in_speech,
            confidence=prob,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
        )

    def reset(self) -> None:
        """Reset internal recurrent states and context buffer."""
        self._h.fill(0)
        self._c.fill(0)
        self._context.fill(0)
        self._in_speech = False
