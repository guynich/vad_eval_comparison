"""Speech detection using Silero VAD model."""

import torch

torch.set_num_threads(1)


class SpeechDetector:
    """Wraps Silero VAD model (onnx version)."""

    CHUNK_SIZES = {16000: 512, 8000: 256}

    def __init__(self, rate: int = 16000):
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=True,
            onnx=True,
            trust_repo=True,
        )

        if rate not in self.model.sample_rates:
            raise ValueError(f"Silero VAD does not support {rate} Hz")

        self.rate = rate
        self.chunk_size = self.CHUNK_SIZES[rate]
        self.reset()

    @torch.no_grad()
    def __call__(self, audio_chunk) -> float:
        """Process audio chunk and return smoothed VAD probability."""
        if len(audio_chunk) != self.chunk_size:
            raise ValueError("Unexpected chunk size")

        return self.model(torch.Tensor(audio_chunk), self.rate).item()

    def get_name(self) -> str:
        """Get model name."""
        return "Silero VAD"

    def reset(self) -> None:
        """Reset model state."""
        self.model.reset_states()
