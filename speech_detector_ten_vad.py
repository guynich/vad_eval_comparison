"""Speech detection using TEN VAD model."""

import numpy as np
from ten_vad import TenVad


class SpeechDetector:
    """Wraps TEN VAD mode with a chunk size of 512 samples."""
    CHUNK_SIZES = {16000: 512}

    def __init__(self, rate: int = 16000):
        if rate not in self.CHUNK_SIZES.keys():
            raise ValueError(f"SpeechDetector for TEN VAD is not configured for {rate} Hz")

        self.rate = rate
        self.chunk_size = self.CHUNK_SIZES[rate]
        self.model = TenVad(hop_size=self.chunk_size)

    def __call__(self, audio_chunk) -> float:
        # TEN VAD requires audio data in int16 format.
        audio = audio_chunk.copy()
        if type(audio[0]) is not np.int16:
            audio = (audio_chunk * (2 ** 15 - 1)).astype(np.int16)

        prob_value, flag_value = self.model.process(audio)
        return prob_value

    def get_name(self) -> str:
        """Get model name."""
        return "TEN VAD"

    def reset(self) -> None:
        """Reset model state."""
        for _ in range(3):
            self.__call__(np.zeros(self.chunk_size, dtype=np.int16))
