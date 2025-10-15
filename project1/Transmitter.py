import numpy as np
import crcmod
import scipy.signal
from CONFIG import Config
import random


class Transmitter:
    def __init__(self, config: Config):
        self.config = config
        self.t_symbol = np.linspace(
            0,
            config.SAMPLES_PER_SYMBOL / config.SAMPLE_RATE,
            config.SAMPLES_PER_SYMBOL,
            endpoint=False,
        )
        self.carrier_0 = np.sin(2 * np.pi * config.CARRIER_FREQ * self.t_symbol)
        self.carrier_1 = np.sin(2 * np.pi * config.CARRIER_FREQ * self.t_symbol + np.pi)
        self.preamble = self._generate_preamble()
        # self.crc16 = crcmod.predefined.mkCrcFun("crc-16-buypass")

    def _generate_preamble(self) -> np.ndarray:
        """Generates the triangle chirp signal for synchronization (matching MATLAB)."""

        dt = 1.0 / self.config.SAMPLE_RATE
        t = np.arange(0, self.config.PREAMBLE_SAMPLES) * dt

        half_samples = self.config.PREAMBLE_SAMPLES // 2
        f1 = np.linspace(
            self.config.PREAMBLE_FREQ_START, self.config.PREAMBLE_FREQ_MID, half_samples
        )
        f2 = np.linspace(
            self.config.PREAMBLE_FREQ_MID, self.config.PREAMBLE_FREQ_END, half_samples
        )
        f_sweep = np.concatenate([f1, f2])
        phase = 2 * np.pi * np.cumsum(f_sweep) * dt
        preamble = np.sin(phase)

        return preamble.astype(np.float32)

    def create_frame(self, payload_bits: np.ndarray):
        preamble = self.preamble.copy()
        payload_len = len(payload_bits)
        # for i in range(payload_len // 100 + 1):
        #     if i == payload_len // 100:
        #         frame_slice = payload_bits[i * 100 :]
        #     else:
        #         frame_slice = payload_bits[i * 100 : (i + 1) * 100]
        #     frame_signal = self._bits_to_signal(frame_slice)

        # output_track = np.array([], dtype=np.float32)
        output_track = preamble
        # for bit in payload_bits:
        #     if bit == 0:
        #         output_track = np.concatenate((output_track, self.carrier_0))
        #         output_track = np.concatenate(
        #             (output_track, np.zeros(random.randint(0, 100)))
        #         )
        #     else:
        #         output_track = np.concatenate((output_track, self.carrier_1))
        #         output_track = np.concatenate(
        #             (output_track, np.zeros(random.randint(0, 100)))
        #         )
        return output_track.astype(np.float32)
