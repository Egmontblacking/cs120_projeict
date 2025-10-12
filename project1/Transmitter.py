import numpy as np
import crcmod
import scipy.signal
from CONFIG import Config
import random


class Transmitter:
    def __init__(self, config: Config):
        self.config = config

        # Generate continuous carrier for entire second (matching MATLAB)
        t_carrier = np.arange(0, config.SAMPLE_RATE) / config.SAMPLE_RATE
        self.carrier = np.sin(2 * np.pi * config.CARRIER_FREQ * t_carrier)

        self.preamble = self._generate_preamble()

        # CRC-8: x^8+x^7+x^5+x^2+x+1 = 0x1D5 (matching MATLAB)
        self.crc8_func = crcmod.mkCrcFun(0x1D5, initCrc=0, xorOut=0)

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

    def _modulate_bits(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits using BPSK (matching MATLAB modulation)."""
        num_bits = len(bits)
        frame_wave = np.zeros(num_bits * self.config.SAMPLES_PER_BIT, dtype=np.float32)

        for j in range(num_bits):
            start_idx = j * self.config.SAMPLES_PER_BIT
            end_idx = start_idx + self.config.SAMPLES_PER_BIT
            # BPSK: bit*2-1, so 0->-1, 1->+1
            amplitude = bits[j] * 2 - 1
            frame_wave[start_idx:end_idx] = self.carrier[start_idx:end_idx] * amplitude

        return frame_wave

    def create_frame(self, payload_bits: np.ndarray):
        """Create complete transmission from payload_bits (matching MATLAB)."""
        np.random.seed(1)  # Match MATLAB seed

        output_track = np.array([], dtype=np.float32)

        # Calculate number of frames based on payload_bits length
        # Each frame contains FRAME_DATA_BITS (100 bits including ID)
        num_frames = len(payload_bits) // self.config.FRAME_DATA_BITS

        print(f"Creating {num_frames} frames from {len(payload_bits)} bits")

        # Generate frames
        for frame_id in range(1, num_frames + 1):
            # Get FRAME_DATA_BITS for this frame from payload
            start_idx = (frame_id - 1) * self.config.FRAME_DATA_BITS
            end_idx = start_idx + self.config.FRAME_DATA_BITS
            frame_bits = payload_bits[start_idx:end_idx]

            # Add CRC-8
            crc_val = self.crc8_func(frame_bits.tobytes())
            crc_bits = np.array(
                [int(b) for b in format(crc_val, "08b")], dtype=np.uint8
            )

            # Total: 108 bits
            frame_with_crc = np.concatenate([frame_bits, crc_bits])

            # Modulate
            frame_wave = self._modulate_bits(frame_with_crc)

            # Add preamble
            frame_wave_pre = np.concatenate([self.preamble, frame_wave])

            # Add random inter-frame gaps (matching MATLAB)
            gap_before = np.zeros(
                int(np.random.rand() * self.config.INTER_FRAME_GAP_MAX),
                dtype=np.float32,
            )
            gap_after = np.zeros(
                int(np.random.rand() * self.config.INTER_FRAME_GAP_MAX),
                dtype=np.float32,
            )

            # Append to output track
            output_track = np.concatenate(
                [output_track, gap_before, frame_wave_pre, gap_after]
            )

            if frame_id % 10 == 0:
                print(f"Generated frame {frame_id}/{num_frames}")

        print(
            f"Total transmission length: {len(output_track)/self.config.SAMPLE_RATE:.2f} seconds"
        )
        return output_track.astype(np.float32)

    # def create_frame(self, payload_bits: np.ndarray):
    #     return self._generate_preamble()
