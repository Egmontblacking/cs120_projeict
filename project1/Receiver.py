import numpy as np
import crcmod
import scipy.signal
from scipy.ndimage import uniform_filter1d
from CONFIG import Config
from Transmitter import Transmitter

from scipy.signal import correlate, spectrogram
import matplotlib.pyplot as plt


class ReceiverState:
    WAIT_PREAMBLE = 0
    RECEIVE_FRAME = 1


class Receiver:
    def __init__(self, config: Config):
        self.config = config
        self.state = ReceiverState.WAIT_PREAMBLE

        self.sample_index = -config.BLOCK_SIZE
        self.preamble_buffer = np.zeros(
            config.PREAMBLE_SAMPLES * 1000, dtype=np.float32
        )
        self.preamble_template = Transmitter(config).preamble

        # Generate continuous carrier (matching MATLAB)
        t_carrier = np.arange(0, config.SAMPLE_RATE) / config.SAMPLE_RATE
        self.carrier = np.sin(2 * np.pi * config.CARRIER_FREQ * t_carrier)

        # CRC-8 (same as transmitter)
        self.crc8_func = crcmod.mkCrcFun(0x1D5, initCrc=0, xorOut=0)

        # State variables
        self.power = 0.0
        self.power_alpha = 1.0 / 64.0

        self.decode_fifo = np.array([], dtype=np.float32)
        self.expected_samples = config.FRAME_WITH_CRC_BITS * config.SAMPLES_PER_BIT

        self.start_index = 0
        self.sync_power_local_max = 0.0

        # Statistics
        self.correct_frames = 0
        self.total_frames = 0

        # Store all received data
        self.all_received_bits = []

    def _demodulate_frame(self, signal: np.ndarray) -> np.ndarray:
        """Demodulate BPSK signal (matching MATLAB method)."""
        # Remove carrier
        demod_signal = signal * self.carrier[: len(signal)]

        # Smooth with window of 10 (matching MATLAB smooth(...,10))
        smoothed = uniform_filter1d(demod_signal, size=10, mode="nearest")

        # Sample at bit centers
        num_bits = self.config.FRAME_WITH_CRC_BITS
        bits = np.zeros(num_bits, dtype=np.uint8)

        for j in range(num_bits):
            # Sample from 10 to 30 within each bit period (matching MATLAB)
            start_idx = 10 + j * self.config.SAMPLES_PER_BIT
            end_idx = 30 + j * self.config.SAMPLES_PER_BIT

            if end_idx <= len(smoothed):
                bit_power = np.sum(smoothed[start_idx:end_idx])
                bits[j] = 1 if bit_power > 0 else 0

        return bits

    def receive_sample(self, block):
        """Process incoming audio samples (matching MATLAB logic)."""
        block_len = len(block)

        self.preamble_buffer[:-block_len] = self.preamble_buffer[block_len:]
        self.preamble_buffer[-block_len:] = block
        self.sample_index += block_len

        if self.state == ReceiverState.WAIT_PREAMBLE:

            correlation = correlate(
                self.preamble_buffer, self.preamble_template, mode="same"
            )

            corr_abs = np.abs(correlation)
            noise_median = np.median(corr_abs)
            noise_std = np.std(corr_abs)

            k = 10
            threshold = max(0.1, noise_median + k * noise_std)

            tmp_detected_index = np.argmax(np.abs(correlation))
            if corr_abs[tmp_detected_index] > threshold:
                print(corr_abs[tmp_detected_index], threshold)
                detected_index = tmp_detected_index
                print(
                    "Detected preamble at sample index:",
                    self.sample_index + detected_index,
                )
                # self.state = ReceiverState.RECEIVE_FRAME
                return
                self.start_index = self.sample_index + detected_index

            # i += detected_index

            # print(np.abs(correlation[detected_index]))

        elif self.state == ReceiverState.RECEIVE_FRAME:
            # Accumulate samples for decoding
            self.decode_fifo = np.append(self.decode_fifo, sample)

            # Check if we have enough samples
            if len(self.decode_fifo) >= self.expected_samples:
                # Decode the frame
                frame_bits = self._demodulate_frame(
                    self.decode_fifo[: self.expected_samples]
                )

                # Extract frame data and CRC
                frame_data = frame_bits[: self.config.FRAME_TOTAL_BITS]
                received_crc_bits = frame_bits[self.config.FRAME_TOTAL_BITS :]

                # Compute CRC
                computed_crc = self.crc8_func(frame_data.tobytes())
                received_crc = int("".join(map(str, received_crc_bits)), 2)

                # Extract frame ID
                frame_id = int("".join(map(str, frame_data[:8])), 2)

                self.total_frames += 1

                # Check CRC
                if computed_crc == received_crc:
                    print(f"✓ Frame {frame_id} correct (CRC: {computed_crc})")
                    self.correct_frames += 1
                    # Store received data
                    self.all_received_bits.extend(frame_data)
                else:
                    print(
                        f"✗ Frame {frame_id} CRC failed (rx={received_crc}, calc={computed_crc})"
                    )

                # Reset to WAIT_PREAMBLE state
                self.state = ReceiverState.WAIT_PREAMBLE
                self.start_index = 0
                self.decode_fifo = np.array([], dtype=np.float32)

    def get_statistics(self):
        """Return reception statistics."""
        return {
            "correct_frames": self.correct_frames,
            "total_frames": self.total_frames,
            "all_bits": self.all_received_bits,
        }
