import numpy as np
import crcmod
import scipy.signal
from CONFIG import Config
from Transmitter import Transmitter


class ReceiverState:
    WAIT_PREAMBLE = 0
    RECEIVE_HEADER = 1
    RECEIVE_PAYLOAD = 2
    FRAME_RECEIVED = 3


class Receiver:
    def __init__(self, config: Config):
        self.config = config
        self.state = ReceiverState.WAIT_PREAMBLE

        self.sample_index = 0

        self.preamble_buffer = np.zeros(config.PREAMBLE_SAMPLES, dtype=np.float32)

        self.preamble_template = Transmitter(
            config
        ).preamble  # Generate the same preamble

        # self.rx_buffer = np.zeros(
        #     int(config.RX_BUFFER_DURATION * config.SAMPLE_RATE), dtype=np.float32
        # )
        self.buffer_ptr = 0
        # self.crc16 = crcmod.predefined.mkCrcFun("crc-16-buypass")

        # Create reference symbols for demodulation
        t_symbol = np.linspace(
            0,
            config.SAMPLES_PER_SYMBOL / config.SAMPLE_RATE,
            config.SAMPLES_PER_SYMBOL,
            endpoint=False,
        )
        self.ref_symbol_0 = np.sin(2 * np.pi * config.CARRIER_FREQ * t_symbol).astype(
            np.float32
        )
        self.ref_symbol_1 = np.sin(
            2 * np.pi * config.CARRIER_FREQ * t_symbol + np.pi
        ).astype(np.float32)

    def _find_preamble(self) -> int:

        if len(self.rx_buffer) < len(self.preamble_template):
            return -1
        """Finds the preamble in the buffer using correlation. Returns start index or -1."""
        corr = scipy.signal.correlate(
            self.rx_buffer, self.preamble_template, mode="valid", method="fft"
        )
        corr_normalized = corr / np.max(np.abs(corr)) if np.max(corr) > 0 else corr

        peak_idx = np.argmax(corr_normalized)
        if corr_normalized[peak_idx] > self.config.PREAMBLE_CORR_THRESHOLD:
            return peak_idx
        return -1

    def receive_sample(self, block):
        """Process incoming audio samples."""
        sliding_power_sum = 0
        start_index = 0
        syncPower_localMax = 0

        for sample in block:
            self.sample_index += 1
            i = self.sample_index
            self.preamble_buffer[:-1] = self.preamble_buffer[1:]
            self.preamble_buffer[-1] = sample

            sliding_power_sum = sliding_power_sum * (1 - 1 / 64) + sample**2 / 64
            corr_power = sum(self.preamble_buffer * self.preamble_template) / 200

            if (
                (corr_power > sliding_power_sum * 2)
                and (corr_power > syncPower_localMax)
                and (corr_power > 0.05)
            ):
                syncPower_localMax = corr_power
                print(f"Sync power local max updated: {syncPower_localMax}")
                start_index = i
            elif (i - start_index > 200) and (start_index != 0):
                print(f"Preamble detected at sample index: {start_index}")
                syncPower_localMax = 0
                self.preamble_buffer = np.zeros(len(self.preamble_buffer))
                self.state = ReceiverState.RECEIVE_HEADER
