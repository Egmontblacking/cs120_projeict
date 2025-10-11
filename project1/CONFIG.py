class Config:
    """Class to hold all physical layer and application parameters."""

    # --- Audio Settings ---
    CHANNELS = 1  # Mono

    OUTPUT_WAV_FILENAME = "recording.wav"

    # --- Physical Layer Parameters ---
    BAUD_RATE = 2000  # Symbols per second
    CARRIER_FREQ = 8000.0  # Hz
    # SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BAUD_RATE)

    PREAMBLE_SAMPLES = 440  # 440 samples
    PREAMBLE_FREQ_START = 2000  # Hz (10kHz - 8kHz)
    PREAMBLE_FREQ_MID = 10000  # Hz
    PREAMBLE_FREQ_END = 2000  # Hz (triangle wave: up then down)

    # Header fields length in bits
    LEN_FIELD_BITS = 16  # For payload length
    CRC_FIELD_BITS = 16  # For CRC-16

    # --- Receiver Settings ---
    # Buffer duration for processing incoming audio
    RX_BUFFER_DURATION = 2.0  # seconds
    # Correlation threshold to detect a preamble
    PREAMBLE_CORR_THRESHOLD = 0.5

    # --- FEC (Task 4) ---
    FEC_ENABLED = True  # Set to True to enable Hamming(7,4) code

    # --- Application ---
    INPUT_FILENAME = "INPUT.txt"
    OUTPUT_FILENAME = "OUTPUT.txt"
    RECORDING_FILENAME = "recording.wav"
    PAYLOAD_BITS = 10000

    def __init__(self, sample_rate, block_size):
        self.SAMPLE_RATE = sample_rate
        self.BLOCK_SIZE = block_size
        self.SAMPLES_PER_SYMBOL = int(self.SAMPLE_RATE / self.BAUD_RATE)
