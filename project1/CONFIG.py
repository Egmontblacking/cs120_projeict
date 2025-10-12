class Config:
    """Class to hold all physical layer and application parameters."""

    # --- Audio Settings ---
    CHANNELS = 1  # Mono

    OUTPUT_WAV_FILENAME = "recording.wav"

    # --- Physical Layer Parameters (matching MATLAB) ---
    CARRIER_FREQ = 10000.0  # Hz, 10kHz (matching MATLAB fc)
    SAMPLES_PER_BIT = 44  # 44 samples per bit (matching MATLAB)

    PREAMBLE_SAMPLES = 4400  # 4400 samples
    PREAMBLE_FREQ_START = 2000  # Hz (10kHz - 8kHz)
    PREAMBLE_FREQ_MID = 10000  # Hz
    PREAMBLE_FREQ_END = 2000  # Hz (triangle wave: up then down)

    # Frame structure (matching MATLAB)
    FRAME_ID_BITS = 8  # Frame ID
    FRAME_DATA_BITS = 100  # Data bits
    FRAME_TOTAL_BITS = 108  # ID + Data
    CRC_BITS = 8  # CRC-8
    FRAME_WITH_CRC_BITS = 116  # Total bits per frame

    NUM_FRAMES = 100  # Number of frames to transmit
    INTER_FRAME_GAP_MAX = 100  # Max random gap between frames

    # --- Receiver Settings ---
    PREAMBLE_CORR_THRESHOLD = 0.05  # Absolute threshold
    POWER_RATIO_THRESHOLD = 2.0  # Power ratio threshold
    SYNC_COOLDOWN = 200  # Samples to wait after peak detection

    # --- Application ---
    INPUT_FILENAME = "input.txt"
    OUTPUT_FILENAME = "output.txt"
    RECORDING_FILENAME = "recording.wav"

    def __init__(self, sample_rate, block_size):
        self.SAMPLE_RATE = sample_rate
        self.BLOCK_SIZE = block_size
        # Calculate carrier samples for entire second
        self.CARRIER_SAMPLES = int(sample_rate)
